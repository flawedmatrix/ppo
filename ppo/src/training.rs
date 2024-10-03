use dfdx::prelude::*;
use ndarray::{ArrayView1, Axis};
use tracing::{info, span, Level};

use crate::{
    common::{ExperienceBuffer, VecRunner},
    model::{ExperienceBatcher, Learner, ModelConfig, PolicyNetworkConfig, TrainingStats},
    Environment,
};

#[derive(Debug, Clone, Copy)]
pub struct TrainingConfig {
    /// Number of environments to run in parallel for each training pass
    pub num_envs: usize,

    /// Number of action steps to train for in each update
    pub num_steps: usize,

    /// Number of training passes for the model
    pub num_epochs: usize,

    /// Number of iterations during the model training pass
    pub num_train_iterations: usize,

    /// Number of experiences per batch of data used in a single iteration
    /// during the model training pass
    pub batch_size: usize,

    /// Learning rate for the model
    pub lr: f64,

    pub model_config: ModelConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_envs: 160,
            num_steps: 32,
            num_epochs: 15000,
            num_train_iterations: 4,
            batch_size: 1280,
            lr: 3e-4,
            model_config: ModelConfig::default(),
        }
    }
}

impl TrainingConfig {
    pub fn new(model_config: ModelConfig) -> Self {
        Self {
            model_config,
            ..Default::default()
        }
    }

    pub fn with_num_envs(mut self, num_envs: usize) -> Self {
        self.num_envs = num_envs;
        self
    }

    pub fn with_num_steps(mut self, num_steps: usize) -> Self {
        self.num_steps = num_steps;
        self
    }

    pub fn with_num_epochs(mut self, num_epochs: usize) -> Self {
        self.num_epochs = num_epochs;
        self
    }

    pub fn with_num_train_iterations(mut self, num_train_iterations: usize) -> Self {
        self.num_train_iterations = num_train_iterations;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }
}

// Given targets and predicted values, compute a metric to determine how good
// the prediction is.
fn explained_variance(predictions: ArrayView1<f32>, targets: ArrayView1<f32>) -> f32 {
    let target_variance = targets.var_axis(Axis(0), 0.).into_scalar();
    if target_variance == -1.0 {
        f32::NAN
    } else {
        let diff = targets.into_owned() - predictions;
        let diff_variance = diff.var_axis(Axis(0), 0.).into_scalar();
        0.0 - (diff_variance / target_variance)
    }
}

#[cfg(feature = "dfdx-cuda")]
type Dev = Cuda;
#[cfg(not(feature = "dfdx-cuda"))]
type Dev = Cpu;

pub fn train<T, P, const NUM_ENVS: usize, const OBS_SIZE: usize, const NUM_ACTIONS: usize>(
    init_env_state: T,
    config: TrainingConfig,
    model_path: P,
) where
    T: Environment<OBS_SIZE, NUM_ACTIONS>,
    P: AsRef<std::path::Path>,
{
    let mut exp_buf: ExperienceBuffer<NUM_ENVS, OBS_SIZE> = ExperienceBuffer::new(config.num_steps);

    let dev = Dev::default();
    dev.enable_cache();

    // let model_path = model_path.as_ref();

    // if model_path.exists() {
    //     info!("Loading checkpoint from path {:?}", model_path);
    //     let record = recorder
    //         .load(model_path.to_path_buf(), &device)
    //         .expect("Should be able to load the model weights from the provided file");
    //     learner.model = learner.model.load_record(record);
    // }

    info!("Instantiating model with config {config:?}");
    let mut learner = Learner::<OBS_SIZE, 1024, NUM_ACTIONS, _>::new(config, dev);
    info!("Model instantiated.");

    // let checkpoint_path = if model_path.is_file() && model_path.parent().is_some_and(|p| p.exists())
    // {
    //     info!(
    //         "Storing checkpoints in parent path of model : {}",
    //         model_path.parent().unwrap().display()
    //     );
    //     model_path.parent().unwrap().to_owned()
    // } else if model_path.is_dir() && model_path.exists() {
    //     info!("Storing checkpoints in path: {}", model_path.display());
    //     model_path.to_owned()
    // } else {
    //     info!("Model path does not exist. Storing checkpoints in $CWD/checkpoints/");
    //     let cwd = std::env::current_dir().expect("Could not access current working directory.");
    //     cwd.join("checkpoints/")
    // };

    let mut runner: VecRunner<T, OBS_SIZE, NUM_ACTIONS> = VecRunner::new(init_env_state, NUM_ENVS);

    let mut high_score: f32 = -9999.0;

    for i in 1..=config.num_epochs {
        let mut dones = vec![false; NUM_ENVS];

        exp_buf.reset_counter();
        let mut eprews: Vec<Vec<f32>> = Vec::new();
        let mut eplens: Vec<Vec<i64>> = Vec::new();

        for _ in 0..config.num_steps {
            let obs = runner.current_state();
            let (critic, actions, neglogps) = learner.infer(&obs, None, true);

            let run_step = runner.step(&actions);
            exp_buf.add_experience(
                &obs,
                &run_step.rewards,
                &actions,
                &critic,
                &dones,
                &neglogps,
            );

            dones = run_step.dones;
            eprews.push(run_step.final_scores);
            eplens.push(run_step.final_step_nums);
        }

        let returns = exp_buf.returns(&dones);
        let (observations, actions, values, neglogps) = exp_buf.training_views();

        let explained_variance = explained_variance(values, returns.view());

        let exp_batcher = ExperienceBatcher::new(
            observations,
            actions,
            values,
            neglogps,
            returns.view(),
            config.batch_size,
        );

        let mut stats = TrainingStats::default();
        for _ in 0..config.num_train_iterations {
            for batch in exp_batcher.into_iter() {
                stats = learner.step(batch);
            }
        }
        stats.explained_variance = explained_variance;

        let ep_scores: Vec<f32> = eprews.into_iter().flatten().collect();
        let ep_lens: Vec<i64> = eplens.into_iter().flatten().collect();
        let num_eps = ep_scores.len();
        let avg_ep_len = ep_lens.into_iter().sum::<i64>() as f32 / num_eps as f32;
        let avg_score = ep_scores.into_iter().sum::<f32>() / num_eps as f32;

        if (i > 10) && (avg_score > high_score) {
            info!(
                "New best score: Learning update {}, Num Eps {}, Avg Ep len {}, Avg Ep Score {}, Loss {:?}",
                i, num_eps, avg_ep_len, avg_score, stats
            );
            info!("Skipping file save for now");
            // learner
            //     .model
            //     .clone()
            //     .save_file(checkpoint_path.join(format!("best_so_far_{i}")), &recorder)
            //     .expect("Failed to save high score checkpoint");
            high_score = avg_score;
        }
        if i % 10 == 0 {
            info!(
                "Learning update {}, Num Eps {}, Avg Ep len {}, Avg Ep Score {}, Loss {:?}",
                i, num_eps, avg_ep_len, avg_score, stats
            );
        }
        if i % 100 == 0 || i == config.num_epochs {
            info!("Skipping file save for now");
            // learner
            //     .model
            //     .clone()
            //     .save_file(checkpoint_path.join(format!("checkpoint_{i}")), &recorder)
            //     .expect("Failed to save checkpoint");
        }
    }
}
