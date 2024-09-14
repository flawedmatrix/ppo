use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    grad_clipping::GradientClippingConfig,
    optim::AdamConfig,
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use ndarray::{ArrayView1, Axis};
use rand::RngCore;
use tracing::{info, span, Level};

use crate::{
    common::ExperienceBuffer,
    data::ExperienceBatcher,
    model::{Learner, ModelConfig, TrainingStats},
    runner::VecRunner,
    Environment,
};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 160)]
    /// Number of environments to run in parallel for each training pass
    pub num_envs: usize,

    #[config(default = 32)]
    /// Number of action steps to train for in each update
    pub num_steps: usize,

    #[config(default = 15000)]
    /// Number of training passes for the model
    pub num_epochs: usize,

    #[config(default = 4)]
    /// Number of iterations during the model training pass
    pub num_train_iterations: usize,

    #[config(default = 1280)]
    /// Number of experiences per batch of data used in a single iteration
    /// during the model training pass
    pub batch_size: usize,

    pub model_config: ModelConfig,
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

pub fn train<T, P, const NUM_ENVS: usize, const OBS_SIZE: usize, const NUM_ACTIONS: usize>(
    init_env_state: T,
    config: TrainingConfig,
    model_path: P,
) where
    T: Environment<OBS_SIZE, NUM_ACTIONS>,
    P: AsRef<std::path::Path>,
{
    let mut exp_buf: ExperienceBuffer<NUM_ENVS, OBS_SIZE> = ExperienceBuffer::new(config.num_steps);

    type TrainingBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::BestAvailable;

    let mut rng = rand::thread_rng();
    let seed: u64 = rng.next_u64();
    TrainingBackend::seed(seed);

    info!("Instantiating model with config {config:?}");
    let mut learner = Learner {
        model: config.model_config.init(&device),
        optim: AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(
                config.model_config.max_grad_norm,
            )))
            .init(),
        span: span!(Level::TRACE, "learner.step"),
        config: config.model_config,
    };
    info!("Model instantiated.");

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let model_path = model_path.as_ref();

    if model_path.exists() {
        info!("Loading checkpoint from path {:?}", model_path);
        let record = recorder
            .load(model_path.to_path_buf(), &device)
            .expect("Should be able to load the model weights from the provided file");
        learner.model = learner.model.load_record(record);
    }

    let checkpoint_path = if model_path.is_file() && model_path.parent().is_some_and(|p| p.exists())
    {
        info!(
            "Storing checkpoints in parent path of model : {}",
            model_path.parent().unwrap().display()
        );
        model_path.parent().unwrap().to_owned()
    } else if model_path.is_dir() && model_path.exists() {
        info!("Storing checkpoints in path: {}", model_path.display());
        model_path.to_owned()
    } else {
        info!("Model path does not exist. Storing checkpoints in $CWD/checkpoints/");
        let cwd = std::env::current_dir().expect("Could not access current working directory.");
        cwd.join("checkpoints/")
    };

    let mut runner: VecRunner<T, OBS_SIZE, NUM_ACTIONS> = VecRunner::new(init_env_state, NUM_ENVS);

    let mut high_score: f32 = -9999.0;

    for i in 1..=config.num_epochs {
        let mut dones = vec![false; NUM_ENVS];

        exp_buf.reset_counter();
        let mut eprews: Vec<Vec<f32>> = Vec::new();
        let mut eplens: Vec<Vec<i64>> = Vec::new();

        for _ in 0..config.num_steps {
            let obs = runner.current_state();
            let (critic, actions, neglogps) = learner
                .model
                .infer::<OBS_SIZE, NUM_ACTIONS>(&obs, None, true, &device);

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

        let exp_batcher = ExperienceBatcher::<TrainingBackend>::new(
            observations,
            actions,
            values,
            neglogps,
            returns.view(),
            config.batch_size,
            device.clone(),
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
            learner
                .model
                .clone()
                .save_file(checkpoint_path.join(format!("best_so_far_{i}")), &recorder)
                .expect("Failed to save high score checkpoint");
            high_score = avg_score;
        }
        if i % 10 == 0 {
            info!(
                "Learning update {}, Num Eps {}, Avg Ep len {}, Avg Ep Score {}, Loss {:?}",
                i, num_eps, avg_ep_len, avg_score, stats
            );
        }
        if i % 100 == 0 || i == config.num_epochs {
            learner
                .model
                .clone()
                .save_file(checkpoint_path.join(format!("checkpoint_{i}")), &recorder)
                .expect("Failed to save checkpoint");
        }
    }
}
