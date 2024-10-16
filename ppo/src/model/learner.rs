use dfdx::prelude::*;
use rand_distr::Uniform;
use tracing::trace_span;

use crate::TrainingConfig;

use super::data::ExperienceBatch;

use super::PolicyNetworkConfig;
use super::{
    util::{dist_entropy, neglog_probs},
    ModelConfig, PolicyNetwork,
};

#[derive(Default, Debug)]
pub struct TrainingStats {
    pub pg_loss: f32,
    pub vf_loss: f32,
    pub entropy: f32,
    pub approxkl: f32,
    pub clipfrac: f32,
    pub explained_variance: f32,
}

fn argmax<T: PartialOrd>(x: &[T]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap()
}

pub struct Learner<
    const OBS_SIZE: usize,
    const HIDDEN_DIM: usize,
    const NUM_ACTIONS: usize,
    D: Device<f32>,
> {
    model: PolicyNetwork<OBS_SIZE, NUM_ACTIONS, HIDDEN_DIM, f32, D>,
    optim: Adam<PolicyNetwork<OBS_SIZE, NUM_ACTIONS, HIDDEN_DIM, f32, D>, f32, D>,

    infer_span: tracing::Span,
    step_span: tracing::Span,

    config: ModelConfig,

    grads: Gradients<f32, D>,
    cpu_device: Cpu,

    obs_tensor: Tensor<(usize, Const<OBS_SIZE>), f32, D>,
}

impl<const OBS_SIZE: usize, const HIDDEN_DIM: usize, const NUM_ACTIONS: usize, D: Device<f32>>
    Learner<OBS_SIZE, HIDDEN_DIM, NUM_ACTIONS, D>
{
    pub fn new(config: TrainingConfig, device: D) -> Self {
        Self::new_learner(config, device, true)
    }

    fn new_learner(config: TrainingConfig, device: D, require_init: bool) -> Self {
        let cpu_device = Cpu::default();
        cpu_device.enable_cache();

        let model = device.build_module::<f32>(PolicyNetworkConfig::new(
            config.model_config.num_hidden_layers,
            require_init,
        ));
        let optim = Adam::new(
            &model,
            AdamConfig {
                lr: config.lr,
                ..Default::default()
            },
        );

        let grads = model.alloc_grads();

        Self {
            model,
            optim,
            grads,
            infer_span: trace_span!("learner.infer"),
            step_span: trace_span!("learner.step"),
            config: config.model_config,
            cpu_device,

            obs_tensor: device.zeros_like(&(config.num_envs, Const::<OBS_SIZE>)),
        }
    }

    pub fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
        config: TrainingConfig,
        device: D,
    ) -> Self {
        let mut learner = Self::new_learner(config, device, false);
        learner
            .model
            .load_safetensors(path)
            .expect("Failed to load model");
        learner
    }

    /// Runs an inference step of the model with critic and negative log probs
    /// and optionally randomize action selection among the best choices
    ///
    /// Input: [num_envs, OBS_SIZE]
    /// Output:
    ///  - Critic: [num_envs]
    ///  - Chosen Action: [num_envs]
    ///  - Negative Log Probs: [num_envs]
    pub fn infer(
        &self,
        obs: &[[f32; OBS_SIZE]],
        action_mask: Option<[bool; NUM_ACTIONS]>,
        randomize: bool,
    ) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
        let _enter = self.infer_span.enter();

        let cpu_device = &self.cpu_device;

        let mut obs_tensor = self.obs_tensor.clone();
        obs_tensor.copy_from(obs.as_flattened());

        let (critic, actor) = self.model.forward(obs_tensor);

        let (critic, actor) = (critic.to_device(cpu_device), actor.to_device(cpu_device));

        let actor = match action_mask {
            Some(m) => {
                let mask = m.iter().map(|&x| !x as u8 as f32).collect::<Vec<f32>>();
                let neg_mask = cpu_device.tensor_from_vec(mask, (1, Const::<NUM_ACTIONS>)) * 500.0;
                actor - neg_mask
            }
            None => actor,
        };

        let len = critic.shape().concrete()[0];

        // Get uniform distribution on [0, 1) in the shape of logits
        let dist = Uniform::new(0f32, 1f32);
        let u = cpu_device.sample_like(actor.shape(), dist);

        // Sample 1 action from actor probs
        let logprobs = if randomize {
            actor.clone() - (-u.ln()).ln()
        } else {
            actor.clone()
        };

        let mut actions: Vec<usize> = Vec::with_capacity(len);
        for probs in logprobs.as_vec().chunks(NUM_ACTIONS) {
            actions.push(argmax(probs));
        }

        let actions_tensor = cpu_device.tensor_from_vec(actions.clone(), (len,));

        let neglogp = neglog_probs(actor, actions_tensor);

        let vf = critic.reshape_like(&(len,));

        (vf.as_vec(), actions, neglogp.as_vec())
    }

    pub fn step(
        &mut self,
        batch: ExperienceBatch<OBS_SIZE, D>,
        collect_stats: bool,
    ) -> Option<TrainingStats> {
        let _enter = self.step_span.enter();
        let mut stats = TrainingStats::default();

        let batch_size = batch.actions.len();

        let input = batch.observations;

        let (critic, policy_latent) = self.model.forward(input.trace(self.grads.clone()));

        let critic = critic.reshape_like(&(batch_size,));
        let neglogps = neglog_probs(policy_latent.with_empty_tape(), batch.actions);

        if collect_stats {
            stats.approxkl = trace_span!("approxkl").in_scope(|| -> f32 {
                let nlp_diff = neglogps.with_empty_tape() - batch.neglogps.clone();
                nlp_diff.square().mean().to_device(&self.cpu_device).array() * 0.5
            });
        }

        let entropy = dist_entropy(policy_latent).mean();
        if collect_stats {
            stats.entropy = trace_span!("entropy")
                .in_scope(|| -> f32 { entropy.to_device(&self.cpu_device).array() });
        }

        let values = batch.values;
        let returns = batch.returns;

        let values_clipped = (critic.with_empty_tape() - values.clone())
            .clamp(-self.config.clip_range, self.config.clip_range)
            + values;
        let vf_losses1 = (critic - returns.clone()).square();
        let vf_losses2 = (values_clipped - returns).square();
        let vf_loss = Tensor::maximum(vf_losses1, vf_losses2).mean() * 0.5;
        if collect_stats {
            stats.vf_loss =
                trace_span!("vf_loss").in_scope(|| vf_loss.to_device(&self.cpu_device).array());
        }

        let ratio = (-neglogps + batch.neglogps).exp();

        if collect_stats {
            stats.clipfrac = trace_span!("clip_frac").in_scope(|| -> f32 {
                let r = ratio.to_device(&self.cpu_device);
                let gt_mask = (r - 1.0).abs().gt(self.config.clip_range).to_dtype::<u32>();
                gt_mask.sum().array() as f32 / batch_size as f32
            });
        }

        let neg_advs = -batch.advantages;

        let pg_losses1 = ratio.with_empty_tape() * neg_advs.clone();
        let pg_losses2 =
            ratio.clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * neg_advs;
        let pg_loss = Tensor::maximum(pg_losses1, pg_losses2).mean();

        if collect_stats {
            stats.pg_loss =
                trace_span!("pg_loss").in_scope(|| pg_loss.to_device(&self.cpu_device).array());
        }

        let loss = (pg_loss - (entropy * self.config.entropy_coefficient))
            + (vf_loss * self.config.vf_coefficient);

        self.grads = loss.backward();
        self.optim
            .update(&mut self.model, &self.grads)
            .expect("Unused params");
        self.model.zero_grads(&mut self.grads);

        if !collect_stats {
            return None;
        }
        Some(stats)
    }

    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) {
        self.model
            .save_safetensors(path)
            .expect("Failed to save model");
    }
}
