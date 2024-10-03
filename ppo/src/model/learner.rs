use dfdx::prelude::*;
use rand_distr::Uniform;
use tracing::trace_span;

use super::data::ExperienceBatch;

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

pub struct Learner<
    const OBS_SIZE: usize,
    const HIDDEN_DIM: usize,
    const NUM_ACTIONS: usize,
    D: Device<f32>,
> {
    pub model: PolicyNetwork<OBS_SIZE, NUM_ACTIONS, HIDDEN_DIM, f32, D>,
    pub optim: Adam<PolicyNetwork<OBS_SIZE, NUM_ACTIONS, HIDDEN_DIM, f32, D>, f32, D>,

    pub infer_span: tracing::Span,
    pub step_span: tracing::Span,

    pub config: ModelConfig,

    pub grads: Gradients<f32, D>,
    pub cpu_device: Cpu,
}

fn argmax<T: PartialOrd>(x: &[T]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap()
}

/// Computes the mean and unbiased standard deviation over all elements in the tensor
fn correct_std(var: f32, n: usize) -> f32 {
    let n = n as f32;
    let corrected_var = var * (n / (n - 1.0));
    corrected_var.sqrt()
}

impl<const OBS_SIZE: usize, const HIDDEN_DIM: usize, const NUM_ACTIONS: usize, D: Device<f32>>
    Learner<OBS_SIZE, HIDDEN_DIM, NUM_ACTIONS, D>
{
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
        let num_envs = obs.len();

        let model_device = self.model.device();
        let cpu_device = &self.cpu_device;

        let obs_tensor = model_device
            .tensor_from_vec(obs.as_flattened().to_vec(), (num_envs, Const::<OBS_SIZE>));

        let forward_span = trace_span!("model.forward");
        let _forward_enter = forward_span.enter();
        let (critic, actor) = self.model.forward(obs_tensor);
        drop(_forward_enter);

        let (critic, actor) = (critic.to_device(cpu_device), actor.to_device(cpu_device));

        let actor = match action_mask {
            Some(m) => {
                let mask = m.iter().map(|&x| !x as u8 as f32).collect::<Vec<f32>>();
                let neg_mask = cpu_device.tensor_from_vec(mask, (1, Const::<NUM_ACTIONS>)) * 500.0;
                actor - neg_mask
            }
            None => actor,
        };

        // Get uniform distribution on [0, 1) in the shape of logits

        let len = critic.shape().concrete()[0];

        // Get uniform distribution on [0, 1) in the shape of logits
        let dist = Uniform::new(0f32, 1f32);
        let u =
            trace_span!("device.sample").in_scope(|| cpu_device.sample_like(actor.shape(), dist));

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

        let actions_tensor = trace_span!("actions_tensor")
            .in_scope(|| cpu_device.tensor_from_vec(actions.clone(), (len,)));

        let neglogp = trace_span!("neglogp").in_scope(|| neglog_probs(actor, actions_tensor));

        let vf = critic.reshape_like(&(len,));

        (vf.as_vec(), actions, neglogp.as_vec())
    }

    pub fn step(&mut self, batch: ExperienceBatch<OBS_SIZE>) -> TrainingStats {
        let _enter = self.step_span.enter();
        let mut stats = TrainingStats::default();

        let batch_size = batch.actions.len();

        let model_device = self.model.device();

        let advs = trace_span!("advs").in_scope(|| {
            let a = batch.returns.clone() - batch.values.clone();
            let a_mean = a.clone().mean().array();
            let a_std = correct_std(a.clone().var().array(), batch_size);
            (a - a_mean) / (a_std + 1e-8)
        });

        let input = batch.observations.to_device(&model_device);

        let forward_span = trace_span!("model.forward");
        let _forward_enter = forward_span.enter();

        let (critic, policy_latent) = self.model.forward(input.trace(self.grads.clone()));

        drop(_forward_enter);

        let critic = critic.reshape_like(&(batch_size,));
        let actions = batch.actions.to_device(&model_device);
        let neglogps = neglog_probs(policy_latent.with_empty_tape(), actions);

        stats.approxkl = trace_span!("approxkl").in_scope(|| -> f32 {
            let nlp = neglogps.to_device(&self.cpu_device);
            let nlp_diff = nlp - batch.neglogps.clone();
            nlp_diff.square().mean().array() * 0.5
        });

        let entropy = trace_span!("entropy").in_scope(|| {
            let entropy = dist_entropy(policy_latent).mean();
            stats.entropy = entropy.to_device(&self.cpu_device).array();
            entropy
        });

        let values = batch.values.to_device(&model_device);
        let returns = batch.returns.to_device(&model_device);

        let values_clipped = (critic.with_empty_tape() - values.clone())
            .clamp(-self.config.clip_range, self.config.clip_range)
            + values;
        let vf_losses1 = (critic - returns.clone()).square();
        let vf_losses2 = (values_clipped - returns).square();
        let vf_loss = Tensor::maximum(vf_losses1, vf_losses2).mean() * 0.5;
        stats.vf_loss =
            trace_span!("vf_loss").in_scope(|| vf_loss.to_device(&self.cpu_device).array());

        let nlp_old = batch.neglogps.to_device(&model_device);
        let ratio = (-neglogps + nlp_old).exp();

        stats.clipfrac = trace_span!("clip_frac").in_scope(|| -> f32 {
            let r = ratio.to_device(&self.cpu_device);
            let gt_mask = (r - 1.0)
                .abs()
                .gt(self.config.clip_range)
                .to_dtype::<usize>();
            gt_mask.sum().array() as f32 / batch_size as f32
        });

        let advs = advs.to_device(&model_device);

        let pg_losses1 = ratio.with_empty_tape() * -advs.clone();
        let pg_losses2 =
            ratio.clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * -advs;
        let pg_loss = Tensor::maximum(pg_losses1, pg_losses2).mean();

        stats.pg_loss =
            trace_span!("pg_loss").in_scope(|| pg_loss.to_device(&self.cpu_device).array());

        let loss = (pg_loss - (entropy * self.config.entropy_coefficient))
            + (vf_loss * self.config.vf_coefficient);

        self.grads = trace_span!("loss.backward").in_scope(|| loss.backward());
        trace_span!("optim.step").in_scope(|| {
            self.optim
                .update(&mut self.model, &self.grads)
                .expect("Unused params")
        });
        self.model.zero_grads(&mut self.grads);

        stats
    }
}
