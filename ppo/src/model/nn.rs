use candle_core::{Device, Result, Tensor, D};
use candle_nn::{seq, Activation, Module, Sequential, VarBuilder};

use super::{
    linear::{critic_actor_heads, linear_with_span, LinearWithSpan},
    util::neglog_probs,
    ModelConfig,
};

pub struct PolicyModel {
    pub(super) input: LinearWithSpan,
    pub(super) hidden: Sequential,
    /// Combined policy and value head: [HIDDEN_DIM, NUM_ACTIONS + 1]
    pub(super) output: LinearWithSpan,

    span: tracing::Span,
}

impl PolicyModel {
    /// Returns the initialized model.
    pub fn new(cfg: ModelConfig, vb: VarBuilder) -> Result<Self> {
        let sqrt_2 = f32::sqrt(2.);
        let mut hidden_layers = seq();

        for i in 0..cfg.num_hidden_layers {
            hidden_layers = hidden_layers.add(linear_with_span(
                cfg.hidden_size,
                cfg.hidden_size,
                sqrt_2,
                vb.pp("hidden").pp(i.to_string()),
            )?);
            hidden_layers = hidden_layers.add(Activation::Relu);
        }

        Ok(Self {
            input: linear_with_span(
                cfg.observation_size,
                cfg.hidden_size,
                sqrt_2,
                vb.pp("input"),
            )?,
            hidden: hidden_layers,
            output: critic_actor_heads(cfg.hidden_size, cfg.num_actions, vb.pp("output"))?,
            span: tracing::span!(tracing::Level::TRACE, "policy_model"),
        })
    }
}

impl Module for PolicyModel {
    fn forward(&self, obs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut x = self.input.forward(obs)?;
        x = x.relu()?;
        // The hidden layers already handle activations
        x = self.hidden.forward(&x)?;

        self.output.forward(&x)
    }
}

impl PolicyModel {
    /// Runs a forward pass of the model and returns the critic and actor logits
    ///
    /// Input: [batch_size, OBS_SIZE]
    /// Output:
    ///  - Critic: [batch_size]
    ///  - Actor: [batch_size, NUM_ACTIONS]
    pub fn forward_critic_actor(&self, obs: &Tensor) -> Result<(Tensor, Tensor)> {
        let output = self.forward(obs)?; // [batch_size, num_actions + 1]

        let num_actions = output.dims()[1] - 1;

        let critic = output.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
        let actor = output.narrow(D::Minus1, 1, num_actions)?;

        Ok((critic, actor))
    }

    /// Runs an inference step of the model with critic and negative log probs
    /// and optionally randomize action selection among the best choices
    ///
    /// Input: [num_envs, OBS_SIZE]
    /// Output:
    ///  - Critic: [num_envs]
    ///  - Chosen Action: [num_envs]
    ///  - Negative Log Probs: [num_envs]
    pub fn infer<const OBS_SIZE: usize, const NUM_ACTIONS: usize>(
        &self,
        obs: &[[f32; OBS_SIZE]],
        action_mask: Option<[bool; NUM_ACTIONS]>,
        randomize: bool,
        device: Device,
    ) -> Result<(Vec<f32>, Vec<u32>, Vec<f32>)> {
        let num_envs = obs.len();
        let obs_tensor = Tensor::from_slice(obs.as_flattened(), &[num_envs, OBS_SIZE], &device)?;
        let (critic, actor) = self.forward_critic_actor(&obs_tensor)?;

        let actor = match action_mask {
            Some(m) => {
                let mask = m.iter().map(|&x| !x as u8 as f32).collect::<Vec<f32>>();
                let neg_mask = (Tensor::from_slice(&mask, &[num_envs], &device)? * 500.0)?;
                actor.broadcast_sub(&neg_mask)?
            }
            None => actor,
        };

        // Get uniform distribution on [0, 1) in the shape of logits

        // Sample 1 action from actor probs
        let logprobs = if randomize {
            let u = actor.rand_like(0., 1.)?;
            (actor.clone() - (u.log()?.neg()?).log()?)?
        } else {
            actor.clone()
        };
        let actions = logprobs.argmax(D::Minus1)?;

        let neglogps = neglog_probs(&actor, &actions)?;

        Ok((critic.to_vec1()?, actions.to_vec1()?, neglogps.to_vec1()?))
    }
}
