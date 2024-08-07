use burn::{nn::Linear, prelude::*, tensor::Distribution};
use nn::Relu;

#[derive(Module, Debug)]
pub struct PolicyModel<B: Backend> {
    pub(super) input: Linear<B>,
    pub(super) hidden: Vec<Linear<B>>,
    /// Combined policy and value head: [HIDDEN_DIM, NUM_ACTIONS + 1]
    pub(super) output: Linear<B>,
    pub(super) activation: Relu,
}

impl<B: Backend> PolicyModel<B> {
    /// Runs a forward pass of the model
    ///
    /// Input: [batch_size, OBS_SIZE]
    /// Output:
    ///  - Critic: [batch_size]
    ///  - Actor: [batch_size, NUM_ACTIONS]
    pub fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 1>, Tensor<B, 2>) {
        let mut x = self.input.forward(obs); // [batch_size, hidden_dim]
        x = self.activation.forward(x);

        for layer in &self.hidden {
            x = layer.forward(x); // [batch_size, hidden_dim]
            x = self.activation.forward(x);
        }
        let x = self.output.forward(x); //  [batch_size, num_actions + 1]
        let [batch_size, num_actions_p1] = x.dims();

        let critic = x.clone().slice([0..batch_size, 0..1]).reshape([-1]);
        let actor = x.slice([0..batch_size, 1..num_actions_p1]);

        (critic, actor)
    }

    /// Runs an inference step of the model with critic and negative log probs
    /// and optionally randomize action selection among the best choices
    ///
    /// Input: [num_envs, OBS_SIZE]
    /// Output:
    ///  - Critic: [num_envs]
    ///  - Chosen Action: [num_envs]
    ///  - Negative Log Probs: [num_envs]
    pub fn infer<const NUM_ACTIONS: usize>(
        &self,
        obs: Tensor<B, 2>,
        action_mask: Option<[bool; NUM_ACTIONS]>,
        randomize: bool,
        device: &B::Device,
    ) -> (Tensor<B, 1>, Vec<u32>, Tensor<B, 1>) {
        let (critic, actor) = self.forward(obs);
        let actor = match action_mask {
            Some(m) => {
                let neg_mask =
                    Tensor::<B, 1, Bool>::from_bool(TensorData::from(m), device).bool_not();
                actor - neg_mask.float().unsqueeze_dim(1) * 500.0
            }
            None => actor,
        };

        // Get uniform distribution on [0, 1) in the shape of logits

        // Sample 1 action from actor probs
        let logprobs = if randomize {
            let u = actor.random_like(Distribution::Uniform(0., 1.));
            actor.clone() - (-u.log()).log()
        } else {
            actor.clone()
        };
        let actions = logprobs.argmax(1).squeeze(1);

        let neglogps = super::learner::neglog_probs(actor, actions.clone());

        (critic, actions.to_data().to_vec::<u32>().unwrap(), neglogps)
    }
}
