use burn::{nn::Relu, prelude::*};

use crate::common::{combine_linear_heads, OrthoLinearConfig};

use super::PolicyModel;

#[derive(Config, Debug)]
/// ModelConfig describes the parameters of the training process for the model.
/// Must provide observation size and number of actions when instantiating.
/// e.g.
/// ```
/// use ppo::ModelConfig;
/// let observation_size = 20;
/// let num_actions = 10;
/// ModelConfig::new(observation_size, num_actions);
/// ``````
pub struct ModelConfig {
    pub observation_size: usize,
    pub num_actions: usize,

    #[config(default = 0.2)]
    pub clip_range: f32,
    #[config(default = 0.01)]
    pub entropy_coefficient: f32,
    #[config(default = 0.5)]
    pub vf_coefficient: f32,
    #[config(default = 0.5)]
    pub max_grad_norm: f32,

    #[config(default = 3e-4)]
    pub lr: f64,
    #[config(default = 2)]
    pub num_hidden_layers: usize,
    #[config(default = 1024)]
    pub hidden_size: usize,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PolicyModel<B> {
        let sqrt_2 = f32::sqrt(2.);
        let critic_output = OrthoLinearConfig::new(self.hidden_size, 1, 1.0)
            .with_bias(false)
            .init(device);
        let actor_output = OrthoLinearConfig::new(self.hidden_size, self.num_actions, 0.01)
            .with_bias(false)
            .init(device);
        let output = combine_linear_heads(critic_output, actor_output, device);

        PolicyModel {
            input: OrthoLinearConfig::new(self.observation_size, self.hidden_size, sqrt_2)
                .init(device),
            hidden: (0..self.num_hidden_layers)
                .map(|_| {
                    OrthoLinearConfig::new(self.hidden_size, self.hidden_size, sqrt_2).init(device)
                })
                .collect(),
            output,
            activation: Relu::new(),
        }
    }
}
