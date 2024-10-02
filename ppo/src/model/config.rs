/// ModelConfig describes the parameters of the training process for the model.
/// Must provide observation size and number of actions when instantiating.
/// e.g.
/// ```
/// use ppo::ModelConfig;
/// let observation_size = 20;
/// let num_actions = 10;
/// ModelConfig::new(observation_size, num_actions);
/// ``````
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    pub clip_range: f32,
    pub entropy_coefficient: f32,
    pub vf_coefficient: f32,

    // TODO: Doesn't support gradient clipping yet
    pub max_grad_norm: f32,

    /// Number of hidden layers to create
    pub num_hidden_layers: usize,
}

impl ModelConfig {
    /// Sets the clip range for the model.
    pub fn with_clip_range(mut self, clip_range: f32) -> Self {
        self.clip_range = clip_range;
        self
    }

    /// Sets the entropy coefficient for the model.
    pub fn with_entropy_coefficient(mut self, entropy_coefficient: f32) -> Self {
        self.entropy_coefficient = entropy_coefficient;
        self
    }

    /// Sets the value function coefficient for the model.
    pub fn with_vf_coefficient(mut self, vf_coefficient: f32) -> Self {
        self.vf_coefficient = vf_coefficient;
        self
    }

    /// Sets the maximum gradient norm for the model.
    pub fn with_max_grad_norm(mut self, max_grad_norm: f32) -> Self {
        self.max_grad_norm = max_grad_norm;
        self
    }

    /// Sets the number of hidden layers for the model.
    pub fn with_num_hidden_layers(mut self, num_hidden_layers: usize) -> Self {
        self.num_hidden_layers = num_hidden_layers;
        self
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            clip_range: 0.2,
            entropy_coefficient: 0.01,
            vf_coefficient: 0.5,
            max_grad_norm: 0.5,
            num_hidden_layers: 2,
        }
    }
}
