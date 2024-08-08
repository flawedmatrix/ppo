mod common;
mod data;
mod model;
mod runner;
mod training;

use std::{fmt::Debug, hash::Hash};

/// An environment that supports being vectorized, performing actions, and
/// scoring a particular state.
///
/// Currently only supports 1D environments.
pub trait Environment<const OBS_SIZE: usize, const NUM_ACTIONS: usize>:
    Eq + Hash + Debug + Copy + Clone
{
    /// Returns a vectorized snapshot of the environment state
    fn as_vector(&self) -> [f32; OBS_SIZE];
    /// A boolean mask of valid actions, where the index of the mask corresponds
    /// to the action ID
    fn valid_actions(&self) -> [bool; NUM_ACTIONS];
    /// Step number of the environment
    fn step_num(&self) -> i64;
    /// Returns true if the Environment is in a finished or "game over"
    /// state
    fn is_done(&self) -> bool;
    /// Performs the action associated with the action ID
    fn do_action(&mut self, action_id: usize);
    /// Returns a score (could be a game score or a heuristic evaluation) of
    /// the environment
    fn score(&self) -> f32;
}

pub use model::ModelConfig;
pub use training::*;

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, NdArray};

    use super::*;

    #[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
    struct TestEnv(i32);

    impl Environment<3, 3> for TestEnv {
        fn as_vector(&self) -> [f32; 3] {
            [self.0 as f32, self.0 as f32, self.0 as f32]
        }
        fn valid_actions(&self) -> [bool; 3] {
            [true, true, true]
        }

        fn step_num(&self) -> i64 {
            100
        }

        fn is_done(&self) -> bool {
            self.0 == 3
        }

        fn do_action(&mut self, action_id: usize) {
            if action_id == 1 {
                self.0 += 1;
            }
        }

        fn score(&self) -> f32 {
            self.0 as f32
        }
    }

    #[test]
    fn training_lifecycle() {
        let init_state = TestEnv(0);

        let model_config = ModelConfig::new(3, 3)
            .with_num_hidden_layers(2)
            .with_hidden_size(10);
        let training_config = TrainingConfig::new(model_config)
            .with_num_epochs(10)
            .with_num_steps(5)
            .with_batch_size(2)
            .with_num_envs(10);

        type TrainingBackend = Autodiff<NdArray>;

        println!("Starting training...");
        train::<_, _, TrainingBackend, 10, 3, 3>(init_state, training_config, "temp/");
    }
}
