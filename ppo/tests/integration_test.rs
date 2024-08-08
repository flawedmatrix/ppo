use burn::backend::{Autodiff, NdArray};

use ppo::{train, Environment, ModelConfig, TrainingConfig};

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
    let device = Default::default();

    train::<_, _, TrainingBackend, 10, 3, 3>(init_state, training_config, "temp/", &device);
}
