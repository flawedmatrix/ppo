use crate::common::RunningMeanStd;
use crate::Environment;
use burn::prelude::*;

use ndarray::prelude::*;

#[derive(Debug)]
/// A single step of running an action on each environment in [NUM_ENVS]
pub struct VecRunStep<const NUM_ENVS: usize, const OBS_SIZE: usize> {
    // Size of num_envs
    pub rewards: Box<[f32; NUM_ENVS]>,
    pub dones: Box<[bool; NUM_ENVS]>,

    // For statistical purposes only
    // Only populated with the final score of finished envs
    pub final_scores: Vec<f32>,
    // Only populated with the step num of finished envs
    pub final_step_nums: Vec<i64>,
}

pub struct VecRunner<T, const NUM_ENVS: usize, const OBS_SIZE: usize, const NUM_ACTIONS: usize>
where
    T: Environment<OBS_SIZE, NUM_ACTIONS>,
{
    init_state: T,
    envs: Vec<T>,

    gamma: f32,
    epsilon: f32,
    cliprew: f32,

    ret: Array1<f32>,
    ret_rms: RunningMeanStd<Ix0>,
}

impl<T, const NUM_ENVS: usize, const OBS_SIZE: usize, const NUM_ACTIONS: usize>
    VecRunner<T, NUM_ENVS, OBS_SIZE, NUM_ACTIONS>
where
    T: Environment<OBS_SIZE, NUM_ACTIONS>,
{
    pub fn new(init_state: T) -> Self {
        Self::new_with_params(init_state, 0.99, 1e-8, 10.)
    }

    pub fn new_with_params(init_state: T, gamma: f32, epsilon: f32, cliprew: f32) -> Self {
        let mut envs: Vec<T> = Vec::new();
        for _ in 0..NUM_ENVS {
            envs.push(init_state);
        }
        Self {
            init_state,
            envs,

            gamma,
            epsilon,
            cliprew,

            ret: Array::zeros((NUM_ENVS,)),
            ret_rms: RunningMeanStd::new(()),
        }
    }
    fn vec_to_boxed_array<U>(vec: Vec<U>) -> Box<[U; NUM_ENVS]> {
        let boxed_slice = vec.into_boxed_slice();

        let ptr = ::std::boxed::Box::into_raw(boxed_slice) as *mut [U; NUM_ENVS];

        unsafe { Box::from_raw(ptr) }
    }

    // actions should be a Vec of size num_envs
    pub fn step(&mut self, actions: Vec<usize>) -> VecRunStep<NUM_ENVS, OBS_SIZE> {
        let mut next_states: Vec<[f32; OBS_SIZE]> = Vec::new();
        let mut rewards: Vec<f32> = Vec::new();
        let mut dones: Vec<bool> = Vec::new();

        let mut final_scores: Vec<f32> = Vec::new();
        let mut final_step_nums: Vec<i64> = Vec::new();

        for (i, env) in self.envs.iter_mut().enumerate() {
            let action = actions[i];
            let valid_actions = env.valid_actions();

            let old_score = env.score();

            if valid_actions[action] {
                env.do_action(action);

                let new_score = env.score();
                if env.is_done() {
                    rewards.push(new_score);
                    dones.push(true);

                    final_scores.push(new_score);
                    final_step_nums.push(env.step_num());
                    *env = self.init_state;
                } else {
                    rewards.push(new_score - old_score);
                    dones.push(false);
                }
            } else {
                // Heavily disincentivize choosing invalid actions
                rewards.push(-5.0);
                dones.push(true);

                final_scores.push(-5.0);
                final_step_nums.push(env.step_num());
                *env = self.init_state.clone();
            }

            next_states.push(env.as_vector());
        }
        VecRunStep {
            rewards: self.normalized_rewards(rewards, dones.clone()),
            dones: Self::vec_to_boxed_array(dones),
            final_scores,
            final_step_nums,
        }
    }

    fn normalized_rewards(&mut self, rewards: Vec<f32>, dones: Vec<bool>) -> Box<[f32; NUM_ENVS]> {
        let rews = Array::from_vec(rewards);
        let dones = dones.iter().map(|d| *d as u8 as f32).collect();
        let dones_mask = Array::from_vec(dones);

        self.ret = &self.ret * self.gamma + &rews;
        self.ret_rms.update(&self.ret);
        let denom = (&self.ret_rms.var + self.epsilon).mapv(|x| x.sqrt());
        let norm_rews = (&rews / denom).mapv(|x| x.clamp(-self.cliprew, self.cliprew));

        self.ret = &self.ret * dones_mask;

        Self::vec_to_boxed_array(norm_rews.to_vec())
    }

    /// Returns the current state of all the envs, represented as a 2D tensor:
    /// [NUM_ENVS, OBS_SIZE]
    pub fn current_state<B: Backend>(&self) -> Tensor<B, 2> {
        let mut data: Vec<f32> = Vec::new();
        for env in self.envs.iter() {
            data.extend_from_slice(&env.as_vector());
        }
        let device = Default::default();
        Tensor::<B, 1>::from_floats(data.as_slice(), &device).reshape([NUM_ENVS, OBS_SIZE])
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Wgpu;

    use super::*;

    #[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
    struct TestEnv(i32);

    impl Environment<3, 3> for TestEnv {
        fn as_vector(&self) -> [f32; 3] {
            [self.0 as f32, self.0 as f32, self.0 as f32]
        }
        fn valid_actions(&self) -> [bool; 3] {
            [true, true, false]
        }

        fn step_num(&self) -> i64 {
            100
        }

        fn is_done(&self) -> bool {
            if self.0 == 3 {
                true
            } else {
                false
            }
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
    fn runner_step_lifecycle() {
        let init_state = TestEnv(0);

        let mut runner: VecRunner<TestEnv, 3, 3, 3> = VecRunner::new(init_state);

        let result = runner.step(vec![0, 0, 1]);
        assert_eq!(result.dones, Box::new([false, false, false]));
        assert_eq!(result.final_scores.len(), 0);
        assert_eq!(result.final_step_nums.len(), 0);

        let device = Default::default();

        let expected = Tensor::<Wgpu, 2>::from_floats(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            &device,
        );

        runner
            .current_state::<Wgpu>()
            .to_data()
            .assert_eq(&expected.to_data(), true);

        let result = runner.step(vec![1, 1, 1]);
        assert_eq!(result.dones, Box::new([false, false, false]));
        assert_eq!(result.final_scores.len(), 0);
        assert_eq!(result.final_step_nums.len(), 0);

        let expected = Tensor::<Wgpu, 2>::from_floats(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            &device,
        );

        runner
            .current_state::<Wgpu>()
            .to_data()
            .assert_eq(&expected.to_data(), true);

        // Expect any finished envs to be reset
        let result = runner.step(vec![1, 1, 1]);
        assert_eq!(result.dones, Box::new([false, false, true]));
        assert_eq!(result.final_scores, vec![3.0]);
        assert_eq!(result.final_step_nums, vec![100]);

        let expected = Tensor::<Wgpu, 2>::from_floats(
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]],
            &device,
        );
        runner
            .current_state::<Wgpu>()
            .to_data()
            .assert_eq(&expected.to_data(), true);

        let result = runner.step(vec![0, 1, 1]);
        assert_eq!(result.dones, Box::new([false, true, false]));
        assert_eq!(result.final_scores, vec![3.0]);
        assert_eq!(result.final_step_nums, vec![100]);

        let expected = Tensor::<Wgpu, 2>::from_floats(
            [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            &device,
        );
        runner
            .current_state::<Wgpu>()
            .to_data()
            .assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn runner_invalid_step_resets_env() {
        let init_state = TestEnv(0);

        let mut runner: VecRunner<TestEnv, 3, 3, 3> = VecRunner::new(init_state);

        let result = runner.step(vec![0, 0, 1]);
        assert_eq!(result.dones, Box::new([false, false, false]));
        assert_eq!(result.final_scores.len(), 0);
        assert_eq!(result.final_step_nums.len(), 0);

        let device = Default::default();

        let expected = Tensor::<Wgpu, 2>::from_floats(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            &device,
        );
        runner
            .current_state::<Wgpu>()
            .to_data()
            .assert_eq(&expected.to_data(), true);

        // Using an invalid action resets the env
        let result = runner.step(vec![1, 1, 2]);
        assert_eq!(result.dones, Box::new([false, false, true]));
        assert_eq!(result.final_scores, vec![-5.0]);
        assert_eq!(result.final_step_nums, vec![100]);

        let expected = Tensor::<Wgpu, 2>::from_floats(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            &device,
        );
        runner
            .current_state::<Wgpu>()
            .to_data()
            .assert_eq(&expected.to_data(), true);
    }
}
