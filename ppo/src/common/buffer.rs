use burn::prelude::*;
use ndarray::{prelude::*, Array, Axis, RemoveAxis};

trait IndexView<T> {
    fn index_view(&mut self, idx: usize) -> &mut [T];
}

impl<T, D> IndexView<T> for Array<T, D>
where
    D: RemoveAxis,
{
    fn index_view(&mut self, idx: usize) -> &mut [T] {
        let mut axis_mut = self.index_axis_mut(Axis(0), idx);
        let data = axis_mut.as_slice_mut().unwrap();
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr(), data.len()) }
    }
}

/// Experience buffer that supports multiple parallel environments and a 1D
/// observation. (Multiple dimensions should be flattened before use).
///
/// The experience buffer has fixed capacity, so any additional experiences past
/// the
pub struct ExperienceBuffer<const NUM_ENVS: usize, const OBS_SIZE: usize> {
    counter: usize,
    capacity: usize,

    gamma: f32,  // reward discounting factor
    lambda: f32, // advantage estimation discounting factor (lambda in the paper)

    // This storage allows only a number of entries up to a certain capacity.
    obs: Array3<f32>,       // [capacity, NUM_ENVS, OBS_SIZE]
    rewards: Array2<f32>,   // [capacity, NUM_ENVS]
    actions: Array2<usize>, // [capacity, NUM_ENVS]
    values: Array2<f32>,    // [capacity, NUM_ENVS]
    dones: Array2<f32>,     // [capacity, NUM_ENVS]
    neglogps: Array2<f32>,  // [capacity, NUM_ENVS]
}

impl<const NUM_ENVS: usize, const OBS_SIZE: usize> ExperienceBuffer<NUM_ENVS, OBS_SIZE> {
    /// Creates a new Experience buffer with specified capacity and default
    /// gamma (reward discounting factor) and lambda (advantage estimation
    /// discounting factor)
    pub fn new(capacity: usize) -> Self {
        Self::new_with_params(capacity, 0.99, 0.95)
    }

    /// Creates a new Experience buffer with specified capacity,
    /// gamma (reward discounting factor), and lambda (advantage estimation
    /// discounting factor)
    pub fn new_with_params(capacity: usize, gamma: f32, lambda: f32) -> Self {
        Self {
            counter: 0,
            capacity,

            gamma,
            lambda,

            obs: Array::zeros((capacity, NUM_ENVS, OBS_SIZE)),
            rewards: Array::zeros((capacity, NUM_ENVS)),
            actions: Array::zeros((capacity, NUM_ENVS)),
            values: Array::zeros((capacity, NUM_ENVS)),
            dones: Array::zeros((capacity, NUM_ENVS)),
            neglogps: Array::zeros((capacity, NUM_ENVS)),
        }
    }

    /// Adds a new experience to the buffer. All inputs must be the same size
    /// on the first axis (the number of envs).
    ///
    /// Takes the following as input:
    ///
    /// Observation used as input in the last inference - [NUM_ENVS, OBS_SIZE]
    /// Rewards from running the recommended actions on the envs - [NUM_ENVS]
    /// The recommended actions output from the last inference - [NUM_ENVS]
    /// The critic values for the last inference - [NUM_ENVS]
    /// A boolean slice indicating which environments are in the terminal state - [NUM_ENVS]
    /// The neglogps from the last inference - [NUM_ENVS]
    ///
    /// Panics if the inputs are an incorrect size
    pub fn add_experience<B: Backend>(
        &mut self,
        obs: Tensor<B, 2>, // [NUM_ENVS, OBS_SIZE]
        rewards: Box<[f32; NUM_ENVS]>,
        actions: Box<[usize; NUM_ENVS]>,
        vals: Tensor<B, 1>, // [NUM_ENVS]
        dones: Box<[bool; NUM_ENVS]>,
        neglogps: Tensor<B, 1>,
    ) {
        let idx = self.counter % self.capacity;

        let dones = dones.map(|d| d as u8 as f32);

        assert_eq!(obs.dims(), [NUM_ENVS, OBS_SIZE]);
        let obs_data = obs.into_data();
        self.obs
            .index_view(idx)
            .copy_from_slice(obs_data.as_slice().unwrap());

        self.rewards
            .index_view(idx)
            .copy_from_slice(rewards.as_slice());

        self.actions
            .index_view(idx)
            .copy_from_slice(actions.as_slice());

        assert_eq!(vals.dims(), [NUM_ENVS]);
        let vals_data = vals.into_data();
        self.values
            .index_view(idx)
            .copy_from_slice(vals_data.as_slice().unwrap());

        self.dones.index_view(idx).copy_from_slice(&dones);

        assert_eq!(neglogps.dims(), [NUM_ENVS]);
        let neglogsp_data = neglogps.into_data();
        self.neglogps
            .index_view(idx)
            .copy_from_slice(neglogsp_data.as_slice().unwrap());

        self.counter += 1;
        if self.counter >= self.capacity {
            self.counter = (self.counter % self.capacity) + self.capacity;
        }
    }

    fn len(&self) -> usize {
        if self.counter >= self.capacity {
            self.capacity
        } else {
            self.counter
        }
    }

    pub fn reset_counter(&mut self) {
        self.counter = 0;
    }

    /// Outputs the entire experience buffer as training views
    ///
    /// Returns a tuple of:
    /// Observations - Float [num_steps, OBS_SIZE]
    /// Actions - Int [num_steps]
    /// Values - Float [num_steps]
    /// Neglogps - Float [num_steps]
    pub fn training_views<B: Backend>(
        &self,
    ) -> (Tensor<B, 2>, Tensor<B, 1, Int>, Tensor<B, 1>, Tensor<B, 1>) {
        let len = self.len();
        let num_steps = len * NUM_ENVS;

        let device = Default::default();

        let obs_tensor = Tensor::<B, 1>::from_floats(
            self.obs.slice(s![0..len, .., ..]).as_slice().unwrap(),
            &device,
        )
        .reshape([num_steps, OBS_SIZE]);

        let actions_tensor = Tensor::<B, 1, Int>::from_ints(
            self.actions.slice(s![0..len, ..]).as_slice().unwrap(),
            &device,
        );

        let values_tensor = Tensor::<B, 1>::from_floats(
            self.values.slice(s![0..len, ..]).as_slice().unwrap(),
            &device,
        );

        let neglogps_tensor = Tensor::<B, 1>::from_floats(
            self.neglogps.slice(s![0..len, ..]).as_slice().unwrap(),
            &device,
        );

        (obs_tensor, actions_tensor, values_tensor, neglogps_tensor)
    }

    /// Calculates the estimated returns for each step in the experience buffer
    ///
    /// Takes a size [NUM_ENVS] Tensor of critic values from the last inference and
    /// a slice of booleans indicating whether the environment has reached
    /// a terminal state.
    ///
    /// Panics of last_values is of incorrect size.
    ///
    /// Returns a Tensor that is equal to the length of the experience buffer
    /// [num_steps]
    pub fn returns<B: Backend>(
        &self,
        last_values: Tensor<B, 1>, // [NUM_ENVS]
        last_dones: Box<[bool; NUM_ENVS]>,
    ) -> Tensor<B, 1> {
        let len = self.len();

        assert_eq!(last_values.dims(), [NUM_ENVS]);

        let mut lastgaelam: Array1<f32> = Array::zeros(NUM_ENVS);
        let mut advs: Array2<f32> = Array::zeros((len, NUM_ENVS));

        let nonterminals = 1f32 - &self.dones;

        let last_nonterminal = arr1(&last_dones.map(|d| !d as u8 as f32));
        let last_values_data = last_values.into_data();
        let last_values = arr1(last_values_data.as_slice().unwrap());

        for t in (0..len).rev() {
            // 1D arrays of length NUM_ENVS
            let (nextvalues, nextnonterminal) = {
                if t == len - 1 {
                    (last_values.view(), last_nonterminal.view())
                } else {
                    let idx = t + 1;
                    (self.values.row(idx), nonterminals.row(idx))
                }
            };

            let delta = &self.rewards.row(t) + &nextvalues * &nextnonterminal * self.gamma
                - self.values.row(t);
            lastgaelam = delta + &nextnonterminal * &lastgaelam * (self.gamma * self.lambda);
            advs.row_mut(t).assign(&lastgaelam);
        }
        let res = advs + &self.values;

        let device = Default::default();

        Tensor::<B, 1>::from_floats(res.slice(s![0..len, ..]).as_slice().unwrap(), &device)
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Wgpu;

    use super::*;

    #[test]
    fn exp_buffer_below_capacity() {
        let mut exp_buf: ExperienceBuffer<2, 3> = ExperienceBuffer::new(3);

        let device = Default::default();

        let obs1 = Tensor::from_floats([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], &device);
        exp_buf.add_experience::<Wgpu>(
            obs1,
            Box::new([0.1, 1.1]),
            Box::new([1, 2]),
            Tensor::from_floats([3.0, 6.0], &device),
            Box::new([false, false]),
            Tensor::from_floats([20.0, 21.0], &device),
        );
        let (obs, actions, values, neglogps) = exp_buf.training_views::<Wgpu>();

        assert_eq!(obs.dims(), [2, 3]);

        assert_eq!(
            obs.to_data().to_vec::<f32>().unwrap(),
            vec![
                0.0, 1.0, 2.0, // First row
                1.0, 2.0, 3.0 // Second row
            ]
        );

        assert_eq!(actions.dims(), [2]);
        assert_eq!(actions.to_data().to_vec::<i32>().unwrap(), vec![1, 2]);

        assert_eq!(values.dims(), [2]);
        assert_eq!(values.to_data().to_vec::<f32>().unwrap(), vec![3.0, 6.0]);

        assert_eq!(neglogps.dims(), [2]);
        assert_eq!(
            neglogps.to_data().to_vec::<f32>().unwrap(),
            vec![20.0, 21.0]
        );

        assert_eq!(obs.dims()[0], actions.dims()[0]);
        assert_eq!(actions.dims()[0], values.dims()[0]);
        assert_eq!(values.dims()[0], neglogps.dims()[0]);
    }

    #[test]
    fn exp_buffer_at_capacity() {
        let mut exp_buf: ExperienceBuffer<2, 3> = ExperienceBuffer::new(3);

        let device = Default::default();

        let obs1 = Tensor::from_floats([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], &device);
        let obs2 = Tensor::from_floats([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], &device);
        let obs3 = Tensor::from_floats([[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]], &device);

        exp_buf.add_experience::<Wgpu>(
            obs1,
            Box::new([0.1, 1.1]),
            Box::new([1, 2]),
            Tensor::from_floats([3.0, 6.0], &device),
            Box::new([false, false]),
            Tensor::from_floats([20.0, 21.0], &device),
        );

        exp_buf.add_experience::<Wgpu>(
            obs2,
            Box::new([1.1, 2.1]),
            Box::new([2, 3]),
            Tensor::from_floats([6.0, 9.0], &device),
            Box::new([false, false]),
            Tensor::from_floats([21.0, 22.0], &device),
        );

        exp_buf.add_experience::<Wgpu>(
            obs3,
            Box::new([2.1, 3.1]),
            Box::new([3, 4]),
            Tensor::from_floats([9.0, 12.0], &device),
            Box::new([false, false]),
            Tensor::from_floats([22.0, 23.0], &device),
        );

        let (obs, actions, values, neglogps) = exp_buf.training_views::<Wgpu>();

        assert_eq!(obs.dims(), [6, 3]);

        assert_eq!(
            obs.to_data().to_vec::<f32>().unwrap(),
            vec![
                0.0, 1.0, 2.0, // First row
                1.0, 2.0, 3.0, // Second row
                2.0, 3.0, 4.0, // Third row
                3.0, 4.0, 5.0, // Fourth row
                4.0, 5.0, 6.0, // Fifth row
                5.0, 6.0, 7.0 // Sixth row
            ]
        );

        assert_eq!(actions.dims(), [6]);
        assert_eq!(
            actions.to_data().to_vec::<i32>().unwrap(),
            vec![1, 2, 2, 3, 3, 4]
        );

        assert_eq!(values.dims(), [6]);
        assert_eq!(
            values.to_data().to_vec::<f32>().unwrap(),
            vec![3.0, 6.0, 6.0, 9.0, 9.0, 12.0]
        );

        assert_eq!(neglogps.dims(), [6]);
        assert_eq!(
            neglogps.to_data().to_vec::<f32>().unwrap(),
            vec![20.0, 21.0, 21.0, 22.0, 22.0, 23.0]
        );

        assert_eq!(obs.dims()[0], actions.dims()[0]);
        assert_eq!(actions.dims()[0], values.dims()[0]);
        assert_eq!(values.dims()[0], neglogps.dims()[0]);
    }

    #[test]
    fn exp_buffer_over_capacity() {
        // The experience buffer should overwrite the oldest data first
        let mut exp_buf: ExperienceBuffer<2, 3> = ExperienceBuffer::new(3);

        let device = Default::default();

        let obs1 = Tensor::from_floats([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], &device);
        let obs2 = Tensor::from_floats([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], &device);
        let obs3 = Tensor::from_floats([[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]], &device);
        let obs4 = Tensor::from_floats([[5.0, 6.0, 7.0], [6.0, 7.0, 8.0]], &device);
        let obs5 = Tensor::from_floats([[6.0, 7.0, 8.0], [7.0, 8.0, 9.0]], &device);

        exp_buf.add_experience::<Wgpu>(
            obs1,
            Box::new([0.1, 1.1]),
            Box::new([1, 2]),
            Tensor::from_floats([3.0, 6.0], &device),
            Box::new([false, false]),
            Tensor::from_floats([20.0, 21.0], &device),
        );

        for _ in 0..(3 * 456 - 2) {
            exp_buf.add_experience::<Wgpu>(
                obs2.clone(),
                Box::new([1.1, 2.1]),
                Box::new([2, 3]),
                Tensor::from_floats([6.0, 9.0], &device),
                Box::new([false, false]),
                Tensor::from_floats([21.0, 22.0], &device),
            );
        }

        // Should end up in the third slot
        exp_buf.add_experience::<Wgpu>(
            obs3,
            Box::new([2.1, 3.1]),
            Box::new([3, 4]),
            Tensor::from_floats([9.0, 12.0], &device),
            Box::new([false, false]),
            Tensor::from_floats([22.0, 23.0], &device),
        );

        exp_buf.add_experience::<Wgpu>(
            obs4,
            Box::new([3.1, 4.1]),
            Box::new([4, 5]),
            Tensor::from_floats([12.0, 15.0], &device),
            Box::new([false, false]),
            Tensor::from_floats([23.0, 24.0], &device),
        );

        exp_buf.add_experience::<Wgpu>(
            obs5,
            Box::new([4.1, 5.1]),
            Box::new([5, 6]),
            Tensor::from_floats([15.0, 18.0], &device),
            Box::new([false, true]),
            Tensor::from_floats([24.0, 25.0], &device),
        );

        let (obs, actions, values, neglogps) = exp_buf.training_views::<Wgpu>();

        assert_eq!(
            obs.to_data().to_vec::<f32>().unwrap(),
            vec![
                5.0, 6.0, 7.0, // First row
                6.0, 7.0, 8.0, // Second row
                6.0, 7.0, 8.0, // Third row
                7.0, 8.0, 9.0, // Fourth row
                4.0, 5.0, 6.0, // Fifth row
                5.0, 6.0, 7.0 // Sixth row
            ]
        );

        assert_eq!(actions.dims(), [6]);
        assert_eq!(
            actions.to_data().to_vec::<i32>().unwrap(),
            vec![4, 5, 5, 6, 3, 4]
        );

        assert_eq!(values.dims(), [6]);
        assert_eq!(
            values.to_data().to_vec::<f32>().unwrap(),
            vec![12.0, 15.0, 15.0, 18.0, 9.0, 12.0]
        );

        assert_eq!(neglogps.dims(), [6]);
        assert_eq!(
            neglogps.to_data().to_vec::<f32>().unwrap(),
            vec![23.0, 24.0, 24.0, 25.0, 22.0, 23.0]
        );

        assert_eq!(obs.dims()[0], actions.dims()[0]);
        assert_eq!(actions.dims()[0], values.dims()[0]);
        assert_eq!(values.dims()[0], neglogps.dims()[0]);
    }

    #[test]
    fn exp_buffer_returns_sanity() {
        let mut exp_buf: ExperienceBuffer<2, 3> = ExperienceBuffer::new(3);

        let device = Default::default();

        let obs1 = Tensor::from_floats([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], &device);
        let obs2 = Tensor::from_floats([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], &device);
        let obs3 = Tensor::from_floats([[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]], &device);

        exp_buf.add_experience::<Wgpu>(
            obs1,
            Box::new([0.1, 1.1]),
            Box::new([1, 2]),
            Tensor::from_floats([3.0, 6.0], &device),
            Box::new([false, false]),
            Tensor::from_floats([20.0, 21.0], &device),
        );

        exp_buf.add_experience::<Wgpu>(
            obs2,
            Box::new([1.1, 2.1]),
            Box::new([2, 3]),
            Tensor::from_floats([6.0, 9.0], &device),
            Box::new([false, false]),
            Tensor::from_floats([21.0, 22.0], &device),
        );

        exp_buf.add_experience::<Wgpu>(
            obs3,
            Box::new([2.1, 3.1]),
            Box::new([3, 4]),
            Tensor::from_floats([9.0, 12.0], &device),
            Box::new([false, false]),
            Tensor::from_floats([22.0, 23.0], &device),
        );

        let returns = exp_buf.returns::<Wgpu>(
            Tensor::from_floats([12.0, 15.0], &device),
            Box::new([true, true]),
        );
        let ret = returns.to_data().to_vec::<f32>().unwrap();
        print!("ret {ret:?}");
        assert!(ret[0] > 3.708 && ret[0] < 3.7081);
        assert!(ret[1] > 6.821 && ret[1] < 6.822);
        assert!(ret[2] > 3.52 && ret[2] < 3.521);
        assert!(ret[3] > 5.609 && ret[3] < 5.61);
        assert!(ret[4] > 2.09 && ret[4] < 2.11);
        assert!(ret[5] > 3.09 && ret[5] < 3.11);

        let (obs, _, _, _) = exp_buf.training_views::<Wgpu>();
        assert_eq!(obs.dims()[0], returns.dims()[0]);
    }
}
