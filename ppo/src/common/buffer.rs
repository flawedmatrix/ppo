use ndarray::{prelude::*, RemoveAxis};

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

    last_values_idx: usize,
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

            last_values_idx: 0,
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
    pub fn add_experience(
        &mut self,
        obs: &[[f32; OBS_SIZE]], // [NUM_ENVS, OBS_SIZE]
        rewards: &[f32],         // [NUM_ENVS]
        actions: &[usize],       // [NUM_ENVS]
        vals: &[f32],            // [NUM_ENVS]
        dones: &[bool],          // [NUM_ENVS]
        neglogps: &[f32],        // [NUM_ENVS]
    ) {
        let idx = self.counter % self.capacity;

        let dones: Vec<f32> = dones.iter().map(|d| *d as u8 as f32).collect();

        assert_eq!(obs.len(), NUM_ENVS);
        self.obs.index_view(idx).copy_from_slice(obs.as_flattened());

        assert_eq!(rewards.len(), NUM_ENVS);
        self.rewards.index_view(idx).copy_from_slice(rewards);

        assert_eq!(actions.len(), NUM_ENVS);
        self.actions.index_view(idx).copy_from_slice(actions);

        assert_eq!(vals.len(), NUM_ENVS);
        self.values.index_view(idx).copy_from_slice(vals);

        assert_eq!(dones.len(), NUM_ENVS);
        self.dones.index_view(idx).copy_from_slice(&dones);

        assert_eq!(neglogps.len(), NUM_ENVS);
        self.neglogps.index_view(idx).copy_from_slice(neglogps);

        self.last_values_idx = idx;

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
    pub fn training_views(
        &self,
    ) -> (
        ArrayView2<f32>,
        ArrayView1<usize>,
        ArrayView1<f32>,
        ArrayView1<f32>,
    ) {
        let len = self.len();
        let num_steps = len * NUM_ENVS;

        let obs = self
            .obs
            .slice(s![0..len, .., ..])
            .into_shape((num_steps, OBS_SIZE))
            .unwrap();
        let actions = self
            .actions
            .slice(s![0..len, ..])
            .into_shape((num_steps,))
            .unwrap();
        let values = self
            .values
            .slice(s![0..len, ..])
            .into_shape((num_steps,))
            .unwrap();
        let neglogps = self
            .neglogps
            .slice(s![0..len, ..])
            .into_shape((num_steps,))
            .unwrap();

        (obs, actions, values, neglogps)
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
    pub fn returns(
        &self,
        last_dones: &[bool], // [NUM_ENVS]
    ) -> Array1<f32> {
        let len = self.len();
        let num_steps = len * NUM_ENVS;

        assert_eq!(last_dones.len(), NUM_ENVS);

        let mut lastgaelam: Array1<f32> = Array::zeros(NUM_ENVS);
        let mut advs: Array2<f32> = Array::zeros((len, NUM_ENVS));

        let nonterminals = 1f32 - &self.dones;

        let last_dones: Vec<f32> = last_dones.iter().map(|d| !*d as u8 as f32).collect();

        let last_nonterminal = arr1(&last_dones);
        let last_values = self.values.row(self.last_values_idx);

        for t in (0..len).rev() {
            // 1D arrays of length NUM_ENVS
            let (nextvalues, nextnonterminal) = {
                if t == len - 1 {
                    (last_values, last_nonterminal.view())
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
        let res = advs + self.values.slice(s![0..len, ..]);

        res.into_shape((num_steps,)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_buffer_below_capacity() {
        let mut exp_buf: ExperienceBuffer<2, 3> = ExperienceBuffer::new(3);

        let obs1 = vec![[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]];
        exp_buf.add_experience(
            &obs1,
            &[0.1, 1.1],
            &[1, 2],
            &[3.0, 6.0],
            &[false, false],
            &[20.0, 21.0],
        );
        let (obs, actions, values, neglogps) = exp_buf.training_views();

        assert_eq!(obs.shape(), [2, 3]);

        assert_eq!(obs, array![[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]);

        assert_eq!(actions.shape(), [2]);
        assert_eq!(actions, array![1, 2]);

        assert_eq!(values.shape(), [2]);
        assert_eq!(values, array![3., 6.]);

        assert_eq!(neglogps.shape(), [2]);
        assert_eq!(neglogps, array![20.0, 21.0]);

        let returns = exp_buf.returns(&[true, true]);
        assert_eq!(returns.shape(), [2]);
    }

    #[test]
    fn exp_buffer_at_capacity() {
        let mut exp_buf: ExperienceBuffer<2, 3> = ExperienceBuffer::new(3);

        let obs1 = vec![[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]];
        let obs2 = vec![[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]];
        let obs3 = vec![[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]];

        exp_buf.add_experience(
            &obs1,
            &[0.1, 1.1],
            &[1, 2],
            &[3.0, 6.0],
            &[false, false],
            &[20.0, 21.0],
        );

        exp_buf.add_experience(
            &obs2,
            &[1.1, 2.1],
            &[2, 3],
            &[6.0, 9.0],
            &[false, false],
            &[21.0, 22.0],
        );

        exp_buf.add_experience(
            &obs3,
            &[2.1, 3.1],
            &[3, 4],
            &[9.0, 12.0],
            &[false, false],
            &[22.0, 23.0],
        );

        let (obs, actions, values, neglogps) = exp_buf.training_views();

        assert_eq!(obs.shape(), [6, 3]);

        assert_eq!(
            obs,
            array![
                [0., 1., 2.],
                [1., 2., 3.],
                [2., 3., 4.],
                [3., 4., 5.],
                [4., 5., 6.],
                [5., 6., 7.],
            ]
        );

        assert_eq!(actions.shape(), [6]);
        assert_eq!(actions, array![1, 2, 2, 3, 3, 4]);

        assert_eq!(values.shape(), [6]);
        assert_eq!(values, array![3.0, 6.0, 6.0, 9.0, 9.0, 12.0]);

        assert_eq!(neglogps.shape(), [6]);
        assert_eq!(neglogps, array![20.0, 21.0, 21.0, 22.0, 22.0, 23.0]);
    }

    #[test]
    fn exp_buffer_over_capacity() {
        // The experience buffer should overwrite the oldest data first
        let mut exp_buf: ExperienceBuffer<2, 3> = ExperienceBuffer::new(3);

        let obs1 = vec![[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]];
        let obs2 = vec![[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]];
        let obs3 = vec![[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]];
        let obs4 = vec![[5.0, 6.0, 7.0], [6.0, 7.0, 8.0]];
        let obs5 = vec![[6.0, 7.0, 8.0], [7.0, 8.0, 9.0]];

        exp_buf.add_experience(
            &obs1,
            &[0.1, 1.1],
            &[1, 2],
            &[3.0, 6.0],
            &[false, false],
            &[20.0, 21.0],
        );

        for _ in 0..(3 * 456 - 2) {
            exp_buf.add_experience(
                &obs2,
                &[1.1, 2.1],
                &[2, 3],
                &[6.0, 9.0],
                &[false, false],
                &[21.0, 22.0],
            );
        }

        // Should end up in the third slot
        exp_buf.add_experience(
            &obs3,
            &[2.1, 3.1],
            &[3, 4],
            &[9.0, 12.0],
            &[false, false],
            &[22.0, 23.0],
        );

        exp_buf.add_experience(
            &obs4,
            &[3.1, 4.1],
            &[4, 5],
            &[12.0, 15.0],
            &[false, false],
            &[23.0, 24.0],
        );

        exp_buf.add_experience(
            &obs5,
            &[4.1, 5.1],
            &[5, 6],
            &[15.0, 18.0],
            &[false, true],
            &[24.0, 25.0],
        );

        let (obs, actions, values, neglogps) = exp_buf.training_views();

        assert_eq!(
            obs,
            array![
                [5.0, 6.0, 7.0],
                [6.0, 7.0, 8.0],
                [6.0, 7.0, 8.0],
                [7.0, 8.0, 9.0],
                [4.0, 5.0, 6.0],
                [5.0, 6.0, 7.0],
            ]
        );

        assert_eq!(actions.shape(), [6]);
        assert_eq!(actions, array![4, 5, 5, 6, 3, 4]);

        assert_eq!(values.shape(), [6]);
        assert_eq!(values, array![12.0, 15.0, 15.0, 18.0, 9.0, 12.0]);

        assert_eq!(neglogps.shape(), [6]);
        assert_eq!(neglogps, array![23.0, 24.0, 24.0, 25.0, 22.0, 23.0]);

        let returns = exp_buf.returns(&[true, true]);
        assert_eq!(returns.shape(), [6]);
    }

    #[test]
    fn exp_buffer_returns_sanity() {
        let mut exp_buf: ExperienceBuffer<2, 3> = ExperienceBuffer::new(3);

        let obs1 = vec![[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]];
        let obs2 = vec![[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]];
        let obs3 = vec![[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]];

        exp_buf.add_experience(
            &obs1,
            &[0.1, 1.1],
            &[1, 2],
            &[3.0, 6.0],
            &[false, false],
            &[20.0, 21.0],
        );

        exp_buf.add_experience(
            &obs2,
            &[1.1, 2.1],
            &[2, 3],
            &[6.0, 9.0],
            &[false, false],
            &[21.0, 22.0],
        );

        exp_buf.add_experience(
            &obs3,
            &[2.1, 3.1],
            &[3, 4],
            &[9.0, 12.0],
            &[false, false],
            &[22.0, 23.0],
        );

        let returns = exp_buf.returns(&[true, true]);
        let ret = returns.as_slice().unwrap();
        print!("ret {ret:?}");
        assert!(ret[0] > 3.708 && ret[0] < 3.7081);
        assert!(ret[1] > 6.821 && ret[1] < 6.822);
        assert!(ret[2] > 3.52 && ret[2] < 3.521);
        assert!(ret[3] > 5.609 && ret[3] < 5.61);
        assert!(ret[4] > 2.09 && ret[4] < 2.11);
        assert!(ret[5] > 3.09 && ret[5] < 3.11);

        let (obs, _, _, _) = exp_buf.training_views();
        assert_eq!(obs.shape()[0], returns.shape()[0]);
    }
}
