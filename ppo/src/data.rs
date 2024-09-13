use burn::prelude::*;
use ndarray::{ArrayView1, ArrayView2};
use rand::prelude::*;

#[derive(Clone, Debug)]
pub struct ExperienceBatch<B: Backend> {
    pub observations: Tensor<B, 2>,
    pub actions: Tensor<B, 1, Int>,
    pub values: Tensor<B, 1>,
    pub neglogps: Tensor<B, 1>,
    pub returns: Tensor<B, 1>,
}

pub struct ExperienceBatcher<B: Backend> {
    set: ExperienceBatch<B>,
    shuffled_indices: Vec<usize>,
    batch_size: usize,
    device: B::Device,
}

impl<B: Backend> ExperienceBatcher<B> {
    pub fn new(
        observations: ArrayView2<f32>,
        actions: ArrayView1<u32>,
        values: ArrayView1<f32>,
        neglogps: ArrayView1<f32>,
        returns: ArrayView1<f32>,
        batch_size: usize,
        device: B::Device,
    ) -> Self {
        let (buf_size, obs_size) = observations.dim();
        let set = ExperienceBatch {
            observations: Tensor::<B, 1>::from_floats(
                observations.as_standard_layout().as_slice().unwrap(),
                &device,
            )
            .reshape([buf_size, obs_size]),
            actions: Tensor::<B, 1, Int>::from_ints(
                actions.as_standard_layout().as_slice().unwrap(),
                &device,
            ),
            values: Tensor::<B, 1>::from_floats(
                values.as_standard_layout().as_slice().unwrap(),
                &device,
            ),
            neglogps: Tensor::<B, 1>::from_floats(
                neglogps.as_standard_layout().as_slice().unwrap(),
                &device,
            ),
            returns: Tensor::<B, 1>::from_floats(
                returns.as_standard_layout().as_slice().unwrap(),
                &device,
            ),
        };
        let mut indices: Vec<usize> = (0..buf_size).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
        ExperienceBatcher {
            set,
            shuffled_indices: indices,
            batch_size,
            device,
        }
    }
}

impl<'a, B: Backend> IntoIterator for &'a ExperienceBatcher<B> {
    type Item = ExperienceBatch<B>;
    type IntoIter = ExperienceBatcherIterator<'a, B>;

    fn into_iter(self) -> Self::IntoIter {
        ExperienceBatcherIterator {
            batcher: self,
            iter_idx: 0,
        }
    }
}

pub struct ExperienceBatcherIterator<'a, B: Backend> {
    batcher: &'a ExperienceBatcher<B>,
    iter_idx: usize,
}

impl<'a, B: Backend> Iterator for ExperienceBatcherIterator<'a, B> {
    type Item = ExperienceBatch<B>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.iter_idx >= self.batcher.shuffled_indices.len() {
            return None;
        }
        let start = self.iter_idx;
        let end = std::cmp::min(
            self.iter_idx + self.batcher.batch_size,
            self.batcher.shuffled_indices.len(),
        );
        let indices = &self.batcher.shuffled_indices[start..end];
        let indices_tensor = Tensor::from_ints(indices, &self.batcher.device);

        self.iter_idx += self.batcher.batch_size;

        Some(ExperienceBatch {
            observations: self
                .batcher
                .set
                .observations
                .clone()
                .select(0, indices_tensor.clone()),
            actions: self
                .batcher
                .set
                .actions
                .clone()
                .select(0, indices_tensor.clone()),
            values: self
                .batcher
                .set
                .values
                .clone()
                .select(0, indices_tensor.clone()),
            neglogps: self
                .batcher
                .set
                .neglogps
                .clone()
                .select(0, indices_tensor.clone()),
            returns: self.batcher.set.returns.clone().select(0, indices_tensor),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use burn::backend::NdArray;

    use crate::common::ExperienceBuffer;

    use super::*;

    #[test]
    fn batcher_sanity() {
        let mut exp_buf = ExperienceBuffer::<2, 3>::new(10);

        for i in 0..5 {
            let obs = vec![[0.0 + i as f32, 1.0, 2.0], [100.0 + i as f32, 2.0, 3.0]];
            exp_buf.add_experience(
                &obs,
                &[0.1, 1.1],
                &[1, 2],
                &[3.0, 6.0],
                &[false, false],
                &[20.0, 21.0],
            );
        }

        let (observations, actions, values, neglogps) = exp_buf.training_views();

        let returns = exp_buf.returns(&[12.0, 15.0], &[true, true]);

        let device = Default::default();

        let experience_set = ExperienceBatcher::<NdArray>::new(
            observations,
            actions,
            values,
            neglogps,
            returns.view(),
            6,
            device,
        );

        let mut seen_obs = HashSet::new();
        let mut num_iterations = 0;
        for batch in experience_set.into_iter() {
            let [batch_size, obs_size] = batch.observations.dims();
            assert_eq!(obs_size, 3);

            let obs_data = batch.observations.to_data();
            let obs = obs_data.as_slice::<f32>().unwrap();

            for i in (0..(batch_size * obs_size)).step_by(obs_size) {
                assert!(seen_obs.insert(obs[i].round() as i64));
            }

            num_iterations += 1;
        }
        assert_eq!(num_iterations, 2);

        for v in &[0, 1, 2, 3, 4, 100, 101, 102, 103, 104] {
            assert!(seen_obs.contains(v));
        }
    }
}
