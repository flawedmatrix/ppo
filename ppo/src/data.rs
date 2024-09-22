use candle_core::{Device, Result, Tensor};
use ndarray::{ArrayView1, ArrayView2};
use rand::prelude::*;
use tracing::error;

#[derive(Clone, Debug)]
pub struct ExperienceBatch {
    pub observations: Tensor, // [batch_size, obs_size]
    pub actions: Tensor,      // [batch_size] (Ints)
    pub values: Tensor,       // [batch_size]
    pub neglogps: Tensor,     // [batch_size]
    pub returns: Tensor,      // [batch_size]
}

pub struct ExperienceBatcher {
    set: ExperienceBatch,
    shuffled_indices: Vec<u32>,
    batch_size: usize,
    device: Device,
}

impl ExperienceBatcher {
    pub fn new(
        observations: ArrayView2<f32>,
        actions: ArrayView1<u32>,
        values: ArrayView1<f32>,
        neglogps: ArrayView1<f32>,
        returns: ArrayView1<f32>,
        batch_size: usize,
        device: Device,
    ) -> Result<Self> {
        let (buf_size, obs_size) = observations.dim();
        let set = ExperienceBatch {
            observations: Tensor::from_slice(
                observations.as_standard_layout().as_slice().unwrap(),
                &[buf_size, obs_size],
                &device,
            )?,
            actions: Tensor::from_slice(
                actions.as_standard_layout().as_slice().unwrap(),
                &[buf_size],
                &device,
            )?,
            values: Tensor::from_slice(
                values.as_standard_layout().as_slice().unwrap(),
                &[buf_size],
                &device,
            )?,
            neglogps: Tensor::from_slice(
                neglogps.as_standard_layout().as_slice().unwrap(),
                &[buf_size],
                &device,
            )?,
            returns: Tensor::from_slice(
                returns.as_standard_layout().as_slice().unwrap(),
                &[buf_size],
                &device,
            )?,
        };
        let mut indices: Vec<u32> = (0..buf_size as u32).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
        Ok(Self {
            set,
            shuffled_indices: indices,
            batch_size,
            device,
        })
    }
}

impl<'a> IntoIterator for &'a ExperienceBatcher {
    type Item = ExperienceBatch;
    type IntoIter = ExperienceBatcherIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        ExperienceBatcherIterator {
            batcher: self,
            iter_idx: 0,
        }
    }
}

pub struct ExperienceBatcherIterator<'a> {
    batcher: &'a ExperienceBatcher,
    iter_idx: usize,
}

impl<'a> ExperienceBatcherIterator<'a> {
    fn next_impl(&mut self) -> Result<ExperienceBatch> {
        if self.iter_idx >= self.batcher.shuffled_indices.len() {
            return Err(candle_core::Error::InvalidIndex {
                op: "ExperienceBatcherIterator::next",
                index: self.iter_idx,
                size: self.batcher.shuffled_indices.len(),
            });
        }
        let start = self.iter_idx;
        let end = std::cmp::min(
            self.iter_idx + self.batcher.batch_size,
            self.batcher.shuffled_indices.len(),
        );
        let indices = &self.batcher.shuffled_indices[start..end];
        let indices_tensor = Tensor::from_slice(indices, &[indices.len()], &self.batcher.device)?;

        self.iter_idx += self.batcher.batch_size;

        Ok(ExperienceBatch {
            observations: self
                .batcher
                .set
                .observations
                .clone()
                .index_select(&indices_tensor, 0)?,
            actions: self
                .batcher
                .set
                .actions
                .clone()
                .index_select(&indices_tensor, 0)?,
            values: self
                .batcher
                .set
                .values
                .clone()
                .index_select(&indices_tensor, 0)?,
            neglogps: self
                .batcher
                .set
                .neglogps
                .clone()
                .index_select(&indices_tensor, 0)?,
            returns: self
                .batcher
                .set
                .returns
                .clone()
                .index_select(&indices_tensor, 0)?,
        })
    }
}

impl<'a> Iterator for ExperienceBatcherIterator<'a> {
    type Item = ExperienceBatch;
    fn next(&mut self) -> Option<Self::Item> {
        if self.iter_idx >= self.batcher.shuffled_indices.len() {
            return None;
        }
        match self.next_impl() {
            Ok(batch) => Some(batch),
            Err(e) => {
                error!("Error in ExperienceBatcherIterator::next: {:?}", e);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::common::ExperienceBuffer;

    use super::*;

    #[test]
    fn batcher_sanity() -> Result<()> {
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

        let returns = exp_buf.returns(&[true, true]);

        let device = Device::Cpu;

        let experience_set = ExperienceBatcher::new(
            observations,
            actions,
            values,
            neglogps,
            returns.view(),
            6,
            device,
        )?;

        let mut seen_obs = HashSet::new();
        let mut num_iterations = 0;
        for batch in experience_set.into_iter() {
            let obs_size = batch.observations.dims()[1];
            assert_eq!(obs_size, 3);

            let batch_obs = batch.observations.to_vec2::<f32>()?;

            batch_obs.iter().for_each(|obs| {
                assert!(seen_obs.insert(obs[0].round() as i64));
            });

            num_iterations += 1;
        }
        assert_eq!(num_iterations, 2);

        for v in &[0, 1, 2, 3, 4, 100, 101, 102, 103, 104] {
            assert!(seen_obs.contains(v));
        }

        Ok(())
    }
}
