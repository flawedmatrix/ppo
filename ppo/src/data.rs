use candle_core::{Device, Result, Tensor};
use ndarray::{ArrayView1, ArrayView2};
use rand::prelude::*;

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
        device: &Device,
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "ExperienceBatcher::new");
        let _guard = span.enter();

        let cpu_device = Device::Cpu;

        let (buf_size, obs_size) = observations.dim();
        let observations = Tensor::from_slice(
            observations.as_standard_layout().as_slice().unwrap(),
            &[buf_size, obs_size],
            &cpu_device,
        )?;
        let actions = Tensor::from_slice(
            actions.as_standard_layout().as_slice().unwrap(),
            &[buf_size],
            &cpu_device,
        )?;
        let values = Tensor::from_slice(
            values.as_standard_layout().as_slice().unwrap(),
            &[buf_size],
            &cpu_device,
        )?;
        let neglogps = Tensor::from_slice(
            neglogps.as_standard_layout().as_slice().unwrap(),
            &[buf_size],
            &cpu_device,
        )?;
        let returns = Tensor::from_slice(
            returns.as_standard_layout().as_slice().unwrap(),
            &[buf_size],
            &cpu_device,
        )?;
        let set = ExperienceBatch {
            observations,
            actions,
            values,
            neglogps,
            returns,
        };
        Ok(Self {
            set,
            batch_size,
            device: device.clone(),
        })
    }

    fn _into_iter(&self) -> Result<ExperienceBatcherIterator> {
        let buf_size = self.set.actions.elem_count();
        let mut indices: Vec<u32> = (0..buf_size as u32).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        let indices_tensor = Tensor::from_slice(&indices, &[buf_size], &Device::Cpu)?;

        let data = ExperienceBatch {
            observations: self
                .set
                .observations
                .index_select(&indices_tensor, 0)?
                .to_device(&self.device)?,
            actions: self
                .set
                .actions
                .index_select(&indices_tensor, 0)?
                .to_device(&self.device)?,
            values: self
                .set
                .values
                .index_select(&indices_tensor, 0)?
                .to_device(&self.device)?,
            neglogps: self
                .set
                .neglogps
                .index_select(&indices_tensor, 0)?
                .to_device(&self.device)?,
            returns: self
                .set
                .returns
                .index_select(&indices_tensor, 0)?
                .to_device(&self.device)?,
        };
        Ok(ExperienceBatcherIterator {
            data,
            iter_idx: 0,
            batch_size: self.batch_size,
        })
    }
}

impl IntoIterator for &ExperienceBatcher {
    type Item = ExperienceBatch;
    type IntoIter = ExperienceBatcherIterator;

    fn into_iter(self) -> Self::IntoIter {
        match self._into_iter() {
            Ok(iter) => iter,
            Err(e) => {
                panic!("Error in ExperienceBatcher::into_iter: {:?}", e);
            }
        }
    }
}

pub struct ExperienceBatcherIterator {
    data: ExperienceBatch,
    batch_size: usize,
    iter_idx: usize,
}

impl ExperienceBatcherIterator {
    fn _next_impl(&mut self) -> Result<ExperienceBatch> {
        let buf_size = self.data.actions.elem_count();
        if self.iter_idx >= buf_size {
            return Err(candle_core::Error::InvalidIndex {
                op: "ExperienceBatcherIterator::next",
                index: self.iter_idx,
                size: buf_size,
            });
        }

        let start = self.iter_idx;
        let len = std::cmp::min(start + self.batch_size, buf_size) - start;
        self.iter_idx += len;

        Ok(ExperienceBatch {
            observations: self.data.observations.narrow(0, start, len)?,
            actions: self.data.actions.narrow(0, start, len)?,
            values: self.data.values.narrow(0, start, len)?,
            neglogps: self.data.neglogps.narrow(0, start, len)?,
            returns: self.data.returns.narrow(0, start, len)?,
        })
    }
}

impl Iterator for ExperienceBatcherIterator {
    type Item = ExperienceBatch;
    fn next(&mut self) -> Option<Self::Item> {
        let buf_size = self.data.actions.elem_count();
        if self.iter_idx >= buf_size {
            return None;
        }
        match self._next_impl() {
            Ok(batch) => Some(batch),
            Err(e) => {
                panic!("Error in ExperienceBatcherIterator::next: {:?}", e);
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
            &device,
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
