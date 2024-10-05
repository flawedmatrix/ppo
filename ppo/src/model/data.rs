use dfdx::prelude::*;
use ndarray::{ArrayView1, ArrayView2};
use rand::prelude::*;
use tracing::trace_span;

#[derive(Clone, Debug)]
pub struct ExperienceBatch<const OBS_SIZE: usize, D: Device<f32>> {
    pub observations: Tensor<(usize, Const<OBS_SIZE>), f32, D>, // [batch_size, obs_size]
    pub actions: Tensor<(usize,), usize, D>,                    // [batch_size] (Ints)
    pub values: Tensor<(usize,), f32, D>,                       // [batch_size]
    pub neglogps: Tensor<(usize,), f32, D>,                     // [batch_size]
    pub returns: Tensor<(usize,), f32, D>,                      // [batch_size]
    pub advantages: Tensor<(usize,), f32, D>,                   // [batch_size]
}

pub struct ExperienceBatcher<const OBS_SIZE: usize, D: Device<f32>> {
    set: ExperienceBatch<OBS_SIZE, Cpu>,
    batch_size: usize,
    cpu_device: Cpu,
    cache: ExperienceBatch<OBS_SIZE, D>,
}

impl<const OBS_SIZE: usize, D: Device<f32>> ExperienceBatcher<OBS_SIZE, D> {
    pub fn new(
        observations: ArrayView2<f32>,
        actions: ArrayView1<usize>,
        values: ArrayView1<f32>,
        neglogps: ArrayView1<f32>,
        returns: ArrayView1<f32>,
        batch_size: usize,
        cache: ExperienceBatch<OBS_SIZE, D>,
    ) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "ExperienceBatcher::new");
        let _guard = span.enter();

        let cpu_device = Cpu::default();

        let buf_size = observations.dim().0;
        let observations = cpu_device.tensor_from_vec(
            observations
                .as_standard_layout()
                .as_slice()
                .unwrap()
                .to_vec(),
            (buf_size, Const::<OBS_SIZE>),
        );
        let actions = cpu_device.tensor_from_vec(
            actions.as_standard_layout().as_slice().unwrap().to_vec(),
            (buf_size,),
        );
        let values = cpu_device.tensor_from_vec(
            values.as_standard_layout().as_slice().unwrap().to_vec(),
            (buf_size,),
        );
        let neglogps = cpu_device.tensor_from_vec(
            neglogps.as_standard_layout().as_slice().unwrap().to_vec(),
            (buf_size,),
        );
        let returns = cpu_device.tensor_from_vec(
            returns.as_standard_layout().as_slice().unwrap().to_vec(),
            (buf_size,),
        );

        let advantages = returns.clone() - values.clone();
        let set = ExperienceBatch {
            observations,
            actions,
            values,
            neglogps,
            returns,
            advantages,
        };

        Self {
            set,
            batch_size,
            cpu_device,
            cache,
        }
    }
}

impl<'a, const OBS_SIZE: usize, D: Device<f32>> IntoIterator
    for &'a ExperienceBatcher<OBS_SIZE, D>
{
    type Item = ExperienceBatch<OBS_SIZE, D>;
    type IntoIter = ExperienceBatcherIterator<'a, OBS_SIZE, D>;

    fn into_iter(self) -> Self::IntoIter {
        let buf_size = self.set.actions.len();
        let mut indices: Vec<usize> = (0..buf_size).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        ExperienceBatcherIterator {
            batcher: self,
            indices,
            iter_idx: 0,
        }
    }
}

pub struct ExperienceBatcherIterator<'a, const OBS_SIZE: usize, D: Device<f32>> {
    batcher: &'a ExperienceBatcher<OBS_SIZE, D>,
    indices: Vec<usize>,
    iter_idx: usize,
}

impl<'a, const OBS_SIZE: usize, D: Device<f32>> Iterator
    for ExperienceBatcherIterator<'a, OBS_SIZE, D>
{
    type Item = ExperienceBatch<OBS_SIZE, D>;
    fn next(&mut self) -> Option<Self::Item> {
        let buf_size = self.indices.len();
        if self.iter_idx >= buf_size {
            return None;
        }

        let start = self.iter_idx;
        let end = std::cmp::min(start + self.batcher.batch_size, buf_size);
        self.iter_idx += self.batcher.batch_size;

        let indices_slice = self.indices[start..end].to_vec();
        let len = indices_slice.len();

        let indices_span = trace_span!("indices_tensor");
        let _indices_enter = indices_span.enter();

        let indices_tensor = self
            .batcher
            .cpu_device
            .tensor_from_vec(indices_slice, (len,));
        drop(_indices_enter);

        let gather_span = trace_span!("gather");
        let _gather_enter = gather_span.enter();

        let obs_cpu = {
            let obs_view = self.batcher.set.observations.clone();
            obs_view.gather(indices_tensor.clone())
        };
        let actions_cpu = {
            let actions_view = self.batcher.set.actions.clone();
            actions_view.gather(indices_tensor.clone())
        };
        let values_cpu = {
            let values_view = self.batcher.set.values.clone();
            values_view.gather(indices_tensor.clone())
        };
        let neglogps_cpu = {
            let neglogps_view = self.batcher.set.neglogps.clone();
            neglogps_view.gather(indices_tensor.clone())
        };
        let returns_cpu = {
            let returns_view = self.batcher.set.returns.clone();
            returns_view.gather(indices_tensor.clone())
        };
        let advantages_cpu = {
            // Standardize advantages at time of batching
            let advantages_view = self.batcher.set.advantages.clone();
            let a = advantages_view.gather(indices_tensor.clone());
            let a_mean = a.clone().mean().array();
            let a_std = correct_std(a.clone().var().array(), len);
            (a - a_mean) / (a_std + 1e-8)
        };
        drop(_gather_enter);

        let copy_span = trace_span!("copy");
        let _copy_enter = copy_span.enter();

        let mut observations = self.batcher.cache.observations.clone().slice((0..len, ..));
        observations.copy_from(&obs_cpu.as_vec());

        let mut actions = self.batcher.cache.actions.clone().slice((0..len,));
        actions.copy_from(&actions_cpu.as_vec());

        let mut values = self.batcher.cache.values.clone().slice((0..len,));
        values.copy_from(&values_cpu.as_vec());

        let mut neglogps = self.batcher.cache.neglogps.clone().slice((0..len,));
        neglogps.copy_from(&neglogps_cpu.as_vec());

        let mut returns = self.batcher.cache.returns.clone().slice((0..len,));
        returns.copy_from(&returns_cpu.as_vec());

        let mut advantages = self.batcher.cache.advantages.clone().slice((0..len,));
        advantages.copy_from(&advantages_cpu.as_vec());

        drop(_copy_enter);

        Some(ExperienceBatch {
            observations,
            actions,
            values,
            neglogps,
            returns,
            advantages,
        })
    }
}

/// Computes the mean and unbiased standard deviation over all elements in the tensor
fn correct_std(var: f32, n: usize) -> f32 {
    let n = n as f32;
    let corrected_var = var * (n / (n - 1.0));
    corrected_var.sqrt()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::common::ExperienceBuffer;

    use super::*;

    #[test]
    fn batcher_sanity() {
        const OBS_SIZE: usize = 3;
        let mut exp_buf = ExperienceBuffer::<2, OBS_SIZE>::new(10);

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

        let device = Cpu::default();

        const BATCH_SIZE: usize = 6;

        let cache = ExperienceBatch {
            observations: device.zeros_like(&(BATCH_SIZE, Const::<OBS_SIZE>)),
            actions: device.zeros_like(&(BATCH_SIZE,)),
            values: device.zeros_like(&(BATCH_SIZE,)),
            neglogps: device.zeros_like(&(BATCH_SIZE,)),
            returns: device.zeros_like(&(BATCH_SIZE,)),
            advantages: device.zeros_like(&(BATCH_SIZE,)),
        };

        let (observations, actions, values, neglogps) = exp_buf.training_views();

        let returns = exp_buf.returns(&[true, true]);

        let experience_set = ExperienceBatcher::<OBS_SIZE, _>::new(
            observations,
            actions,
            values,
            neglogps,
            returns.view(),
            BATCH_SIZE,
            cache,
        );

        let mut seen_obs = HashSet::new();
        let mut num_iterations = 0;
        for batch in experience_set.into_iter() {
            let batch_obs = batch.observations.as_vec();

            (0..batch_obs.len()).step_by(OBS_SIZE).for_each(|idx| {
                assert!(seen_obs.insert(batch_obs[idx].round() as i64));
            });

            num_iterations += 1;
        }
        assert_eq!(num_iterations, 2);

        for v in &[0, 1, 2, 3, 4, 100, 101, 102, 103, 104] {
            assert!(seen_obs.contains(v));
        }
    }
}
