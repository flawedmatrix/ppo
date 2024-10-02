use dfdx::prelude::*;
use ndarray::{ArrayView1, ArrayView2};
use rand::prelude::*;

#[derive(Clone, Debug)]
pub struct ExperienceBatch<const OBS_SIZE: usize> {
    pub observations: Tensor<(usize, Const<OBS_SIZE>), f32, Cpu>, // [batch_size, obs_size]
    pub actions: Tensor<(usize,), usize, Cpu>,                    // [batch_size] (Ints)
    pub values: Tensor<(usize,), f32, Cpu>,                       // [batch_size]
    pub neglogps: Tensor<(usize,), f32, Cpu>,                     // [batch_size]
    pub returns: Tensor<(usize,), f32, Cpu>,                      // [batch_size]
}

pub struct ExperienceBatcher<const OBS_SIZE: usize> {
    set: ExperienceBatch<OBS_SIZE>,
    batch_size: usize,
    cpu_device: Cpu,
}

impl<const OBS_SIZE: usize> ExperienceBatcher<OBS_SIZE> {
    pub fn new(
        observations: ArrayView2<f32>,
        actions: ArrayView1<usize>,
        values: ArrayView1<f32>,
        neglogps: ArrayView1<f32>,
        returns: ArrayView1<f32>,
        batch_size: usize,
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
        let set = ExperienceBatch {
            observations,
            actions,
            values,
            neglogps,
            returns,
        };
        Self {
            set,
            batch_size,
            cpu_device,
        }
    }
}

impl<'a, const OBS_SIZE: usize> IntoIterator for &'a ExperienceBatcher<OBS_SIZE> {
    type Item = ExperienceBatch<OBS_SIZE>;
    type IntoIter = ExperienceBatcherIterator<'a, OBS_SIZE>;

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

pub struct ExperienceBatcherIterator<'a, const OBS_SIZE: usize> {
    batcher: &'a ExperienceBatcher<OBS_SIZE>,
    indices: Vec<usize>,
    iter_idx: usize,
}

impl<'a, const OBS_SIZE: usize> Iterator for ExperienceBatcherIterator<'a, OBS_SIZE> {
    type Item = ExperienceBatch<OBS_SIZE>;
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

        let indices_tensor = self
            .batcher
            .cpu_device
            .tensor_from_vec(indices_slice, (len,));

        let data = ExperienceBatch {
            observations: self
                .batcher
                .set
                .observations
                .clone()
                .gather(indices_tensor.clone()),
            actions: self
                .batcher
                .set
                .actions
                .clone()
                .gather(indices_tensor.clone()),
            values: self
                .batcher
                .set
                .values
                .clone()
                .gather(indices_tensor.clone()),
            neglogps: self
                .batcher
                .set
                .neglogps
                .clone()
                .gather(indices_tensor.clone()),
            returns: self
                .batcher
                .set
                .returns
                .clone()
                .gather(indices_tensor.clone()),
        };

        Some(data)
    }
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

        let (observations, actions, values, neglogps) = exp_buf.training_views();

        let returns = exp_buf.returns(&[true, true]);

        let experience_set = ExperienceBatcher::<OBS_SIZE>::new(
            observations,
            actions,
            values,
            neglogps,
            returns.view(),
            6,
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
