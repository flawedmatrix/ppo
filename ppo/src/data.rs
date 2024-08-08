use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use ndarray::{s, Array1, Array2};

/// A temporary container for training views.
pub struct TrainingView {
    pub observations: Array2<f32>,
    pub actions: Array1<u32>,
    pub values: Array1<f32>,
    pub neglogps: Array1<f32>,
    pub returns: Array1<f32>,
}

#[derive(Clone, Debug)]
pub struct Experience {
    observation: Vec<f32>,
    action: u32,
    value: f32,
    neglogp: f32,
    ret: f32,
}

impl Dataset<Experience> for TrainingView {
    fn get(&self, index: usize) -> Option<Experience> {
        if index > self.len() {
            return None;
        }

        let observation = self
            .observations
            .slice(s![index, ..])
            .into_owned()
            .into_raw_vec();

        Some(Experience {
            observation,
            action: *self.actions.get(index)?,
            value: *self.values.get(index)?,
            neglogp: *self.neglogps.get(index)?,
            ret: *self.returns.get(index)?,
        })
    }

    fn len(&self) -> usize {
        self.actions.len()
    }
}

#[derive(Clone)]
pub struct ExperienceBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> ExperienceBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct ExperienceBatch<B: Backend> {
    pub observations: Tensor<B, 2>,
    pub actions: Tensor<B, 1, Int>,
    pub values: Tensor<B, 1>,
    pub neglogps: Tensor<B, 1>,
    pub returns: Tensor<B, 1>,
}

impl<B: Backend> Batcher<Experience, ExperienceBatch<B>> for ExperienceBatcher<B> {
    fn batch(&self, items: Vec<Experience>) -> ExperienceBatch<B> {
        let observations = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats(item.observation.as_slice(), &self.device))
            .map(|tensor| tensor.reshape([1, -1]))
            .collect();

        let actions: Vec<u32> = items.iter().map(|item| item.action).collect();
        let values: Vec<f32> = items.iter().map(|item| item.value).collect();
        let neglogps: Vec<f32> = items.iter().map(|item| item.neglogp).collect();
        let returns: Vec<f32> = items.iter().map(|item| item.ret).collect();

        let observations = Tensor::cat(observations, 0).to_device(&self.device);
        let actions = Tensor::<B, 1, Int>::from_ints(actions.as_slice(), &self.device);
        let values = Tensor::<B, 1>::from_floats(values.as_slice(), &self.device);
        let neglogps = Tensor::<B, 1>::from_floats(neglogps.as_slice(), &self.device);
        let returns = Tensor::<B, 1>::from_floats(returns.as_slice(), &self.device);

        ExperienceBatch {
            observations,
            actions,
            values,
            neglogps,
            returns,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{backend::NdArray, data::dataloader::DataLoaderBuilder};

    use crate::common::ExperienceBuffer;

    use super::*;

    #[test]
    fn batcher_sanity() {
        let mut exp_buf = ExperienceBuffer::<2, 3>::new(3);

        let device = Default::default();

        let obs1 = Tensor::from_floats([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], &device);
        exp_buf.add_experience::<NdArray>(
            obs1,
            vec![0.1, 1.1],
            vec![1, 2],
            Tensor::from_floats([3.0, 6.0], &device),
            vec![false, false],
            Tensor::from_floats([20.0, 21.0], &device),
        );

        let (observations, actions, values, neglogps) = exp_buf.training_views();

        let returns = exp_buf
            .returns::<NdArray>(Tensor::from_floats([12.0, 15.0], &device), vec![true, true]);

        let training_view = TrainingView {
            observations,
            actions,
            values,
            neglogps,
            returns,
        };

        let batcher = ExperienceBatcher::<NdArray>::new(device.clone());
        let dataloader = DataLoaderBuilder::new(batcher)
            .batch_size(1)
            .shuffle(123)
            .build(training_view);
        for batch in dataloader.iter() {
            assert_eq!(batch.observations.dims()[0], 1);
        }
    }
}
