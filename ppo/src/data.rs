use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use ndarray::{s, ArrayView1, ArrayView2};

/// A temporary container for training views. The views should live for as long
/// as the Experience Buffer persists.
pub struct TrainingView<'a> {
    observations: ArrayView2<'a, f32>,
    actions: ArrayView1<'a, i32>,
    values: ArrayView1<'a, f32>,
    neglogps: ArrayView1<'a, f32>,
    returns: ArrayView1<'a, f32>,
}

pub struct Experience {
    observation: Vec<f32>,
    action: i32,
    value: f32,
    neglogp: f32,
    ret: f32,
}

impl<'a> Dataset<Experience> for TrainingView<'a> {
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

        let actions: Vec<i32> = items.iter().map(|item| item.action).collect();
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
