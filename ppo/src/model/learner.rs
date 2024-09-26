use candle_core::{DType, Result, Tensor};
use candle_nn::{AdamW, Optimizer};
use tracing::trace_span;

use crate::data::ExperienceBatch;

use super::{
    util::{dist_entropy, neglog_probs},
    ModelConfig, PolicyModel,
};

#[derive(Default, Debug)]
pub struct TrainingStats {
    pub pg_loss: f32,
    pub vf_loss: f32,
    pub entropy: f32,
    pub approxkl: f32,
    pub clipfrac: f32,
    pub explained_variance: f32,
}

pub struct Learner {
    pub model: PolicyModel,
    pub optim: AdamW,

    pub span: tracing::Span,
    pub config: ModelConfig,
}

/// Computes the mean and unbiased standard deviation over all elements in the tensor
fn mean_and_std(values: &Tensor) -> Result<(Tensor, Tensor)> {
    let mean = values.mean_all()?;
    let squares = values.broadcast_sub(&mean)?.sqr()?;
    let sum: Tensor = squares.sum_all()?;
    let var: Tensor = (sum / (values.elem_count() - 1) as f64)?;
    Ok((mean, var))
}

impl Learner {
    pub fn step(&mut self, batch: ExperienceBatch) -> Result<TrainingStats> {
        let _enter = self.span.enter();
        let mut stats = TrainingStats::default();

        let batch_size = batch.observations.dims()[0];

        let advs = trace_span!("advs").in_scope(|| {
            let rv_diff = (&batch.returns - &batch.values)?;
            let (a_mean, a_std) = mean_and_std(&rv_diff)?;
            rv_diff.broadcast_sub(&a_mean)?.broadcast_div(&a_std)
        })?;

        let (values, policy_latent) = self.model.forward_critic_actor(&batch.observations)?;

        let neglogps = neglog_probs(&policy_latent, &batch.actions)?;

        let nlp_diff = (neglogps - batch.neglogps)?;
        stats.approxkl = trace_span!("approxkl")
            .in_scope(|| -> Result<f32> { nlp_diff.sqr()?.mean_all()?.to_scalar() })?;

        let entropy = trace_span!("entropy").in_scope(|| -> Result<Tensor> {
            let e = dist_entropy(&policy_latent)?.mean_all()?;
            stats.entropy = e.to_scalar()?;
            Ok(e)
        })?;

        let values_clipped = ((&values - &batch.values)?
            .clamp(-self.config.clip_range, self.config.clip_range)
            + &batch.values)?;

        let vf_losses1 = (values - &batch.returns)?.sqr()?;
        let vf_losses2 = (values_clipped - &batch.returns)?.sqr()?;

        let vf_loss = (vf_losses1.maximum(&vf_losses2)?.mean_all()? * 0.5)?;
        stats.vf_loss = vf_loss.to_scalar()?;

        let ratio = nlp_diff.neg()?.exp()?;

        stats.clipfrac = trace_span!("clip_frac").in_scope(|| -> Result<f32> {
            let gt_mask = (&ratio - 1.0)?.abs()?.gt(self.config.clip_range)?;
            let gt_mask = gt_mask.to_dtype(DType::F32)?;
            Ok(gt_mask.sum_all()?.to_scalar::<f32>()? / batch_size as f32)
        })?;

        let neg_advs = advs.neg()?;

        let pg_losses1 = (&ratio * &neg_advs)?;
        let pg_losses2 =
            (ratio.clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range)? * neg_advs)?;
        let pg_loss = pg_losses1.maximum(&pg_losses2)?.mean_all()?;

        stats.pg_loss = trace_span!("pg_loss").in_scope(|| pg_loss.to_scalar())?;

        let loss = ((pg_loss - (entropy * self.config.entropy_coefficient)?)?
            + (vf_loss * self.config.vf_coefficient))?;

        let grads = trace_span!("loss.backward").in_scope(|| loss.backward())?;
        trace_span!("optim.step").in_scope(|| self.optim.step(&grads))?;

        Ok(stats)
    }
}
