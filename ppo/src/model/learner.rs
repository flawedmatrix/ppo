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

impl Learner {
    pub fn step(&mut self, batch: ExperienceBatch) -> Result<TrainingStats> {
        let _enter = self.span.enter();
        let mut stats = TrainingStats::default();

        let batch_size = batch.observations.dims()[0];

        let (values, policy_latent) = self.model.forward_critic_actor(&batch.observations)?;

        let neglogps = neglog_probs(&policy_latent, &batch.actions)?;

        let nlp_diff = (neglogps - batch.neglogps)?;
        stats.approxkl = trace_span!("approxkl")
            .in_scope(|| -> Result<f32> { nlp_diff.powf(2.0)?.mean_all()?.to_vec0() })?;

        let entropy = trace_span!("entropy").in_scope(|| -> Result<Tensor> {
            let e = dist_entropy(&policy_latent)?.mean_all()?;
            stats.entropy = e.to_vec0()?;
            Ok(e)
        })?;

        let values_clipped = ((&values - &batch.values)?
            .clamp(-self.config.clip_range, self.config.clip_range)
            + &batch.values)?;

        let vf_losses1 = (values - &batch.returns)?.powf(2.0)?;
        let vf_losses2 = (values_clipped - &batch.returns)?.powf(2.0)?;

        let vf_loss = (vf_losses1.maximum(&vf_losses2)?.mean_all()? * 0.5)?;
        stats.vf_loss = vf_loss.to_vec0()?;

        let ratio = nlp_diff.neg()?.exp()?;

        stats.clipfrac = trace_span!("clip_frac").in_scope(|| -> Result<f32> {
            let gt_mask = (ratio.clone() - 1.0)?.abs()?.gt(self.config.clip_range)?;
            let gt_mask = gt_mask.to_dtype(DType::F32)?;
            Ok(gt_mask.sum_all()?.to_vec0::<f32>()? / batch_size as f32)
        })?;

        let advs = trace_span!("advs").in_scope(|| {
            let a = (batch.returns - batch.values)?;
            let (a_var, a_mean) = (a.var_keepdim(0)?, a.mean_keepdim(0)?);
            a.broadcast_sub(&a_mean)?.broadcast_div(&(a_var + 1e-8)?)
        })?;

        let pg_losses1 = (&ratio * advs.neg()?)?;
        let pg_losses2 = (ratio
            .clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range)?
            * advs.neg()?)?;
        let pg_loss = pg_losses1.maximum(&pg_losses2)?.mean_all()?;

        stats.pg_loss = trace_span!("pg_loss").in_scope(|| pg_loss.to_vec0())?;

        let loss = ((pg_loss - (entropy * self.config.entropy_coefficient)?)?
            + (vf_loss * self.config.vf_coefficient))?;

        trace_span!("optim.step").in_scope(|| self.optim.backward_step(&loss))?;

        Ok(stats)
    }
}
