use burn::{
    optim::{adaptor::OptimizerAdaptor, Adam, GradientsParams, Optimizer},
    prelude::*,
    tensor::{activation::log_softmax, backend::AutodiffBackend},
};
use tracing::{instrument, trace_span};

use crate::data::ExperienceBatch;

use super::{ModelConfig, PolicyModel};

#[derive(Default, Debug)]
pub struct TrainingStats {
    pub pg_loss: f32,
    pub vf_loss: f32,
    pub entropy: f32,
    pub approxkl: f32,
    pub clipfrac: f32,
    pub explained_variance: f32,
}

pub struct Learner<B: AutodiffBackend> {
    pub model: PolicyModel<B>,
    pub optim: OptimizerAdaptor<Adam<B::InnerBackend>, PolicyModel<B>, B>,

    pub span: tracing::Span,
    pub config: ModelConfig,
}

impl<B: AutodiffBackend> Learner<B> {
    pub fn step(&mut self, batch: ExperienceBatch<B>) -> TrainingStats {
        let _enter = self.span.enter();
        let mut stats = TrainingStats::default();

        let batch_size = batch.observations.dims()[0];

        let (values, policy_latent) = self.model.forward(batch.observations);

        let neglogps = neglog_probs(policy_latent.clone(), batch.actions);

        let nlp_diff = neglogps - batch.neglogps;
        trace_span!("approxkl").in_scope(|| {
            stats.approxkl = nlp_diff.clone().powi_scalar(2).mean().into_scalar().elem();
        });

        let entropy = dist_entropy(policy_latent.clone()).mean();
        trace_span!("entropy").in_scope(|| {
            stats.entropy = entropy.clone().into_scalar().elem();
        });

        let values_clipped = (values.clone() - batch.values.clone())
            .clamp(-self.config.clip_range, self.config.clip_range)
            + batch.values.clone();

        let vf_losses1 = (values.clone() - batch.returns.clone()).powi_scalar(2);
        let vf_losses2 = (values_clipped.clone() - batch.returns.clone()).powi_scalar(2);

        let vf_loss = vf_losses1.max_pair(vf_losses2).mean() * 0.5;
        stats.vf_loss = vf_loss.clone().into_scalar().elem();

        let ratio = (-nlp_diff).exp();

        trace_span!("clip_frac").in_scope(|| {
            stats.clipfrac = {
                let gt_mask = (ratio.clone() - 1.0)
                    .abs()
                    .greater_elem(self.config.clip_range);
                gt_mask.int().sum().into_scalar().elem::<i32>() as f32 / batch_size as f32
            };
        });

        let advs = trace_span!("advs").in_scope(|| {
            let a = batch.returns.clone() - batch.values.clone();
            let (a_var, a_mean) = a.clone().var_mean(0);
            (a - a_mean) / (a_var + 1e-8)
        });

        let pg_losses1 = ratio.clone() * -advs.clone();
        let pg_losses2 =
            ratio.clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * -advs;
        let pg_loss = pg_losses1.max_pair(pg_losses2).mean();

        trace_span!("pg_loss").in_scope(|| {
            stats.pg_loss = pg_loss.clone().into_scalar().elem();
        });

        let loss = pg_loss - (entropy * self.config.entropy_coefficient)
            + (vf_loss * self.config.vf_coefficient);

        let grads = trace_span!("loss.backward").in_scope(|| loss.backward());

        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = trace_span!("optim.step")
            .in_scope(|| self.optim.step(self.config.lr, self.model.clone(), grads));

        stats
    }
}

#[instrument]
pub(super) fn neglog_probs<B: Backend>(
    logits: Tensor<B, 2>,
    actions: Tensor<B, 1, Int>,
) -> Tensor<B, 1> {
    let logits = logits.detach();
    let log_probs = log_softmax(logits, 1);
    -(log_probs.gather(1, actions.unsqueeze_dim(1))).squeeze(1)
}

#[instrument]
fn dist_entropy<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 1> {
    let logits = logits.detach();
    let logits_max = logits.clone().max_dim(1);
    let a0 = logits - logits_max;
    let ea0 = a0.clone().exp();
    let z0 = ea0.clone().sum_dim(1);
    let p0 = ea0 / z0.clone();
    (p0 * (z0.log() - a0)).sum_dim(1).squeeze(1)
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    #[test]
    fn test_neglog_probs() {
        let device = Default::default();

        let e = 1f32.exp();
        let e2 = 2f32.exp();
        // Batch size = 5, Num actions = 2
        let logits = Tensor::<NdArray, 2>::from_floats(
            [[e, e2], [e2, e], [e, e2], [e2, e], [e2, e]],
            &device,
        );
        let actions = Tensor::<NdArray, 1, Int>::from_ints([1, 0, 0, 1, 0], &device);

        let neglogps = neglog_probs(logits, actions);
        assert_eq!(neglogps.dims(), [5]);
        let neglogps_data = neglogps.to_data().to_vec::<f32>().unwrap();
        assert_eq!(neglogps_data[0], neglogps_data[1]); // -log_softmax(e2)
        assert!(neglogps_data[2] != neglogps_data[0]); // -log_softmax(e) != -log_softmax(e2)
        assert_eq!(neglogps_data[2], neglogps_data[3]); // -log_softmax(e)
        assert_eq!(neglogps_data[4], neglogps_data[0]); // -log_softmax(e2)
    }

    #[test]
    fn test_dist_entropy() {
        let device = Default::default();

        let logits = Tensor::<NdArray, 2>::from_floats(
            [[1., 2.], [3., 5.], [8., 13.], [21., 34.], [55., 89.]],
            &device,
        );
        let entropy = dist_entropy(logits);

        assert_eq!(entropy.dims(), [5]);
    }
}
