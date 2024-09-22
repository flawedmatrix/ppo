use candle_core::{Result, Tensor, D};
use candle_nn::ops::log_softmax;
use tracing::instrument;

#[instrument]
pub fn neglog_probs(logits: &Tensor, actions: &Tensor) -> Result<Tensor> {
    let logits = logits.detach();
    let log_probs = log_softmax(&logits, D::Minus1)?;
    log_probs
        .gather(&actions.unsqueeze(D::Minus1)?, D::Minus1)?
        .neg()?
        .squeeze(1)
}

/// Computes the entropy of a categorical probability distribution.
#[instrument]
pub fn dist_entropy(logits: &Tensor) -> Result<Tensor> {
    let logits = logits.detach();
    let logits_max = logits.max_keepdim(D::Minus1)?;
    let a0 = logits.broadcast_sub(&logits_max)?;
    let exp_a0 = a0.exp()?;
    let z0 = exp_a0.sum_keepdim(D::Minus1)?;
    let p0 = exp_a0.broadcast_div(&z0)?;
    (p0 * z0.log()?.broadcast_sub(&a0)?)?.sum(D::Minus1)
}

#[cfg(test)]
mod tests {
    use std::f32::consts::E;

    use approx::assert_relative_eq;
    use candle_core::{Device, Tensor};

    use super::*;

    #[test]
    fn test_neglog_probs() -> Result<()> {
        let device = Device::Cpu;

        let x = 0.0;
        let y = (E - 1.0).ln();
        // Batch size = 5, Num actions = 2
        let logits = Tensor::from_slice(&[x, y, y, x, x, y, y, x, y, x], &[5, 2], &device)?;
        assert_eq!(
            logits.to_vec2::<f32>()?,
            &[[x, y], [y, x], [x, y], [y, x], [y, x]]
        );

        let actions = Tensor::from_slice(&[1i64, 0, 0, 1, 0], &[5], &device)?;
        let neglogps = neglog_probs(&logits, &actions)?;

        // log_softmax(x) = -ln(e^x / (e^x + e^y)) = -ln(1 / e) = ln(e) = 1
        let lsm_x = 1.0;
        // log_softmax(y) = -ln(e^y / (e^y + e^x)) = -ln((e - 1) / e)
        let lsm_y = -((E - 1.0) / (E)).ln();

        assert_eq!(neglogps.dims(), [5]);
        let neglogps_data: Vec<f32> = neglogps.to_vec1()?;
        let expected_data = vec![lsm_y, lsm_y, lsm_x, lsm_x, lsm_y];
        assert_relative_eq!(neglogps_data.as_slice(), expected_data.as_slice(),);

        Ok(())
    }

    #[test]
    fn test_dist_entropy() -> Result<()> {
        let device = Device::Cpu;

        let logits = Tensor::from_slice(
            &[1f32, 2., 3., 5., 8., 13., 21., 34., 55., 89.],
            &[5, 2],
            &device,
        )?;
        assert_eq!(
            logits.to_vec2::<f32>()?,
            [[1., 2.], [3., 5.], [8., 13.], [21., 34.], [55., 89.]],
        );
        let entropy = dist_entropy(&logits)?;

        assert_eq!(entropy.dims(), [5]);

        Ok(())
    }
}
