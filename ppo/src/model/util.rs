use dfdx::prelude::*;

pub(super) fn neglog_probs<B: Dim, const NUM_ACTIONS: usize, D, T>(
    logits: Tensor<(B, Const<NUM_ACTIONS>), f32, D, T>,
    actions: Tensor<(B,), usize, D>,
) -> Tensor<(B,), f32, D, T>
where
    D: Device<f32>,
    T: Tape<f32, D>,
{
    let log_probs = logits.log_softmax::<Axis<1>>();
    -log_probs.select(actions)
}

/// Computes the entropy of a categorical probability distribution.
pub(super) fn dist_entropy<B: Dim, const NUM_ACTIONS: usize, D, T>(
    logits: Tensor<(B, Const<NUM_ACTIONS>), f32, D, T>,
) -> Tensor<(B,), f32, D, T>
where
    D: Device<f32>,
    T: Tape<f32, D>,
{
    let len = logits.shape().0;
    let logits_max = logits.with_empty_tape().max::<_, Axis<1>>();
    let a0 = logits - logits_max.broadcast_like(&(len, Const));
    let ea0 = a0.with_empty_tape().exp();
    let z0 = ea0
        .with_empty_tape()
        .sum::<_, Axis<1>>()
        .broadcast_like(&(len, Const));
    let p0 = ea0 / z0.with_empty_tape();
    (p0 * (z0.ln() - a0)).sum::<_, Axis<1>>()
}
#[cfg(test)]
mod tests {
    use std::f32::consts::E;

    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_neglog_probs() {
        let device = Cpu::default();

        let x = 0.0;
        let y = (E - 1.0).ln();
        // Batch size = 5, Num actions = 2
        let logits = device.tensor([[x, y], [y, x], [x, y], [y, x], [y, x]]);

        let actions = device.tensor([1usize, 0, 0, 1, 0]);
        let neglogps = neglog_probs(logits, actions);

        // log_softmax(x) = -ln(e^x / (e^x + e^y)) = -ln(1 / e) = ln(e) = 1
        let lsm_x = 1.0;
        // log_softmax(y) = -ln(e^y / (e^y + e^x)) = -ln((e - 1) / e)
        let lsm_y = -((E - 1.0) / (E)).ln();

        let neglogps_data: Vec<f32> = neglogps.as_vec();
        let expected_data = vec![lsm_y, lsm_y, lsm_x, lsm_x, lsm_y];
        assert_relative_eq!(neglogps_data.as_slice(), expected_data.as_slice(),);
    }

    #[test]
    fn test_dist_entropy() {
        let device = Cpu::default();

        let logits: Tensor<(usize, Const<2>), _, _> = device.tensor_from_vec(
            vec![1., 2., 3., 5., 8., 13., 21., 34., 55., 89.],
            (5, Const),
        );
        let entropy = dist_entropy(logits);

        assert_eq!(entropy.shape().concrete(), [5]);
    }
}
