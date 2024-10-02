use dfdx::{shapes::Rank2, tensor::Tensor, tensor_ops::Device};

use rand_distr::{Distribution, StandardNormal};

use ndarray::prelude::*;

#[cfg(not(feature = "blas"))]
use linfa_linalg::svd::SVD;

#[cfg(feature = "blas")]
use ndarray_linalg::{svddc::SVDDC, JobSvd};

/// Initializes a tensor using the Orthogonal init defined here:
/// https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py#L20
/// (Also the Orthogonal initializer implemented in Lasagne)
pub fn ortho_init<const IN_DIM: usize, const OUT_DIM: usize, D: Device<f32>>(
    scale: f32,
    device: D,
) -> Tensor<Rank2<OUT_DIM, IN_DIM>, f32, D> {
    let mut rng = rand::thread_rng();

    let dist = StandardNormal;

    let shape = (IN_DIM, OUT_DIM);
    let gaussian_noise = Array2::<f32>::from_shape_simple_fn(shape, move || dist.sample(&mut rng));

    let (opt_u, _, opt_vt) = {
        #[cfg(not(feature = "blas"))]
        {
            gaussian_noise.svd(true, true).unwrap()
        }
        #[cfg(feature = "blas")]
        {
            gaussian_noise.svddc(JobSvd::Some).unwrap()
        }
    };

    let u = opt_u.unwrap();
    let v = opt_vt.unwrap();

    let weights = if u.shape() == [IN_DIM, OUT_DIM] { u } else { v };
    let weights = scale
        * weights
            .into_shape(shape)
            .unwrap()
            .t()
            .as_standard_layout()
            .to_owned();

    device.tensor_from_vec(weights.into_raw_vec(), Rank2::<OUT_DIM, IN_DIM>::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    use dfdx::prelude::AsArray;
    use dfdx::prelude::*;

    #[test]
    fn ortho_init_default() {
        let dev = Cpu::default();

        let q = ortho_init::<10, 10, _>(2f32.sqrt(), dev.clone());
        let qt = q.clone().permute::<Rank2<10, 10>, _>();

        let res = q.matmul(qt);

        let id = ndarray::Array2::<f32>::eye(10) * 2.;
        let diag = dev.tensor_from_vec(id.into_raw_vec(), Rank2::<10, 10>::default());

        // q * qT should be approximately equal to a diagonal matrix of 2's.
        let abs_diff = f32::abs((res - diag).sum::<Rank0, _>().array());

        assert!(abs_diff < 1e-5);
    }

    #[test]
    fn ortho_init_thin() {
        let dev = Cpu::default();

        let q = ortho_init::<10, 1, _>(2f32.sqrt(), dev.clone());

        // Ensure weights are able to be generated in this case.
        assert_eq!(q.shape(), &(Const::<1>, Const::<10>));

        let qt = q.clone().permute::<Rank2<10, 1>, _>();

        let res = q.matmul(qt).reshape::<()>();

        approx::assert_relative_eq!(res.array(), 2., epsilon = 1e-5);
    }
}
