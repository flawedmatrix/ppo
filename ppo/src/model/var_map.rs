use candle_core::{DType, Device, Error, Result, Shape, Tensor, Var};
use candle_nn::var_builder::SimpleBackend;

use rand_distr::{Distribution, StandardNormal};

use ndarray::prelude::*;

/// TensorMap with the ability to initialize tensors with Orthogonal
/// initialization.
///
/// The gain value in the orthogonal initialization is provided by using the
/// `Const` variant of the `Init` enum.
#[derive(Clone)]
pub struct VarMap(candle_nn::VarMap);

impl VarMap {
    pub fn new() -> Self {
        Self(candle_nn::VarMap::new())
    }

    pub fn all_vars(&self) -> Vec<Var> {
        self.0.all_vars()
    }
}

impl SimpleBackend for VarMap {
    fn get(
        &self,
        s: Shape,
        name: &str,
        init: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> Result<Tensor> {
        if self.0.contains_tensor(name) {
            return self.0.get(s, name, init, dtype, dev);
        }

        let tensor = if name.ends_with("weight") {
            let gain = match init {
                candle_nn::Init::Const(val) => val as f32,
                _ => 2f32.sqrt(),
            };
            ortho_init(&s, gain)?.to_device(dev)
        } else if name.ends_with("bias") {
            Tensor::zeros(&s, dtype, dev)
        } else {
            Err(candle_core::Error::CannotFindTensor {
                path: name.to_string(),
            }
            .bt())
        }?;
        tensor.to_device(dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        if name.ends_with("weight") || name.ends_with("bias") {
            return true;
        }
        self.0.contains_tensor(name)
    }
}

#[cfg(not(feature = "blas"))]
use linfa_linalg::svd::SVD;

#[cfg(feature = "blas")]
use ndarray_linalg::{svddc::SVDDC, JobSvd};

/// Initializes a tensor using the Orthogonal init defined here:
/// https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py#L20
/// (Also the Orthogonal initializer implemented in Lasagne)
///
/// Only supports tensors with 2 dimensions. Might or might not be usable
/// for tensors with more than 2 dimensions
fn ortho_init<S: Into<Shape>>(s: S, gain: f32) -> Result<Tensor> {
    let mut rng = rand::thread_rng();

    let shape: Shape = s.into();

    if shape.rank() != 2 {
        return Err(candle_core::Error::UnexpectedNumberOfDims {
            expected: 2,
            got: shape.rank(),
            shape: shape.clone(),
        }
        .bt());
    }

    let (rows, cols) = (shape.dims()[0], shape.dims()[1]);

    let dist = StandardNormal;
    let gaussian_noise =
        Array2::<f32>::from_shape_simple_fn((rows, cols), move || dist.sample(&mut rng));

    let (opt_u, _, opt_vt) = {
        #[cfg(not(feature = "blas"))]
        {
            gaussian_noise.svd(true, true).map_err(Error::wrap)?
        }
        #[cfg(feature = "blas")]
        {
            gaussian_noise.svddc(JobSvd::Some).map_err(Error::wrap)?
        }
    };

    let u = opt_u.unwrap();
    let v = opt_vt.unwrap();

    let q = if u.shape() == [rows, cols] { u } else { v };

    let q = gain * q;

    Tensor::from_slice(
        q.as_standard_layout()
            .as_slice()
            .expect("generated weights should be in standard layout"),
        &[rows, cols],
        &Device::Cpu,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::Result;
    use candle_nn::{linear, VarBuilder};

    #[test]
    fn ortho_init_default() -> Result<()> {
        let backend = VarMap::new();
        let vb = VarBuilder::from_backend(Box::new(backend), DType::F32, Device::Cpu);

        let layer = linear(10, 10, vb)?;
        let q = layer.weight();
        let res = q.clone().matmul(&q.t()?)?;

        // For some reason the diag mask returns false only along the diagonal,
        // so we have to inverse it
        let diag = (Tensor::eye(10, DType::F32, &Device::Cpu)? * 2.)?;

        // q * qT should be approximately equal to a diagonal matrix of 2's.
        let abs_diff = f32::abs(res.sub(&diag)?.sum_all()?.to_scalar()?);
        assert!(abs_diff < 1e-5);

        let bias_val = f32::abs(
            layer
                .bias()
                .expect("Bias should be created")
                .sum_all()?
                .to_scalar()?,
        );

        assert!(bias_val < 1e-10);

        Ok(())
    }

    #[test]
    fn ortho_init_thin() -> Result<()> {
        let device = Device::Cpu;

        let backend = VarMap::new();
        let vb = VarBuilder::from_backend(Box::new(backend), DType::F32, device);

        let layer = linear(10, 1, vb)?;
        let q = layer.weight();

        // Ensure weights are able to be generated in this case.
        assert_eq!(q.dims(), [1, 10]);

        Ok(())
    }
}
