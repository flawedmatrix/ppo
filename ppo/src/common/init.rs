use burn::{
    module::{Param, ParamId},
    nn::Linear,
    prelude::*,
};
use rand_distr::{Distribution, StandardNormal};

use linfa_linalg::svd::SVD;
use ndarray::prelude::*;

/// Initializes a tensor using the Orthogonal init defined here:
/// https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py#L20
/// (Also the Orthogonal initializer implemented in Lasagne)
///
/// Only supports tensors with 2 dimensions. Might or might not be usable
/// for tensors with more than 2 dimensions.
fn ortho_init<B: Backend, const D: usize, S: Into<Shape<D>>>(
    shape: S,
    gain: f32,
    device: &B::Device,
) -> Tensor<B, D> {
    assert!(
        D >= 2,
        "Can only run orthogonal init tensors with 2 or more dimensions"
    );

    let mut rng = rand::thread_rng();

    let shape: Shape<D> = shape.into();

    let rows = shape.dims[0];
    let num_elems = shape.num_elements();
    let cols = num_elems / rows;

    let dist = StandardNormal;
    let gaussian_noise =
        Array2::<f32>::from_shape_simple_fn((rows, cols), move || dist.sample(&mut rng));

    let (opt_u, _, opt_vt) = gaussian_noise.svd(true, true).unwrap();

    let u = opt_u.unwrap();
    let v = opt_vt.unwrap();

    let q = if u.shape() == [rows, cols] { u } else { v };

    let q = gain * q;

    Tensor::<B, 1>::from_floats(
        q.as_standard_layout()
            .as_slice()
            .expect("generated weights should be in standard layout"),
        device,
    )
    .reshape(shape)
}

/// Configuration to create a Linear layer using Orthogonal initialization for
/// the weights and zero initialzation for the bias.
#[derive(Config, Debug)]
pub struct OrthoLinearConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the output features.
    pub d_output: usize,
    /// If a bias should be applied during the linear transformation.
    #[config(default = true)]
    pub bias: bool,

    pub ortho_init_gain: f32,
}

impl OrthoLinearConfig {
    /// Initialize a new Linear module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Linear<B> {
        let weight = self.init_param([self.d_input, self.d_output], device);
        let bias = if self.bias {
            Some(self.init_param([self.d_output], device))
        } else {
            None
        };

        Linear { weight, bias }
    }

    /// For params of 2 or more dimensions (weights), returns an orthogonal
    /// initialization of the params. For params of 1 dimension (biases), returns
    /// a zero initialization.
    fn init_param<B: Backend, const D: usize, S: Into<Shape<D>>>(
        &self,
        shape: S,
        device: &B::Device,
    ) -> Param<Tensor<B, D>> {
        let device = device.clone();
        let shape: Shape<D> = shape.into();
        let gain = self.ortho_init_gain;

        Param::uninitialized(
            ParamId::new(),
            move |device, require_grad| {
                let mut tensor = if D >= 2 {
                    ortho_init(shape.clone(), gain, device)
                } else {
                    Tensor::<B, D>::zeros(shape.clone(), device)
                };

                if require_grad {
                    tensor = tensor.require_grad();
                }

                tensor
            },
            device,
            true,
        )
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Wgpu;

    use super::*;

    #[test]
    fn ortho_init_default() {
        let device = Default::default();
        let layer = OrthoLinearConfig::new(10, 10, 2f32.sqrt());
        let linear = layer.init::<Wgpu>(&device);
        let q = linear.weight.val();
        let res = q.clone().matmul(q.transpose());

        // For some reason the diag mask returns false only along the diagonal,
        // so we have to inverse it
        let diag = Tensor::<Wgpu, 2, Bool>::diag_mask([10, 10], 0, &device)
            .bool_not()
            .float()
            .mul_scalar(2.);

        // q * qT should be approximately equal to a diagonal matrix of 2's.
        res.to_data().assert_approx_eq(&diag.to_data(), 5);

        linear
            .bias
            .expect("Bias should be present by default")
            .to_data()
            .assert_within_range(-1e-10..1e10);
    }

    #[test]
    fn ortho_init_thin() {
        let device = Default::default();
        let layer = OrthoLinearConfig::new(10, 1, 1.);
        let linear = layer.init::<Wgpu>(&device);
        let q = linear.weight.val();

        // Ensure weights are able to be generated in this case.
        assert_eq!(q.dims(), [10, 1]);
    }
}
