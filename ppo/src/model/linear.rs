use dfdx::prelude::*;

use super::ortho_init;

#[derive(Clone, Copy, Debug, Default)]
pub struct OrthoLinearConfig<I: Dim, O: Dim> {
    pub inp: I,
    pub out: O,

    pub ortho_gain: f32,
    pub require_init: bool,
}

impl<I: Dim, O: Dim> OrthoLinearConfig<I, O> {
    pub fn new(inp: I, out: O, ortho_gain: f32, require_init: bool) -> Self {
        Self {
            inp,
            out,
            ortho_gain,
            require_init,
        }
    }
}

/// Compile time sugar alias around [OrthoLinearConfig].
pub type OrthoLinearConstConfig<const I: usize, const O: usize> =
    OrthoLinearConfig<Const<I>, Const<O>>;

impl<I: Dim, O: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for OrthoLinearConfig<I, O> {
    type Built = OrthoLinear<I, O, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, dfdx::tensor::Error> {
        Ok(OrthoLinear {
            weight: device.try_zeros_like(&(self.out, self.inp))?,
            bias: device.try_zeros_like(&(self.out,))?,
            ortho_gain: self.ortho_gain,
            require_init: self.require_init,
        })
    }
}

#[derive(Clone, Debug, UpdateParams, ZeroGrads, SaveSafeTensors, LoadSafeTensors)]
pub struct OrthoLinear<I: Dim, O: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[serialize]
    pub weight: Tensor<(O, I), Elem, Dev>,
    #[param]
    #[serialize]
    pub bias: Tensor<(O,), Elem, Dev>,

    pub ortho_gain: f32,
    pub require_init: bool,
}

impl<const I: usize, const O: usize, E: Dtype, D> ResetParams<E, D>
    for OrthoLinear<Const<I>, Const<O>, E, D>
where
    D: Device<E> + Device<f32>,
{
    fn try_reset_params(&mut self) -> Result<(), dfdx::tensor::Error> {
        if !self.require_init {
            return Ok(());
        }
        self.weight = ortho_init(self.ortho_gain, self.weight.device().clone()).try_to_dtype()?;
        Ok(())
    }
}

impl<S: Shape, I: Dim, O: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>>
    for OrthoLinear<I, O, E, D>
where
    Tensor<S, E, D, T>: TryMatMul<Tensor<(I, O), E, D, T>>,
    Bias1D<O, E, D>: Module<<Tensor<S, E, D, T> as TryMatMul<Tensor<(I, O), E, D, T>>>::Output>,
{
    type Output = <Bias1D<O, E, D> as Module<
        <Tensor<S, E, D, T> as TryMatMul<Tensor<(I, O), E, D, T>>>::Output,
    >>::Output;
    fn try_forward(&self, x: Tensor<S, E, D, T>) -> Result<Self::Output, Error> {
        let weight = self.weight.retaped::<T>().try_permute()?;
        let bias = Bias1D {
            bias: self.bias.clone(),
        };
        let y = x.try_matmul(weight)?;
        bias.try_forward(y)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    const W: [[f64; 5]; 2] = [
        [-0.3458893, -0.30371523, -0.3712057, 0.14303583, -0.0268966],
        [0.11733949, 0.14059687, -0.10670426, -0.09373143, 0.18974298],
    ];
    const B: [f64; 2] = [0.3765365, -0.290717];

    #[test]
    fn test_forward_1d() {
        let dev = Cpu::default();

        let model = Linear {
            weight: dev.tensor(W).to_dtype::<f32>(),
            bias: dev.tensor(B).to_dtype::<f32>(),
        };

        let x = dev
            .tensor([-0.8808001, 2.4185333, 2.2478335, 0.0565211, 2.031299])
            .to_dtype::<f32>();
        let y = model.forward(x.leaky_trace());
        assert_relative_eq!(y.array()[..], [-0.93430865, 0.08624211]);

        let g = y.square().mean().backward();
        let weight_grad = g.get(&model.weight);
        assert_relative_eq!(
            weight_grad.array()[0][..],
            [0.82293916, -2.2596567, -2.1001704, -0.05280815, -1.8978603],
        );
        assert_relative_eq!(
            weight_grad.array()[1][..],
            [-0.07596206, 0.20857942, 0.19385791, 0.004874499, 0.17518352],
        );
        let bias_grad = g.get(&model.bias);
        assert_relative_eq!(bias_grad.array()[..], [-0.93430865, 0.08624211]);
    }
}
