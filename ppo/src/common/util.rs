use burn::{
    module::{Param, ParamId},
    nn::Linear,
    prelude::*,
};

/// Combines two linear output heads into one Linear layer along the output dimension.
/// Assumes zero initialization for the bias.
pub fn combine_linear_heads<B: Backend>(
    layer1: Linear<B>,
    layer2: Linear<B>,
    device: &B::Device,
) -> Linear<B> {
    let (_, layer1_weights) = layer1.weight.consume();
    let (_, layer2_weights) = layer2.weight.consume();
    let output_weights = Tensor::cat(vec![layer1_weights, layer2_weights], 1);

    let output_size = output_weights.dims()[1];

    Linear {
        weight: Param::uninitialized(
            ParamId::new(),
            move |_, require_grad| {
                let mut tensor = output_weights.clone();
                if require_grad {
                    tensor = tensor.require_grad();
                }
                tensor
            },
            device.clone(),
            true,
        ),
        bias: Some(Param::uninitialized(
            ParamId::new(),
            move |device, require_grad| {
                let mut tensor = Tensor::zeros([output_size], device);
                if require_grad {
                    tensor = tensor.require_grad();
                }
                tensor
            },
            device.clone(),
            true,
        )),
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use nn::LinearConfig;

    use super::*;

    #[test]
    fn combined_heads_behave_like_two() {
        let device = Default::default();

        let head1 = LinearConfig::new(5, 1)
            .with_bias(false)
            .with_initializer(nn::Initializer::Constant { value: 2. })
            .init::<NdArray>(&device);
        let head2 = LinearConfig::new(5, 3)
            .with_bias(false)
            .with_initializer(nn::Initializer::Constant { value: 1. })
            .init::<NdArray>(&device);

        // Maps the input [1, 2, 3, 4, 5] to [30] and [15, 15, 15]

        let combined_heads = combine_linear_heads::<NdArray>(head1, head2, &device);
        let output = combined_heads.forward(Tensor::<NdArray, 1>::from_floats(
            [1., 2., 3., 4., 5.],
            &device,
        ));

        let expected = Tensor::<NdArray, 1>::from_floats([30., 15., 15., 15.], &device);

        output.to_data().assert_approx_eq(&expected.to_data(), 7);
    }
}
