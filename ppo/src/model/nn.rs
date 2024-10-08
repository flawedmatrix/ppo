use dfdx::prelude::*;

use super::linear::OrthoLinearConstConfig;

#[derive(Default, Clone, Sequential)]
#[built(PolicyNetwork)]
pub struct PolicyNetworkConfig<
    const OBS_SIZE: usize,
    const NUM_ACTIONS: usize,
    const HIDDEN_DIM: usize,
> {
    input: (OrthoLinearConstConfig<OBS_SIZE, HIDDEN_DIM>, ReLU),
    hidden: Vec<(OrthoLinearConstConfig<HIDDEN_DIM, HIDDEN_DIM>, ReLU)>,
    // (critic, actor)
    output: SplitInto<(
        OrthoLinearConstConfig<HIDDEN_DIM, 1>,
        OrthoLinearConstConfig<HIDDEN_DIM, NUM_ACTIONS>,
    )>,
}

impl<const OBS_SIZE: usize, const NUM_ACTIONS: usize, const HIDDEN_DIM: usize>
    PolicyNetworkConfig<OBS_SIZE, NUM_ACTIONS, HIDDEN_DIM>
{
    pub fn new(num_layers: usize) -> Self {
        let sqrt_2 = f32::sqrt(2.);
        Self {
            input: (
                OrthoLinearConstConfig::new(Const::<OBS_SIZE>, Const::<HIDDEN_DIM>, sqrt_2),
                ReLU,
            ),
            hidden: vec![
                (
                    OrthoLinearConstConfig::new(Const::<HIDDEN_DIM>, Const::<HIDDEN_DIM>, sqrt_2),
                    ReLU
                );
                num_layers
            ],
            output: SplitInto((
                OrthoLinearConstConfig::new(Const::<HIDDEN_DIM>, Const::<1>, 1.0),
                OrthoLinearConstConfig::new(Const::<HIDDEN_DIM>, Const::<NUM_ACTIONS>, 0.1),
            )),
        }
    }
}
