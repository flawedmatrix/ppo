use dfdx::prelude::*;
use rand_distr::Uniform;

use super::{linear::OrthoLinearConstConfig, util::neglog_probs};

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

impl<const OBS_SIZE: usize, const NUM_ACTIONS: usize, const HIDDEN_DIM: usize, D: Device<f32>>
    PolicyNetwork<OBS_SIZE, NUM_ACTIONS, HIDDEN_DIM, f32, D>
{
    /// Runs an inference step of the model with critic and negative log probs
    /// and optionally randomize action selection among the best choices
    ///
    /// Input: [num_envs, OBS_SIZE]
    /// Output:
    ///  - Critic: [num_envs]
    ///  - Chosen Action: [num_envs]
    ///  - Negative Log Probs: [num_envs]
    pub fn infer(
        &self,
        obs: &[[f32; OBS_SIZE]],
        action_mask: Option<[bool; NUM_ACTIONS]>,
        randomize: bool,
        cpu_device: &Cpu,
    ) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
        let num_envs = obs.len();

        let obs_tensor: Tensor<(usize, Const<OBS_SIZE>), _, _> =
            cpu_device.tensor_from_vec(obs.as_flattened().to_vec(), (num_envs, Const::<OBS_SIZE>));

        let model_device = self.device();

        let (critic, actor) = self.forward(obs_tensor.to_device(&model_device));

        let (critic, actor) = (critic.to_device(cpu_device), actor.to_device(cpu_device));

        let actor = match action_mask {
            Some(m) => {
                let mask = m.iter().map(|&x| !x as u8 as f32).collect::<Vec<f32>>();
                let neg_mask = cpu_device.tensor_from_vec(mask, (1, Const::<NUM_ACTIONS>)) * 500.0;
                actor - neg_mask
            }
            None => actor,
        };

        // Get uniform distribution on [0, 1) in the shape of logits

        let len = critic.shape().concrete()[0];

        // Get uniform distribution on [0, 1) in the shape of logits
        let dist = Uniform::new(0f32, 1f32);
        let u = cpu_device.sample_like(actor.shape(), dist);

        // Sample 1 action from actor probs
        let logprobs = if randomize {
            actor.clone() - (-u.ln()).ln()
        } else {
            actor.clone()
        };

        let mut actions: Vec<usize> = Vec::with_capacity(len);
        for probs in logprobs.as_vec().chunks(NUM_ACTIONS) {
            actions.push(argmax(probs));
        }

        let actions_tensor = cpu_device.tensor_from_vec(actions.clone(), (len,));
        let neglogp = neglog_probs(actor, actions_tensor);

        let vf = critic.reshape_like(&(len,));

        (vf.as_vec(), actions, neglogp.as_vec())
    }

    pub fn device(&self) -> D {
        self.input.0.weight.device().clone()
    }
}

fn argmax<T: PartialOrd>(x: &[T]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap()
}
