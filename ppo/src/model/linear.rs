use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

/// Linear with some more traceability.
/// The layer applies a linear transformation to the incoming data,
/// `y = x@w.t() + b` with an optional bias.
#[derive(Debug, Clone)]
pub struct LinearWithSpan {
    inner: Linear,
    span: tracing::Span,
}

pub fn linear_with_span(
    in_dim: usize,
    out_dim: usize,
    ortho_gain: f32,
    vb: VarBuilder,
) -> Result<LinearWithSpan> {
    let vb_prefix = vb.prefix();

    let init_ws = candle_nn::Init::Const(ortho_gain as f64);
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let init_bs = candle_nn::Init::Const(0.0);
    let bs = vb.get_with_hints(out_dim, "bias", init_bs)?;

    let span = tracing::span!(tracing::Level::TRACE, "linear", vb_prefix, in_dim, out_dim);
    Ok(LinearWithSpan {
        inner: Linear::new(ws, Some(bs)),
        span,
    })
}

impl Module for LinearWithSpan {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

pub fn critic_actor_heads(
    in_dim: usize,
    num_actions: usize,
    vb: VarBuilder,
) -> Result<(LinearWithSpan, LinearWithSpan)> {
    let vb_critic = vb.pp("critic");
    let vb_critic_prefix = vb_critic.prefix();
    let vb_actor = vb.pp("actor");
    let vb_actor_prefix = vb_actor.prefix();

    let init_ws = candle_nn::Init::Const(1.0);
    let init_bs = candle_nn::Init::Const(0.0);

    let ws_critic = vb_critic.get_with_hints((1, in_dim), "weight", init_ws)?;
    let bs_critic = vb_critic.get_with_hints(1, "bias", init_bs)?;

    let init_ws = candle_nn::Init::Const(0.01);

    let ws_actor = vb_actor.get_with_hints((num_actions, in_dim), "weight", init_ws)?;
    let bs_actor = vb_actor.get_with_hints(num_actions, "bias", init_bs)?;

    let critic_span = tracing::span!(
        tracing::Level::TRACE,
        "critic_head",
        vb_critic_prefix,
        in_dim,
    );

    let actor_span = tracing::span!(
        tracing::Level::TRACE,
        "actor_head",
        vb_actor_prefix,
        in_dim,
        num_actions,
    );

    Ok((
        LinearWithSpan {
            inner: Linear::new(ws_critic, Some(bs_critic)),
            span: critic_span,
        },
        LinearWithSpan {
            inner: Linear::new(ws_actor, Some(bs_actor)),
            span: actor_span,
        },
    ))
}
