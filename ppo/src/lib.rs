mod common;
mod data;
mod model;
mod runner;
mod training;

use std::{fmt::Debug, hash::Hash};

/// An environment that supports being vectorized, performing actions, and
/// scoring a particular state.
///
/// Currently only supports 1D environments.
pub trait Environment<const OBS_SIZE: usize, const NUM_ACTIONS: usize>:
    Eq + Hash + Debug + Copy + Clone
{
    /// Returns a vectorized snapshot of the environment state
    fn as_vector(&self) -> [f32; OBS_SIZE];
    /// A boolean mask of valid actions, where the index of the mask corresponds
    /// to the action ID
    fn valid_actions(&self) -> [bool; NUM_ACTIONS];
    /// Step number of the environment
    fn step_num(&self) -> i64;
    /// Returns true if the Environment is in a finished or "game over"
    /// state
    fn is_done(&self) -> bool;
    /// Performs the action associated with the action ID
    fn do_action(&mut self, action_id: usize);
    /// Returns a score (could be a game score or a heuristic evaluation) of
    /// the environment
    fn score(&self) -> f32;
}
