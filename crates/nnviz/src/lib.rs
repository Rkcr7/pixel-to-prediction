//! nnviz — the neural core behind "Pixel to Prediction".
//!
//! One crate does three jobs from a single source of truth:
//!   1. trains the model natively (`src/bin/train.rs`)
//!   2. runs inference in the browser via WASM, keeping every intermediate tensor
//!   3. computes real gradient attributions live, so the explanation shown to a user is
//!      derived from the model rather than written by a copywriter
//!
//! There is no train/serve skew because it is the same code on both sides.
//!
//! Two clippy style lints are turned off deliberately rather than worked around:
//! `needless_range_loop`, because a convolution's loop index addresses several
//! differently shaped buffers at once and an iterator chain obscures that, and
//! `identity_op`, because the shape assertions are written as `5 * 5 * 1 * 8` to mirror
//! the architecture rather than to compute a number.
#![allow(clippy::needless_range_loop, clippy::identity_op)]

pub mod explain;
pub mod layers;
pub mod net;
pub mod preprocess;
pub mod tensor;

#[cfg(not(target_arch = "wasm32"))]
pub mod data;
#[cfg(not(target_arch = "wasm32"))]
pub mod train;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use net::{Acts, Net};
pub use tensor::Vol;
