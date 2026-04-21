// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Prelude
//!
//! Convenience re-exports for the most commonly used types and traits in `lmm`.
//!
//! Import everything with `use lmm::prelude::*;` to bring the full public API into scope
//! without explicitly naming each module path.
//!
//! # Examples
//!
//! ```
//! use lmm::prelude::*;
//!
//! let t = Tensor::zeros(vec![3]);
//! assert_eq!(t.data.len(), 3);
//! ```

pub use crate::causal::*;
pub use crate::compression::*;
pub use crate::consciousness::*;
pub use crate::discovery::*;
pub use crate::equation::*;
pub use crate::error::*;
pub use crate::field::*;
pub use crate::imagen::*;
pub use crate::models::*;
pub use crate::operator::*;
pub use crate::perception::*;
pub use crate::physics::*;
pub use crate::predict::*;
pub use crate::simulation::*;
pub use crate::symbolic::*;
pub use crate::tensor::*;
pub use crate::traits::*;
pub use crate::world::*;

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
