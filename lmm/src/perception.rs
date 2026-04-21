// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Multi-Modal Perception
//!
//! This module provides [`MultiModalPerception`], a stateless sensor adaptor that converts
//! raw byte streams into normalised floating-point [`Tensor`]s suitable for downstream
//! processing by the [`crate::consciousness::Consciousness`] loop and other `lmm` subsystems.
//!
//! Each byte value `b ∈ [0, 255]` is mapped to `b / 255.0 ∈ [0.0, 1.0]`, preserving the
//! full dynamic range in a format compatible with gradient-based operations.

use crate::error::LmmError::Perception;
use crate::error::Result;
use crate::tensor::Tensor;
use crate::traits::Perceivable;

/// A stateless multi-modal perception adaptor.
///
/// `MultiModalPerception` acts as the boundary between raw sensory data (byte streams from
/// images, audio frames, sensor readings, etc.) and the symbolic tensor world used throughout
/// `lmm`. It normalises every byte to `[0.0, 1.0]` and wraps the result in a flat [`Tensor`].
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::perception::MultiModalPerception;
/// use lmm::traits::Perceivable;
///
/// let raw = &[0u8, 128, 255];
/// let tensor = MultiModalPerception::ingest(raw).unwrap();
/// assert_eq!(tensor.shape, vec![3]);
/// assert!((tensor.data[0] - 0.0).abs() < 1e-9);
/// assert!((tensor.data[1] - 128.0 / 255.0).abs() < 1e-9);
/// assert!((tensor.data[2] - 1.0).abs() < 1e-9);
/// ```
pub struct MultiModalPerception;

impl Perceivable for MultiModalPerception {
    /// Converts a raw byte slice into a normalised perception [`Tensor`].
    ///
    /// Each byte `b` maps to `b as f64 / 255.0`, producing values in `[0.0, 1.0]`.
    /// The resulting tensor has shape `[raw_data.len()]`.
    ///
    /// # Arguments
    ///
    /// * `raw_data` - A non-empty byte slice representing any sensory modality.
    ///
    /// # Returns
    ///
    /// (`Result<Tensor>`): A flat, normalised tensor of the same length as the input.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::LmmError::Perception`] when `raw_data` is empty.
    ///
    /// # Time Complexity
    ///
    /// O(n) where n is the number of bytes.
    ///
    /// # Space Complexity
    ///
    /// O(n) for the output tensor data.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::perception::MultiModalPerception;
    /// use lmm::traits::Perceivable;
    ///
    /// assert!(MultiModalPerception::ingest(&[]).is_err());
    ///
    /// let t = MultiModalPerception::ingest(&[255u8]).unwrap();
    /// assert_eq!(t.data[0], 1.0);
    /// ```
    fn ingest(raw_data: &[u8]) -> Result<Tensor> {
        if raw_data.is_empty() {
            return Err(Perception("Empty input data".into()));
        }
        let float_data: Vec<f64> = raw_data.iter().map(|&b| f64::from(b) / 255.0).collect();
        Tensor::new(vec![float_data.len()], float_data)
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
