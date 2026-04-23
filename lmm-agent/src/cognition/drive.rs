// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Internalized Drive Architecture
//!
//! This module implements the fifth missing intelligence property:
//! **internalized motivation**.  Where classical agents wait for an external
//! reward signal to direct behaviour, a genuinely intelligent system has
//! intrinsic drives that guide attention from within:
//!
//! - **Curiosity** fires when the agent encounters high residuals - unexplained
//!   observations that its causal model cannot account for.
//! - **CoherenceSeeking** fires when the knowledge base contains internally
//!   inconsistent entries (measured by inter-chunk contradictions).
//! - **ContradictionResolution** fires when two memory entries assert conflicting
//!   facts about the same subject.
//!
//! These signals modulate the [`crate::cognition::r#loop::ThinkLoop`] goal by
//! biasing which knowledge gaps are prioritized next, independently of any
//! external teacher.
//!
//! ## See Also
//!
//! - [Intrinsic motivation - Wikipedia](https://en.wikipedia.org/wiki/Motivation#Intrinsic_and_extrinsic)
//! - [Curiosity-driven exploration (Schmidhuber, 1991)](https://en.wikipedia.org/wiki/Intrinsic_motivation_(artificial_intelligence))

/// A discrete intrinsic motivation signal with an associated urgency magnitude.
///
/// # Examples
///
/// ```
/// use lmm_agent::cognition::drive::DriveSignal;
///
/// let sig = DriveSignal::Curiosity(0.85);
/// assert_eq!(sig.magnitude(), 0.85);
/// assert_eq!(sig.name(), "Curiosity");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum DriveSignal {
    /// Unexplained observations detected; the system should investigate further.
    Curiosity(f64),
    /// Internal knowledge is inconsistent; the system should reconcile beliefs.
    CoherenceSeeking(f64),
    /// Two known facts contradict each other; the system must resolve the conflict.
    ContradictionResolution(f64),
}

impl DriveSignal {
    /// Returns the urgency magnitude ∈ [0, 1].
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm_agent::cognition::drive::DriveSignal;
    /// assert!((DriveSignal::CoherenceSeeking(0.4).magnitude() - 0.4).abs() < 1e-9);
    /// ```
    pub fn magnitude(&self) -> f64 {
        match self {
            Self::Curiosity(m) | Self::CoherenceSeeking(m) | Self::ContradictionResolution(m) => *m,
        }
    }

    /// Returns the human-readable name of this drive type.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Curiosity(_) => "Curiosity",
            Self::CoherenceSeeking(_) => "CoherenceSeeking",
            Self::ContradictionResolution(_) => "ContradictionResolution",
        }
    }
}

/// A snapshot of the agent's current motivational state across all active drives.
///
/// # Examples
///
/// ```
/// use lmm_agent::cognition::drive::{DriveSignal, DriveState};
///
/// let state = DriveState {
///     signals: vec![
///         DriveSignal::Curiosity(0.9),
///         DriveSignal::CoherenceSeeking(0.3),
///     ],
/// };
/// assert_eq!(state.dominant_drive().unwrap().name(), "Curiosity");
/// assert!((state.total_urgency() - 1.2).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Default)]
pub struct DriveState {
    /// All active drive signals for this tick.
    pub signals: Vec<DriveSignal>,
}

impl DriveState {
    /// Returns the signal with the highest urgency, or `None` when idle.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm_agent::cognition::drive::{DriveSignal, DriveState};
    ///
    /// let state = DriveState { signals: vec![DriveSignal::Curiosity(0.5)] };
    /// assert!(state.dominant_drive().is_some());
    /// ```
    pub fn dominant_drive(&self) -> Option<&DriveSignal> {
        self.signals.iter().max_by(|a, b| {
            a.magnitude()
                .partial_cmp(&b.magnitude())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns the sum of all signal magnitudes.
    pub fn total_urgency(&self) -> f64 {
        self.signals.iter().map(|s| s.magnitude()).sum()
    }

    /// Returns `true` when no signals are active (all magnitudes are below `threshold`).
    pub fn is_idle(&self, threshold: f64) -> bool {
        self.signals.iter().all(|s| s.magnitude() < threshold)
    }
}

/// Stateful intrinsic motivator that produces [`DriveState`] from internal observations.
///
/// [`InternalDrive`] maintains three internal counters that it updates each `tick`:
///
/// - **unexplained_residual**: sum of absolute prediction errors seen this episode.
/// - **incoherence_count**: number of memory entries flagged as internally inconsistent.
/// - **contradiction_count**: number of detected fact-pair contradictions.
///
/// Each `tick` converts these counters into normalised signals and resets the counters.
///
/// # Examples
///
/// ```
/// use lmm_agent::cognition::drive::InternalDrive;
///
/// let mut drive = InternalDrive::new(1.0, 10.0, 5);
/// drive.record_residual(0.8);
/// drive.record_residual(0.6);
/// let state = drive.tick();
/// assert!(!state.signals.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct InternalDrive {
    /// Normalisation constant for residuals (max expected residual per tick).
    pub residual_scale: f64,
    /// Normalisation constant for incoherence counts.
    pub incoherence_scale: f64,
    /// Normalisation constant for contradiction counts.
    pub contradiction_scale: usize,

    unexplained_residual: f64,
    incoherence_count: f64,
    contradiction_count: usize,
}

impl InternalDrive {
    /// Constructs a new [`InternalDrive`] with the given normalisation scales.
    ///
    /// # Arguments
    ///
    /// * `residual_scale`     - Expected maximum per-tick residual magnitude.
    /// * `incoherence_scale`  - Expected maximum incoherence count per tick.
    /// * `contradiction_scale`- Expected maximum contradiction count per tick.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm_agent::cognition::drive::InternalDrive;
    ///
    /// let drive = InternalDrive::new(1.0, 5.0, 3);
    /// assert_eq!(drive.residual_scale, 1.0);
    /// ```
    pub fn new(residual_scale: f64, incoherence_scale: f64, contradiction_scale: usize) -> Self {
        Self {
            residual_scale: residual_scale.max(1e-9),
            incoherence_scale: incoherence_scale.max(1e-9),
            contradiction_scale: contradiction_scale.max(1),
            unexplained_residual: 0.0,
            incoherence_count: 0.0,
            contradiction_count: 0,
        }
    }

    /// Accumulates an unexplained prediction residual toward the next tick.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm_agent::cognition::drive::InternalDrive;
    ///
    /// let mut d = InternalDrive::new(1.0, 1.0, 1);
    /// d.record_residual(0.5);
    /// let state = d.tick();
    /// assert!(state.signals.iter().any(|s| s.name() == "Curiosity"));
    /// ```
    pub fn record_residual(&mut self, magnitude: f64) {
        self.unexplained_residual += magnitude.abs();
    }

    /// Increments the internal incoherence evidence counter.
    pub fn record_incoherence(&mut self, magnitude: f64) {
        self.incoherence_count += magnitude.abs();
    }

    /// Increments the contradiction counter by one detected pair.
    pub fn record_contradiction(&mut self) {
        self.contradiction_count += 1;
    }

    /// Converts accumulated evidence into a [`DriveState`] and resets all counters.
    ///
    /// Signals with magnitude `< 0.01` are suppressed to avoid noise.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm_agent::cognition::drive::InternalDrive;
    ///
    /// let mut d = InternalDrive::new(2.0, 5.0, 10);
    /// d.record_residual(2.5);
    /// d.record_incoherence(3.0);
    /// d.record_contradiction();
    /// let state = d.tick();
    /// assert_eq!(state.signals.len(), 3);
    /// ```
    pub fn tick(&mut self) -> DriveState {
        let curiosity = (self.unexplained_residual / self.residual_scale).clamp(0.0, 1.0);
        let coherence = (self.incoherence_count / self.incoherence_scale).clamp(0.0, 1.0);
        let contradiction =
            (self.contradiction_count as f64 / self.contradiction_scale as f64).clamp(0.0, 1.0);

        self.unexplained_residual = 0.0;
        self.incoherence_count = 0.0;
        self.contradiction_count = 0;

        let mut signals = Vec::with_capacity(3);
        if curiosity >= 0.01 {
            signals.push(DriveSignal::Curiosity(curiosity));
        }
        if coherence >= 0.01 {
            signals.push(DriveSignal::CoherenceSeeking(coherence));
        }
        if contradiction >= 0.01 {
            signals.push(DriveSignal::ContradictionResolution(contradiction));
        }

        DriveState { signals }
    }

    /// Returns the most urgent drive signal from the next tick without resetting counters.
    ///
    /// Useful for peeking at the motivational state without committing to a tick.
    pub fn peek_dominant(&self) -> Option<DriveSignal> {
        let curiosity = (self.unexplained_residual / self.residual_scale).clamp(0.0, 1.0);
        let coherence = (self.incoherence_count / self.incoherence_scale).clamp(0.0, 1.0);
        let contradiction =
            (self.contradiction_count as f64 / self.contradiction_scale as f64).clamp(0.0, 1.0);

        let candidates = [
            DriveSignal::Curiosity(curiosity),
            DriveSignal::CoherenceSeeking(coherence),
            DriveSignal::ContradictionResolution(contradiction),
        ];

        candidates
            .into_iter()
            .filter(|s| s.magnitude() >= 0.01)
            .max_by(|a, b| {
                a.magnitude()
                    .partial_cmp(&b.magnitude())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

impl Default for InternalDrive {
    fn default() -> Self {
        Self::new(1.0, 10.0, 5)
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
