// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Calibrated Uncertainty Quantification
//!
//! This module provides first-class support for representing and propagating
//! epistemic uncertainty throughout the `lmm` inference pipeline.
//!
//! ## Core types
//!
//! | Type | Role |
//! |---|---|
//! | [`BeliefDistribution`] | Gaussian belief `(μ, σ²)` over a single variable |
//! | [`UncertaintyPropagator`] | Propagates variance through a [`CausalGraph`] |
//! | [`CalibrationRecord`] | Tracks Brier scores for long-run calibration monitoring |
//!
//! ## Design motivation
//!
//! A system that reports only point predictions cannot distinguish between
//! "I am confident this is 3.0" and "my best guess is 3.0 but I know nothing".
//! [`BeliefDistribution`] makes that distinction explicit, and
//! [`UncertaintyPropagator`] ensures that uncertainty compounds correctly as
//! it flows through causal chains - matching Pearl's linearised propagation rules.
//!
//! ## See Also
//!
//! - [Pearl, J. (2009). Causality. Cambridge University Press.](https://www.cambridge.org/core/books/causality/B0046844FAE10CBF274D4ACBDAEB5F5B)
//! - [Gneiting, T. & Raftery, A. E. (2007). Strictly Proper Scoring Rules.](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437)

use crate::causal::CausalGraph;
use crate::error::Result;
use std::collections::HashMap;

/// A Gaussian belief over a single continuous variable.
///
/// [`BeliefDistribution`] represents the agent's current state of knowledge
/// about one quantity as a normal distribution `N(mean, variance)`.  A
/// `variance` of `0.0` expresses complete certainty; higher values encode
/// proportionally more ignorance.
///
/// # Examples
///
/// ```
/// use lmm::uncertainty::BeliefDistribution;
///
/// let belief = BeliefDistribution::new(5.0, 0.25);
/// let (lo, hi) = belief.confidence_interval(1.96);
/// assert!(lo < 5.0 && hi > 5.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefDistribution {
    /// The expected value (point estimate) of this variable.
    pub mean: f64,
    /// The variance (σ²) expressing epistemic uncertainty.
    pub variance: f64,
}

impl BeliefDistribution {
    /// Constructs a new [`BeliefDistribution`] with given mean and variance.
    ///
    /// # Arguments
    ///
    /// * `mean`     - Point estimate of the variable.
    /// * `variance` - Non-negative uncertainty; clamped to zero if negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::uncertainty::BeliefDistribution;
    ///
    /// let b = BeliefDistribution::new(3.0, 1.0);
    /// assert_eq!(b.mean, 3.0);
    /// assert_eq!(b.variance, 1.0);
    /// ```
    pub fn new(mean: f64, variance: f64) -> Self {
        Self {
            mean,
            variance: variance.max(0.0),
        }
    }

    /// Constructs a maximally uncertain belief centred on `mean`.
    ///
    /// The variance is set to `f64::MAX / 2.0` - effectively uninformative.
    pub fn uninformative(mean: f64) -> Self {
        Self::new(mean, f64::MAX / 2.0)
    }

    /// Returns the standard deviation `σ = √variance`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::uncertainty::BeliefDistribution;
    ///
    /// let b = BeliefDistribution::new(0.0, 4.0);
    /// assert!((b.std_dev() - 2.0).abs() < 1e-9);
    /// ```
    pub fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Returns the `(lower, upper)` bound of a symmetric confidence interval.
    ///
    /// For `z = 1.96` this gives the conventional 95 % interval.
    ///
    /// # Arguments
    ///
    /// * `z` - Number of standard deviations from the mean.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::uncertainty::BeliefDistribution;
    ///
    /// let b = BeliefDistribution::new(10.0, 1.0);
    /// let (lo, hi) = b.confidence_interval(2.0);
    /// assert!((lo - 8.0).abs() < 1e-9);
    /// assert!((hi - 12.0).abs() < 1e-9);
    /// ```
    pub fn confidence_interval(&self, z: f64) -> (f64, f64) {
        let half_width = z * self.std_dev();
        (self.mean - half_width, self.mean + half_width)
    }

    /// Returns `true` when the point `value` falls inside the `z`-σ interval.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::uncertainty::BeliefDistribution;
    ///
    /// let b = BeliefDistribution::new(5.0, 1.0);
    /// assert!(b.contains(5.5, 1.0));
    /// assert!(!b.contains(10.0, 1.0));
    /// ```
    pub fn contains(&self, value: f64, z: f64) -> bool {
        let (lo, hi) = self.confidence_interval(z);
        value >= lo && value <= hi
    }

    /// Fuses this belief with another via Bayesian precision-weighted update.
    ///
    /// When two independent observers both have Gaussian beliefs about the same
    /// variable, the combined posterior is given by the precision-weighted mean
    /// and the sum-of-precisions inverse.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::uncertainty::BeliefDistribution;
    ///
    /// let a = BeliefDistribution::new(4.0, 1.0);
    /// let b = BeliefDistribution::new(6.0, 1.0);
    /// let fused = a.fuse(&b);
    /// assert!((fused.mean - 5.0).abs() < 1e-9);
    /// ```
    pub fn fuse(&self, other: &BeliefDistribution) -> BeliefDistribution {
        let prec_self = if self.variance > 0.0 {
            1.0 / self.variance
        } else {
            f64::MAX
        };
        let prec_other = if other.variance > 0.0 {
            1.0 / other.variance
        } else {
            f64::MAX
        };
        let prec_total = prec_self + prec_other;
        let fused_mean = (prec_self * self.mean + prec_other * other.mean) / prec_total;
        let fused_variance = if prec_total > 0.0 {
            1.0 / prec_total
        } else {
            0.0
        };
        BeliefDistribution::new(fused_mean, fused_variance)
    }
}

impl Default for BeliefDistribution {
    fn default() -> Self {
        Self::new(0.0, 1.0)
    }
}

/// Propagates [`BeliefDistribution`]s through a [`CausalGraph`] in topological order.
///
/// For each child node, its output variance is computed as the sum of squared-coefficient
/// weighted parent variances - the standard linearised error-propagation rule.  The mean
/// is the same as the forward-pass sum `Σ coeff · parent_mean`.
///
/// # Examples
///
/// ```
/// use lmm::causal::CausalGraph;
/// use lmm::uncertainty::{BeliefDistribution, UncertaintyPropagator};
///
/// let mut g = CausalGraph::new();
/// g.add_node("x", Some(3.0));
/// g.add_node("y", None);
/// g.add_edge("x", "y", Some(2.0)).unwrap();
///
/// let mut beliefs = std::collections::HashMap::new();
/// beliefs.insert("x".to_string(), BeliefDistribution::new(3.0, 0.5));
///
/// let result = UncertaintyPropagator::propagate(&g, &beliefs).unwrap();
/// let y_belief = &result["y"];
/// assert!((y_belief.mean - 6.0).abs() < 1e-9);
/// assert!((y_belief.variance - 4.0 * 0.5).abs() < 1e-9); // coeff² * σ²
/// ```
pub struct UncertaintyPropagator;

impl UncertaintyPropagator {
    /// Propagates beliefs through `graph`, returning a map of all node beliefs.
    ///
    /// Nodes with no initial belief entry are assigned `BeliefDistribution::default()`.
    /// Root nodes keep their supplied beliefs unchanged; derived child beliefs are computed
    /// from the linearised propagation formula `σ²_child = Σ (coeff² · σ²_parent)`.
    ///
    /// # Arguments
    ///
    /// * `graph`   - The causal structure to propagate through.
    /// * `initial` - Seed beliefs for any subset of nodes.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::CausalError`] when the graph contains a cycle.
    pub fn propagate(
        graph: &CausalGraph,
        initial: &HashMap<String, BeliefDistribution>,
    ) -> Result<HashMap<String, BeliefDistribution>> {
        let order = graph.topological_order()?;

        let mut beliefs: HashMap<String, BeliefDistribution> = graph
            .nodes
            .iter()
            .map(|n| {
                let b = initial
                    .get(&n.name)
                    .cloned()
                    .unwrap_or_else(|| BeliefDistribution::new(n.value.unwrap_or(0.0), 1.0));
                (n.name.clone(), b)
            })
            .collect();

        let mut parent_map: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        for edge in &graph.edges {
            parent_map
                .entry(edge.to.clone())
                .or_default()
                .push((edge.from.clone(), edge.coefficient.unwrap_or(1.0)));
        }

        for name in &order {
            if let Some(parents) = parent_map.get(name) {
                let all_known = parents.iter().all(|(p, _)| beliefs.contains_key(p));
                if !all_known {
                    continue;
                }
                let mean: f64 = parents
                    .iter()
                    .map(|(p, coeff)| beliefs[p].mean * coeff)
                    .sum();
                let variance: f64 = parents
                    .iter()
                    .map(|(p, coeff)| coeff * coeff * beliefs[p].variance)
                    .sum();
                beliefs.insert(name.clone(), BeliefDistribution::new(mean, variance));
            }
        }

        Ok(beliefs)
    }
}

/// A single calibration observation: a predicted interval and the realized scalar.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationObservation {
    /// Lower bound of the predicted confidence interval.
    pub lower: f64,
    /// Upper bound of the predicted confidence interval.
    pub upper: f64,
    /// The value that was actually observed.
    pub realized: f64,
}

impl CalibrationObservation {
    /// Returns `true` when `realized` falls inside `[lower, upper]`.
    pub fn is_hit(&self) -> bool {
        self.realized >= self.lower && self.realized <= self.upper
    }
}

/// Accumulates calibration evidence and computes aggregate quality scores.
///
/// A well-calibrated system that predicts 95 % intervals should see approximately
/// 95 % of realized values fall inside those intervals over many observations.
/// [`CalibrationRecord`] tracks this empirically.
///
/// # Examples
///
/// ```
/// use lmm::uncertainty::{CalibrationRecord, CalibrationObservation};
///
/// let mut rec = CalibrationRecord::default();
/// rec.observe(CalibrationObservation { lower: 0.0, upper: 2.0, realized: 1.0 });
/// rec.observe(CalibrationObservation { lower: 0.0, upper: 2.0, realized: 5.0 });
/// assert!((rec.hit_rate() - 0.5).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Default)]
pub struct CalibrationRecord {
    observations: Vec<CalibrationObservation>,
}

impl CalibrationRecord {
    /// Creates an empty [`CalibrationRecord`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one calibration observation.
    pub fn observe(&mut self, obs: CalibrationObservation) {
        self.observations.push(obs);
    }

    /// Returns the fraction of realized values that fell inside the predicted interval.
    ///
    /// Returns `0.0` when no observations have been recorded.
    pub fn hit_rate(&self) -> f64 {
        if self.observations.is_empty() {
            return 0.0;
        }
        let hits = self.observations.iter().filter(|o| o.is_hit()).count();
        hits as f64 / self.observations.len() as f64
    }

    /// Returns the mean Brier-style calibration loss: average squared miss distance.
    ///
    /// For hits the loss is `0.0`; for misses it is the squared distance from the
    /// nearest interval boundary to the realized value, normalised by interval width.
    ///
    /// Returns `0.0` when no observations exist.
    pub fn calibration_loss(&self) -> f64 {
        if self.observations.is_empty() {
            return 0.0;
        }
        let total: f64 = self
            .observations
            .iter()
            .map(|o| {
                if o.is_hit() {
                    0.0
                } else {
                    let width = (o.upper - o.lower).abs().max(1e-12);
                    let miss = if o.realized < o.lower {
                        o.lower - o.realized
                    } else {
                        o.realized - o.upper
                    };
                    (miss / width).powi(2)
                }
            })
            .sum();
        total / self.observations.len() as f64
    }

    /// Returns the total number of observations recorded.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    /// Returns `true` when no observations have been recorded.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
