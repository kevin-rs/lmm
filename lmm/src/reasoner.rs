// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Compositional Symbolic Reasoner
//!
//! This module implements forward-chaining deductive reasoning from first principles,
//! enabling the system to derive novel conclusions by composing axioms rather than
//! interpolating training data.
//!
//! ## Core types
//!
//! | Type | Role |
//! |---|---|
//! | [`SymbolicAxiom`] | A named implication rule antecedents → consequent |
//! | [`DeductionEngine`] | Forward-chains axioms to fixed-point over a fact base |
//! | [`CompositeProof`] | Ordered derivation steps produced by `DeductionEngine::prove` |
//!
//! ## Design motivation
//!
//! Statistical systems interpolate over observed examples; they cannot construct a
//! proof because they have no axioms or inference rules.  [`DeductionEngine`] operates
//! purely from declared rules: given a fact base `{A, B}` and a rule `A ∧ B → C`,
//! it deduces `C` mechanistically.  This process is fully auditable via
//! [`CompositeProof`], which records every derivation step.
//!
//! ## See Also
//!
//! - [Modus ponens - Wikipedia](https://en.wikipedia.org/wiki/Modus_ponens)
//! - [Forward chaining - Wikipedia](https://en.wikipedia.org/wiki/Forward_chaining)

use crate::error::{LmmError, Result};
use std::collections::HashSet;

/// An implication rule: if all `antecedents` are known facts, derive `consequent`.
///
/// Optionally, `weight` can record the logical strength of the rule (default 1.0),
/// allowing the engine to prefer higher-confidence derivations when multiple rules yield
/// the same consequent.
///
/// # Examples
///
/// ```
/// use lmm::reasoner::SymbolicAxiom;
///
/// let axiom = SymbolicAxiom::new(
///     "transitivity",
///     vec!["A_causes_B".into(), "B_causes_C".into()],
///     "A_causes_C",
/// );
/// assert_eq!(axiom.consequent, "A_causes_C");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolicAxiom {
    /// Human-readable identifier for this rule.
    pub name: String,
    /// Propositions that must all be known before the consequent can be derived.
    pub antecedents: Vec<String>,
    /// The proposition derived when all antecedents are satisfied.
    pub consequent: String,
    /// Confidence weight ∈ (0, 1]; higher = more reliable rule.
    pub weight: f64,
}

impl SymbolicAxiom {
    /// Constructs a new axiom with full confidence (`weight = 1.0`).
    ///
    /// # Arguments
    ///
    /// * `name`         - Descriptive identifier for the rule.
    /// * `antecedents`  - Required precondition propositions.
    /// * `consequent`   - The derived proposition.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::reasoner::SymbolicAxiom;
    ///
    /// let ax = SymbolicAxiom::new("modus_ponens", vec!["P".into(), "P_implies_Q".into()], "Q");
    /// assert_eq!(ax.weight, 1.0);
    /// ```
    pub fn new(
        name: impl Into<String>,
        antecedents: Vec<String>,
        consequent: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            antecedents,
            consequent: consequent.into(),
            weight: 1.0,
        }
    }

    /// Attaches a custom confidence weight to this axiom.
    ///
    /// `weight` is clamped to `(0.0, 1.0]`.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight.clamp(f64::MIN_POSITIVE, 1.0);
        self
    }

    /// Returns `true` when every antecedent appears in `facts`.
    pub fn is_applicable(&self, facts: &HashSet<String>) -> bool {
        self.antecedents.iter().all(|a| facts.contains(a))
    }
}

/// A single step in a deductive proof.
///
/// Records which rule fired, which antecedents it consumed, and what new fact
/// was derived.
#[derive(Debug, Clone, PartialEq)]
pub struct DeductionStep {
    /// Name of the axiom that fired in this step.
    pub axiom_name: String,
    /// Antecedent propositions that were satisfied.
    pub from: Vec<String>,
    /// The newly derived proposition.
    pub derived: String,
    /// Confidence inherited from the firing axiom.
    pub confidence: f64,
}

/// The complete audit trail of a successful (or partial) deduction.
///
/// # Examples
///
/// ```
/// use lmm::reasoner::{DeductionEngine, SymbolicAxiom};
///
/// let mut engine = DeductionEngine::new(10);
/// engine.register(SymbolicAxiom::new("r1", vec!["A".into()], "B"));
/// engine.register(SymbolicAxiom::new("r2", vec!["B".into()], "C"));
///
/// let proof = engine.prove(&["A".into()], "C").unwrap();
/// assert!(proof.succeeded);
/// assert_eq!(proof.steps.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct CompositeProof {
    /// Whether the target conclusion was successfully derived.
    pub succeeded: bool,
    /// The conclusion that was targeted.
    pub target: String,
    /// Ordered derivation steps from start to finish.
    pub steps: Vec<DeductionStep>,
    /// All facts known at termination (initial facts ∪ derived facts).
    pub final_facts: HashSet<String>,
}

impl CompositeProof {
    /// Returns the minimum confidence across all derivation steps.
    ///
    /// Returns `1.0` when no steps were required (trivial proofs).
    pub fn proof_confidence(&self) -> f64 {
        self.steps
            .iter()
            .map(|s| s.confidence)
            .fold(1.0_f64, f64::min)
    }
}

/// Forward-chaining deductive reasoner over a registered set of [`SymbolicAxiom`]s.
///
/// At each iteration, every applicable axiom fires and its consequent is added to
/// the fact base.  This repeats until either:
/// - The target proposition appears in the fact base (success), or
/// - No new facts are derivable (fixed-point without target), or
/// - `max_depth` iterations are exhausted.
///
/// # Examples
///
/// ```
/// use lmm::reasoner::{DeductionEngine, SymbolicAxiom};
///
/// let mut engine = DeductionEngine::new(10);
/// engine.register(SymbolicAxiom::new("heating_expands", vec!["heated".into()], "volume_increased"));
/// engine.register(SymbolicAxiom::new("expansion_reduces_density", vec!["volume_increased".into()], "density_reduced"));
///
/// let proof = engine.prove(&["heated".into()], "density_reduced").unwrap();
/// assert!(proof.succeeded);
/// assert_eq!(proof.steps[0].derived, "volume_increased");
/// assert_eq!(proof.steps[1].derived, "density_reduced");
/// ```
#[derive(Debug, Clone, Default)]
pub struct DeductionEngine {
    axioms: Vec<SymbolicAxiom>,
    max_depth: usize,
}

impl DeductionEngine {
    /// Creates a new engine with the given iteration cap.
    ///
    /// # Arguments
    ///
    /// * `max_depth` - Maximum forward-chaining iterations (clamped to ≥ 1).
    pub fn new(max_depth: usize) -> Self {
        Self {
            axioms: Vec::new(),
            max_depth: max_depth.max(1),
        }
    }

    /// Registers an axiom with this engine.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::reasoner::{DeductionEngine, SymbolicAxiom};
    ///
    /// let mut engine = DeductionEngine::new(5);
    /// engine.register(SymbolicAxiom::new("r", vec!["X".into()], "Y"));
    /// assert_eq!(engine.axiom_count(), 1);
    /// ```
    pub fn register(&mut self, axiom: SymbolicAxiom) {
        self.axioms.push(axiom);
    }

    /// Returns the number of registered axioms.
    pub fn axiom_count(&self) -> usize {
        self.axioms.len()
    }

    /// Attempts to derive `target` from `initial_facts` by forward chaining.
    ///
    /// # Arguments
    ///
    /// * `initial_facts` - The known propositions to start from.
    /// * `target`        - The proposition to prove.
    ///
    /// # Returns
    ///
    /// (`Result<CompositeProof>`): Always `Ok`; `proof.succeeded` indicates
    /// whether `target` was reached.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::Timeout`] when the iteration cap is exceeded without
    /// reaching a fixed-point.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::reasoner::{DeductionEngine, SymbolicAxiom};
    ///
    /// let mut engine = DeductionEngine::new(5);
    /// engine.register(SymbolicAxiom::new("r", vec!["P".into()], "Q"));
    /// let proof = engine.prove(&["P".into()], "Q").unwrap();
    /// assert!(proof.succeeded);
    /// ```
    pub fn prove(&self, initial_facts: &[String], target: &str) -> Result<CompositeProof> {
        let mut facts: HashSet<String> = initial_facts.iter().cloned().collect();
        let mut steps: Vec<DeductionStep> = Vec::new();

        if facts.contains(target) {
            return Ok(CompositeProof {
                succeeded: true,
                target: target.to_string(),
                steps,
                final_facts: facts,
            });
        }

        for depth in 0..self.max_depth {
            let mut new_derivations: Vec<(String, &SymbolicAxiom)> = self
                .axioms
                .iter()
                .filter(|ax| ax.is_applicable(&facts) && !facts.contains(&ax.consequent))
                .map(|ax| (ax.consequent.clone(), ax))
                .collect();

            new_derivations.sort_by(|a, b| {
                b.1.weight
                    .partial_cmp(&a.1.weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if new_derivations.is_empty() {
                return Ok(CompositeProof {
                    succeeded: false,
                    target: target.to_string(),
                    steps,
                    final_facts: facts,
                });
            }

            for (derived, axiom) in &new_derivations {
                steps.push(DeductionStep {
                    axiom_name: axiom.name.clone(),
                    from: axiom.antecedents.clone(),
                    derived: derived.clone(),
                    confidence: axiom.weight,
                });
                facts.insert(derived.clone());
            }

            if facts.contains(target) {
                return Ok(CompositeProof {
                    succeeded: true,
                    target: target.to_string(),
                    steps,
                    final_facts: facts,
                });
            }

            if depth + 1 >= self.max_depth {
                return Err(LmmError::Timeout);
            }
        }

        Ok(CompositeProof {
            succeeded: false,
            target: target.to_string(),
            steps,
            final_facts: facts,
        })
    }

    /// Returns `true` when the target is trivially in the initial facts, without chaining.
    pub fn is_known(&self, facts: &[String], target: &str) -> bool {
        facts.iter().any(|f| f == target)
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
