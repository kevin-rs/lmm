// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Symbolic Regression
//!
//! This module implements [`SymbolicRegression`], a genetic programming engine that
//! discovers symbolic mathematical expressions from data. It uses:
//!
//! - **Population-based search** - a pool of expression trees is evolved over generations.
//! - **Tournament selection** - the fittest expressions are preferentially propagated.
//! - **Crossover and mutation** - subtree swap and random perturbation diversify the pool.
//! - **MDL scoring** - the [`crate::compression::mdl_score`] criterion balances fit vs.
//!   complexity to prevent over-fitting.
//!
//! The discovered expression can be used directly as a physics law, text-rhythm model,
//! or causal equation within other `lmm` subsystems.
//!
//! # See Also
//! - [Koza, J. R. (1992). Genetic Programming. MIT Press.](https://en.wikipedia.org/wiki/Genetic_programming) - the original formulation of the genetic programming algorithms adapted here.
//! - [`crate::predict::TextPredictor`] - utilizes this `SymbolicRegression` engine to mathematically model language tone and rhythm trajectories.

use crate::compression::mdl_score;
use crate::equation::Expression;
use crate::error::{LmmError, Result};
use crate::tensor::Tensor;
use crate::traits::Discoverable;
use rand::{Rng, RngExt};

/// A genetic-programming symbolic regression engine.
///
/// Evolves a population of mathematical expression trees over many generations,
/// scoring each against training data with the MDL criterion.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::discovery::SymbolicRegression;
/// use lmm::tensor::Tensor;
///
/// let data = vec![
///     Tensor::from_vec(vec![1.0]),
///     Tensor::from_vec(vec![2.0]),
///     Tensor::from_vec(vec![3.0]),
/// ];
/// let targets = vec![2.0, 4.0, 6.0];
/// let expr = SymbolicRegression::new(3, 40).fit(
///     &data.iter().map(|t| t.data.clone()).collect::<Vec<_>>(),
///     &targets,
/// ).unwrap();
/// println!("Discovered: {expr}");
/// ```
pub struct SymbolicRegression {
    /// Maximum depth of expression trees.
    pub max_depth: usize,
    /// Population size (number of candidate expressions per generation).
    pub population_size: usize,
    /// Number of evolutionary iterations.
    pub iterations: usize,
    /// Names of input variables (e.g. `["x", "y"]`).
    pub variable_names: Vec<String>,
}

impl SymbolicRegression {
    /// Creates a [`SymbolicRegression`] with a single default variable `"x"`.
    ///
    /// # Arguments
    ///
    /// * `max_depth` - Maximum depth of expression trees.
    /// * `iterations` - Number of evolutionary generations.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::discovery::SymbolicRegression;
    /// let sr = SymbolicRegression::new(3, 50);
    /// assert_eq!(sr.max_depth, 3);
    /// ```
    pub fn new(max_depth: usize, iterations: usize) -> Self {
        Self {
            max_depth,
            population_size: 50,
            iterations,
            variable_names: vec!["x".into()],
        }
    }

    pub fn with_variables(mut self, vars: Vec<String>) -> Self {
        self.variable_names = vars;
        self
    }

    pub fn with_population(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    fn seed_templates<R: Rng>(&self, rng: &mut R) -> Vec<Expression> {
        let mut templates = Vec::new();
        for var in &self.variable_names {
            let x = Expression::Variable(var.clone());
            let a: f64 = rng.random_range(0.5..=5.0);
            let b: f64 = rng.random_range(-10.0..=10.0);
            let c: f64 = rng.random_range(-5.0..=5.0);
            templates.push(Expression::Add(
                Box::new(Expression::Mul(
                    Box::new(Expression::Constant(a)),
                    Box::new(x.clone()),
                )),
                Box::new(Expression::Constant(b)),
            ));
            templates.push(Expression::Add(
                Box::new(Expression::Add(
                    Box::new(Expression::Mul(
                        Box::new(Expression::Constant(a)),
                        Box::new(Expression::Pow(
                            Box::new(x.clone()),
                            Box::new(Expression::Constant(2.0)),
                        )),
                    )),
                    Box::new(Expression::Mul(
                        Box::new(Expression::Constant(b)),
                        Box::new(x.clone()),
                    )),
                )),
                Box::new(Expression::Constant(c)),
            ));
            templates.push(Expression::Add(
                Box::new(Expression::Mul(
                    Box::new(Expression::Constant(a)),
                    Box::new(Expression::Sin(Box::new(Expression::Mul(
                        Box::new(Expression::Constant(0.1)),
                        Box::new(x.clone()),
                    )))),
                )),
                Box::new(Expression::Constant(b)),
            ));
            templates.push(x);
        }
        templates
    }

    fn has_variables(expr: &Expression) -> bool {
        match expr {
            Expression::Variable(_) => true,
            Expression::Constant(_) => false,
            Expression::Neg(e)
            | Expression::Abs(e)
            | Expression::Sin(e)
            | Expression::Cos(e)
            | Expression::Exp(e)
            | Expression::Log(e) => Self::has_variables(e),
            Expression::Add(l, r)
            | Expression::Sub(l, r)
            | Expression::Mul(l, r)
            | Expression::Div(l, r)
            | Expression::Pow(l, r) => Self::has_variables(l) || Self::has_variables(r),
        }
    }

    fn random_expr<R: Rng>(&self, rng: &mut R, depth: usize) -> Expression {
        if depth >= self.max_depth || (depth > 0 && rng.random_bool(0.4)) {
            return self.random_leaf(rng);
        }
        let choice: u8 = rng.random_range(0..8);
        match choice {
            0 => Expression::Add(
                Box::new(self.random_expr(rng, depth + 1)),
                Box::new(self.random_expr(rng, depth + 1)),
            ),
            1 => Expression::Sub(
                Box::new(self.random_expr(rng, depth + 1)),
                Box::new(self.random_expr(rng, depth + 1)),
            ),
            2 => Expression::Mul(
                Box::new(self.random_expr(rng, depth + 1)),
                Box::new(self.random_expr(rng, depth + 1)),
            ),
            3 => Expression::Div(
                Box::new(self.random_expr(rng, depth + 1)),
                Box::new(self.random_expr(rng, depth + 1)),
            ),
            4 => Expression::Sin(Box::new(self.random_expr(rng, depth + 1))),
            5 => Expression::Cos(Box::new(self.random_expr(rng, depth + 1))),
            6 => Expression::Exp(Box::new(self.random_expr(rng, depth + 1))),
            _ => {
                let b = self.random_expr(rng, depth + 1);
                let exp_val = f64::from(rng.random_range(2u32..=3));
                Expression::Pow(Box::new(b), Box::new(Expression::Constant(exp_val)))
            }
        }
    }

    fn random_leaf<R: Rng>(&self, rng: &mut R) -> Expression {
        if !self.variable_names.is_empty() && rng.random_bool(0.6) {
            let idx = rng.random_range(0..self.variable_names.len());
            Expression::Variable(self.variable_names[idx].clone())
        } else {
            let v: f64 = rng.random_range(-5.0..=5.0);
            Expression::Constant((v * 10.0).round() / 10.0)
        }
    }

    fn mutate<R: Rng>(&self, expr: &Expression, rng: &mut R) -> Expression {
        if rng.random_bool(0.3) {
            return self.random_expr(rng, 0);
        }
        match expr {
            Expression::Constant(c) => {
                let delta: f64 = rng.random_range(-1.0..=1.0);
                Expression::Constant((c + delta * 0.5).clamp(-100.0, 100.0))
            }
            Expression::Variable(_) => self.random_leaf(rng),
            Expression::Add(l, r) => {
                Expression::Add(Box::new(self.mutate(l, rng)), Box::new(self.mutate(r, rng)))
            }
            Expression::Sub(l, r) => {
                Expression::Sub(Box::new(self.mutate(l, rng)), Box::new(self.mutate(r, rng)))
            }
            Expression::Mul(l, r) => {
                Expression::Mul(Box::new(self.mutate(l, rng)), Box::new(self.mutate(r, rng)))
            }
            Expression::Div(l, r) => {
                Expression::Div(Box::new(self.mutate(l, rng)), Box::new(self.mutate(r, rng)))
            }
            Expression::Sin(e) => Expression::Sin(Box::new(self.mutate(e, rng))),
            Expression::Cos(e) => Expression::Cos(Box::new(self.mutate(e, rng))),
            Expression::Exp(e) => Expression::Exp(Box::new(self.mutate(e, rng))),
            Expression::Log(e) => Expression::Log(Box::new(self.mutate(e, rng))),
            Expression::Pow(b, e) => Expression::Pow(Box::new(self.mutate(b, rng)), e.clone()),
            Expression::Neg(e) => Expression::Neg(Box::new(self.mutate(e, rng))),
            Expression::Abs(e) => Expression::Abs(Box::new(self.mutate(e, rng))),
        }
    }

    fn crossover<R: Rng>(
        &self,
        parent_a: &Expression,
        parent_b: &Expression,
        rng: &mut R,
    ) -> Expression {
        if rng.random_bool(0.5) {
            self.swap_subtree(parent_a, parent_b, rng)
        } else {
            self.swap_subtree(parent_b, parent_a, rng)
        }
    }

    fn swap_subtree<R: Rng>(
        &self,
        base: &Expression,
        donor: &Expression,
        rng: &mut R,
    ) -> Expression {
        if rng.random_bool(0.3) {
            return donor.clone();
        }
        match base {
            Expression::Add(l, r) => Expression::Add(
                Box::new(self.swap_subtree(l, donor, rng)),
                Box::new(self.swap_subtree(r, donor, rng)),
            ),
            Expression::Sub(l, r) => Expression::Sub(
                Box::new(self.swap_subtree(l, donor, rng)),
                Box::new(self.swap_subtree(r, donor, rng)),
            ),
            Expression::Mul(l, r) => Expression::Mul(
                Box::new(self.swap_subtree(l, donor, rng)),
                Box::new(self.swap_subtree(r, donor, rng)),
            ),
            Expression::Div(l, r) => Expression::Div(
                Box::new(self.swap_subtree(l, donor, rng)),
                Box::new(self.swap_subtree(r, donor, rng)),
            ),
            Expression::Sin(e) => Expression::Sin(Box::new(self.swap_subtree(e, donor, rng))),
            Expression::Cos(e) => Expression::Cos(Box::new(self.swap_subtree(e, donor, rng))),
            Expression::Exp(e) => Expression::Exp(Box::new(self.swap_subtree(e, donor, rng))),
            Expression::Log(e) => Expression::Log(Box::new(self.swap_subtree(e, donor, rng))),
            Expression::Neg(e) => Expression::Neg(Box::new(self.swap_subtree(e, donor, rng))),
            Expression::Abs(e) => Expression::Abs(Box::new(self.swap_subtree(e, donor, rng))),
            Expression::Pow(b, e) => {
                Expression::Pow(Box::new(self.swap_subtree(b, donor, rng)), e.clone())
            }
            leaf => leaf.clone(),
        }
    }

    fn tournament_select<'a, R: Rng>(
        population: &'a [Expression],
        fitnesses: &[f64],
        rng: &mut R,
        k: usize,
    ) -> &'a Expression {
        let n = population.len();
        let mut best_idx = rng.random_range(0..n);
        for _ in 1..k {
            let idx = rng.random_range(0..n);
            if fitnesses[idx] < fitnesses[best_idx] {
                best_idx = idx;
            }
        }
        &population[best_idx]
    }

    pub fn fit(&self, inputs: &[Vec<f64>], targets: &[f64]) -> Result<Expression> {
        if inputs.is_empty() || targets.is_empty() {
            return Err(LmmError::Discovery("Empty training data".into()));
        }

        let mut rng = rand::rng();
        let templates = self.seed_templates(&mut rng);
        let mut population: Vec<Expression> = templates;
        while population.len() < self.population_size {
            population.push(self.random_expr(&mut rng, 0));
        }

        let has_vars = !self.variable_names.is_empty();
        let initial_candidate = population
            .iter()
            .find(|e| !has_vars || Self::has_variables(e))
            .cloned()
            .unwrap_or_else(|| population[0].clone());
        let mut best_expr = initial_candidate.clone();
        let mut best_score = {
            let s = mdl_score(&initial_candidate, inputs, targets);
            if s.is_finite() { s } else { f64::MAX }
        };

        for _ in 0..self.iterations {
            let fitnesses: Vec<f64> = population
                .iter()
                .map(|e| {
                    let score = mdl_score(e, inputs, targets);
                    if score.is_nan() || score.is_infinite() {
                        1e9
                    } else {
                        score
                    }
                })
                .collect();

            for (i, &score) in fitnesses.iter().enumerate() {
                if score < best_score && (!has_vars || Self::has_variables(&population[i])) {
                    best_score = score;
                    best_expr = population[i].clone();
                }
            }

            let mut new_pop = vec![best_expr.clone()];
            while new_pop.len() < self.population_size {
                let parent_a = Self::tournament_select(&population, &fitnesses, &mut rng, 5);
                let op: u8 = rng.random_range(0..3);
                let child = match op {
                    0 => {
                        let parent_b =
                            Self::tournament_select(&population, &fitnesses, &mut rng, 5);
                        self.crossover(parent_a, parent_b, &mut rng)
                    }
                    1 => self.mutate(parent_a, &mut rng),
                    _ => parent_a.clone(),
                };
                let simplified = child.simplify();
                if has_vars && !Self::has_variables(&simplified) && rng.random_bool(0.7) {
                    new_pop.push(self.random_expr(&mut rng, 0));
                } else {
                    new_pop.push(simplified);
                }
            }
            population = new_pop;
        }

        Ok(best_expr.simplify())
    }
}

impl Discoverable for SymbolicRegression {
    /// Discovers a symbolic expression fitting `data → targets` with default parameters.
    ///
    /// Uses `max_depth = 3`, `iterations = 50`, and a single variable `x`.
    fn discover(data: &[Tensor], targets: &[f64]) -> Result<Expression> {
        if data.is_empty() {
            return Ok(Expression::Variable("x".into()));
        }
        let inputs: Vec<Vec<f64>> = data.iter().map(|t| t.data.clone()).collect();
        let sr = SymbolicRegression::new(3, 50);
        sr.fit(&inputs, targets)
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
