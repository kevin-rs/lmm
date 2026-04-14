use crate::compression::mdl_score;
use crate::equation::Expression;
use crate::error::{LmmError, Result};
use crate::tensor::Tensor;
use crate::traits::Discoverable;
use rand::{Rng, RngExt};

pub struct SymbolicRegression {
    pub max_depth: usize,
    pub population_size: usize,
    pub iterations: usize,
    pub variable_names: Vec<String>,
}

impl SymbolicRegression {
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
        let mut population: Vec<Expression> = (0..self.population_size)
            .map(|_| self.random_expr(&mut rng, 0))
            .collect();

        let mut best_expr = population[0].clone();
        let mut best_score = f64::INFINITY;

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
                if score < best_score {
                    best_score = score;
                    best_expr = population[i].clone();
                }
            }

            let mut new_pop = vec![best_expr.clone()];
            while new_pop.len() < self.population_size {
                let parent_a = Self::tournament_select(&population, &fitnesses, &mut rng, 3);
                let op: u8 = rng.random_range(0..3);
                let child = match op {
                    0 => {
                        let parent_b =
                            Self::tournament_select(&population, &fitnesses, &mut rng, 3);
                        self.crossover(parent_a, parent_b, &mut rng)
                    }
                    1 => self.mutate(parent_a, &mut rng),
                    _ => parent_a.clone(),
                };
                new_pop.push(child.simplify());
            }
            population = new_pop;
        }

        Ok(best_expr.simplify())
    }
}

impl Discoverable for SymbolicRegression {
    fn discover(data: &[Tensor], targets: &[f64]) -> Result<Expression> {
        if data.is_empty() {
            return Ok(Expression::Variable("x".into()));
        }
        let inputs: Vec<Vec<f64>> = data.iter().map(|t| t.data.clone()).collect();
        let sr = SymbolicRegression::new(3, 50);
        sr.fit(&inputs, targets)
    }
}
