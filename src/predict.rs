use crate::discovery::SymbolicRegression;
use crate::equation::Expression;
use crate::error::{LmmError, Result};
use std::collections::HashMap;

pub struct TextPredictor {
    pub window_size: usize,
    pub iterations: usize,
    pub depth: usize,
}

struct WordVocab {
    words: Vec<String>,
    word_to_id: HashMap<String, usize>,
}

impl WordVocab {
    fn build(text: &str) -> Self {
        let mut words = Vec::new();
        let mut word_to_id = HashMap::new();
        for token in text.split_whitespace() {
            if !word_to_id.contains_key(token) {
                let id = words.len();
                word_to_id.insert(token.to_string(), id);
                words.push(token.to_string());
            }
        }
        Self { words, word_to_id }
    }

    fn id_of(&self, word: &str) -> Option<usize> {
        self.word_to_id.get(word).copied()
    }
}

struct MarkovChain {
    bigram: HashMap<(usize, usize), HashMap<usize, f64>>,
    unigram: HashMap<usize, HashMap<usize, f64>>,
}

impl MarkovChain {
    fn build(tokens: &[String], vocab: &WordVocab) -> Self {
        let ids: Vec<usize> = tokens.iter().filter_map(|t| vocab.id_of(t)).collect();
        let mut uni_counts: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
        for pair in ids.windows(2) {
            *uni_counts
                .entry(pair[0])
                .or_default()
                .entry(pair[1])
                .or_insert(0) += 1;
        }
        let mut unigram: HashMap<usize, HashMap<usize, f64>> = HashMap::new();
        for (from, nexts) in &uni_counts {
            let total: usize = nexts.values().sum();
            if total > 0 {
                unigram.insert(
                    *from,
                    nexts
                        .iter()
                        .map(|(&to, &c)| (to, c as f64 / total as f64))
                        .collect(),
                );
            }
        }
        let mut bi_counts: HashMap<(usize, usize), HashMap<usize, usize>> = HashMap::new();
        for triple in ids.windows(3) {
            *bi_counts
                .entry((triple[0], triple[1]))
                .or_default()
                .entry(triple[2])
                .or_insert(0) += 1;
        }
        let mut bigram: HashMap<(usize, usize), HashMap<usize, f64>> = HashMap::new();
        for (key, nexts) in &bi_counts {
            let total: usize = nexts.values().sum();
            if total > 0 {
                bigram.insert(
                    *key,
                    nexts
                        .iter()
                        .map(|(&to, &c)| (to, c as f64 / total as f64))
                        .collect(),
                );
            }
        }
        Self { bigram, unigram }
    }

    fn prob(&self, prev2: Option<usize>, prev1: usize, next: usize) -> f64 {
        let bigram_prob = prev2.and_then(|p2| {
            self.bigram
                .get(&(p2, prev1))
                .and_then(|row| row.get(&next).copied())
        });
        if let Some(p) = bigram_prob {
            return p;
        }
        self.unigram
            .get(&prev1)
            .and_then(|row| row.get(&next))
            .copied()
            .unwrap_or(0.0)
    }

    fn score(&self, prev2: Option<usize>, prev1: usize, next: usize) -> f64 {
        let p = self.prob(prev2, prev1, next);
        if p > 0.0 { 1.0 - p } else { 1.0 }
    }
}

impl TextPredictor {
    pub fn new(window_size: usize, iterations: usize, depth: usize) -> Self {
        Self {
            window_size,
            iterations,
            depth,
        }
    }

    fn fit_trajectory(&self, positions: &[f64], word_ids: &[f64]) -> Result<Expression> {
        if positions.len() < 2 {
            return Err(LmmError::Discovery("Need at least 2 tokens".into()));
        }
        let inputs: Vec<Vec<f64>> = positions.iter().map(|&p| vec![p]).collect();
        SymbolicRegression::new(self.depth, self.iterations)
            .with_variables(vec!["x".into()])
            .with_population(60)
            .fit(&inputs, word_ids)
    }

    fn fit_rhythm(&self, positions: &[f64], lengths: &[f64]) -> Result<Expression> {
        if positions.len() < 2 {
            return Err(LmmError::Discovery("Need at least 2 tokens".into()));
        }
        let inputs: Vec<Vec<f64>> = positions.iter().map(|&p| vec![p]).collect();
        SymbolicRegression::new(self.depth.min(3), self.iterations / 2)
            .with_variables(vec!["x".into()])
            .with_population(40)
            .fit(&inputs, lengths)
    }

    fn suffix_match(
        window_tokens: &[String],
        recent: &[String],
        max_suffix: usize,
    ) -> Option<String> {
        let n = window_tokens.len();
        for len in (1..=max_suffix.min(recent.len())).rev() {
            let suffix = &recent[recent.len() - len..];
            for wpos in 0..n.saturating_sub(len) {
                if window_tokens[wpos..wpos + len] == *suffix {
                    let next_pos = wpos + len;
                    if next_pos < n {
                        return Some(window_tokens[next_pos].clone());
                    }
                }
            }
        }
        None
    }

    fn eval_score(eq: &Expression, pos: f64, target: f64, scale: f64) -> f64 {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), pos);
        let pred = eq.evaluate(&vars).unwrap_or(0.0);
        (pred - target).abs() / scale.max(1.0)
    }

    pub fn predict_continuation(
        &self,
        text: &str,
        predict_length: usize,
    ) -> Result<PredictedContinuation> {
        if text.is_empty() {
            return Err(LmmError::Perception("Input text is empty".into()));
        }

        let all_tokens: Vec<String> = text.split_whitespace().map(String::from).collect();
        if all_tokens.len() < 2 {
            return Err(LmmError::Perception("Need at least 2 words".into()));
        }

        let vocab = WordVocab::build(text);
        let window_start = all_tokens.len().saturating_sub(self.window_size);
        let window_tokens = &all_tokens[window_start..];
        let markov = MarkovChain::build(window_tokens, &vocab);

        let positions: Vec<f64> = (0..window_tokens.len()).map(|i| i as f64).collect();
        let word_ids: Vec<f64> = window_tokens
            .iter()
            .map(|t| vocab.id_of(t).unwrap_or(0) as f64)
            .collect();
        let lengths: Vec<f64> = window_tokens.iter().map(|t| t.len() as f64).collect();

        let trajectory_eq = self.fit_trajectory(&positions, &word_ids)?;
        let rhythm_eq = self.fit_rhythm(&positions, &lengths)?;

        let vocab_size = vocab.words.len();
        let traj_weight = if vocab_size <= 8 { 0.05 } else { 0.20 };
        let markov_weight = 1.0 - traj_weight - 0.10 - 0.15;

        let mut continuation = String::new();
        let mut generated: Vec<String> =
            window_tokens[window_tokens.len().saturating_sub(3)..].to_vec();
        let mut prev2_id: Option<usize> = if window_tokens.len() >= 2 {
            vocab.id_of(&window_tokens[window_tokens.len() - 2])
        } else {
            None
        };
        let mut prev1_id = vocab.id_of(window_tokens.last().unwrap()).unwrap_or(0);
        let mut pos = window_tokens.len() as f64;
        let mut recency_counts: HashMap<usize, usize> = HashMap::new();

        while continuation.len() < predict_length {
            let max_suffix = 2.min(generated.len());
            let chosen_word =
                if let Some(w) = Self::suffix_match(window_tokens, &generated, max_suffix) {
                    w
                } else {
                    let mut best_id = 0;
                    let mut best_score = f64::MAX;
                    for (id, word) in vocab.words.iter().enumerate() {
                        if continuation.len() + word.len() + 1 > predict_length + 6 {
                            continue;
                        }
                        let m_score = markov.score(prev2_id, prev1_id, id);
                        let t_score =
                            Self::eval_score(&trajectory_eq, pos, id as f64, vocab_size as f64);
                        let r_score = Self::eval_score(&rhythm_eq, pos, word.len() as f64, 20.0);
                        let recency = *recency_counts.get(&id).unwrap_or(&0) as f64;
                        let consec = if id == prev1_id { 2.0 } else { 0.0 };
                        let penalty = recency * 0.4 + consec;

                        let composite = markov_weight * m_score
                            + traj_weight * t_score
                            + 0.10 * r_score
                            + 0.15 * penalty;

                        if composite < best_score {
                            best_score = composite;
                            best_id = id;
                        }
                    }
                    vocab.words[best_id].clone()
                };

            let chosen_id = vocab.id_of(&chosen_word).unwrap_or(0);
            continuation.push(' ');
            continuation.push_str(&chosen_word);
            *recency_counts.entry(chosen_id).or_insert(0) += 1;
            generated.push(chosen_word);
            if generated.len() > 4 {
                generated.remove(0);
            }
            prev2_id = Some(prev1_id);
            prev1_id = chosen_id;
            pos += 1.0;

            if recency_counts.len() == vocab_size && recency_counts.values().all(|&c| c >= 2) {
                break;
            }
        }

        Ok(PredictedContinuation {
            trajectory_equation: trajectory_eq,
            rhythm_equation: rhythm_eq,
            window_used: window_tokens.len(),
            continuation,
        })
    }
}

pub struct PredictedContinuation {
    pub trajectory_equation: Expression,
    pub rhythm_equation: Expression,
    pub window_used: usize,
    pub continuation: String,
}
