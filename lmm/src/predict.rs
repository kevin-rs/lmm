// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Symbolic Text Prediction
//!
//! This module implements [`TextPredictor`], a symbolic continuation engine that extends a
//! text prompt by fitting tone and rhythm trajectories then selecting words token-by-token.
//!
//! ## Algorithm
//!
//! 1. **Tokenise** - split input into a sliding window of tokens.
//! 2. **Tone trajectory** - fit a symbolic expression to `(token_position → mean_byte_value)`.
//! 3. **Rhythm trajectory** - fit a second expression to `(token_position → word_length)`.
//! 4. **Predict** - for each step, determine the expected POS; select the word from a curated
//!    pool or lexicon that best matches the predicted tone and length.
//!
//! ## Optimisation: Compile-time PHF Sets
//!
//! The four lexical pools (nouns, adjectives, verbs, adverbs) used for POS detection are
//! stored as compile-time [`phf::Set`]s, replacing the previous O(n) `slice.contains(&w)`
//! linear scans in `detect_pos` with **O(1) perfect-hash lookups**.

use crate::discovery::SymbolicRegression;
use crate::equation::Expression;
use crate::error::{LmmError, Result};
use crate::lexicon::{Lexicon, word_tone};
use phf::{Set, phf_set};
use std::collections::HashMap;

/// Compile-time perfect-hash set of common function-word articles.
static ARTICLES: Set<&'static str> = phf_set! { "the", "a", "an" };

/// Compile-time perfect-hash set of common prepositions.
static PREPOSITIONS: Set<&'static str> = phf_set! {
    "in", "of", "through", "beyond", "within", "across", "beneath", "among", "into", "from",
    "toward", "over", "between", "after", "before", "along", "behind", "around", "upon",
    "to", "at", "by", "on", "under", "beside", "above", "below"
};

/// Compile-time perfect-hash set of coordinating/subordinating conjunctions.
static CONJUNCTIONS: Set<&'static str> = phf_set! {
    "and", "yet", "or", "while", "because", "where", "that", "which", "though", "whereas",
    "since", "as", "but", "nor", "so", "although", "if", "unless", "until", "when", "who"
};

/// Compile-time perfect-hash set of copular/auxiliary verbs.
static COPULAS: Set<&'static str> = phf_set! {
    "is", "was", "are", "were", "have", "had", "would", "could", "may", "must", "shall",
    "should", "can", "be", "been", "has", "do", "does", "did", "will", "might"
};

/// Compile-time perfect-hash set of common nouns used in generation pools.
static COMMON_NOUNS: Set<&'static str> = phf_set! {
    "world", "life", "time", "mind", "heart", "soul", "truth", "light", "force", "power",
    "knowledge", "form", "space", "nature", "order", "energy", "field", "depth", "path",
    "wave", "core", "edge", "flow", "source", "point", "thought", "reason", "vision",
    "sense", "voice", "code", "pattern", "motion", "change", "state", "law", "ground",
    "center", "realm", "layer", "base", "frame", "structure", "system", "process", "signal",
    "curve", "node", "origin", "principle", "concept", "theory", "equation", "formula",
    "rhythm", "harmony", "logic", "relation", "measure", "value", "symbol", "meaning",
    "wisdom", "insight", "clarity", "beauty", "shape", "dimension", "symmetry", "balance",
    "proportion", "ratio", "scale", "scope", "proof", "creation", "design", "purpose",
    "legacy", "cosmos", "matter", "element", "charge", "pulse", "cycle", "basis", "language",
    "message", "craft", "substance", "presence", "silence", "movement", "boundary", "horizon",
    "current", "gravity", "tension", "density", "volume", "surface", "fabric", "memory",
    "future", "history", "moment", "age", "epoch", "dawn"
};

/// Compile-time perfect-hash set of common adjectives.
static COMMON_ADJECTIVES: Set<&'static str> = phf_set! {
    "ancient", "deep", "vast", "pure", "clear", "bright", "true", "real", "great", "strong",
    "long", "wide", "high", "free", "open", "dark", "light", "still", "calm", "bold", "new",
    "old", "known", "hidden", "sacred", "cosmic", "eternal", "infinite", "divine", "natural",
    "logical", "formal", "primal", "human", "living", "moving", "rising", "central", "vital",
    "basic", "complex", "simple", "ordered", "precise", "exact", "linear", "dynamic", "static",
    "global", "total", "subtle", "dense", "outer", "inner", "silent", "finite", "higher",
    "noble", "fluid", "solid", "curved", "broken", "woven", "resonant", "latent"
};

/// Compile-time perfect-hash set of common verbs.
static COMMON_VERBS: Set<&'static str> = phf_set! {
    "reveal", "encode", "create", "form", "hold", "reach", "grow", "rise", "flows", "moves",
    "builds", "shapes", "contains", "reflects", "extends", "emerges", "unfolds", "expands",
    "compresses", "expresses", "represents", "models", "captures", "defines", "describes",
    "governs", "enables", "transforms", "generates", "solves", "proves", "measures", "encodes",
    "connects", "binds", "weaves", "follows", "guides", "aligns"
};

/// Compile-time perfect-hash set of common adverbs.
static COMMON_ADVERBS: Set<&'static str> = phf_set! {
    "deeply", "clearly", "truly", "freely", "greatly", "widely", "highly", "fully", "still",
    "always", "often", "never", "only", "merely", "precisely", "exactly", "naturally",
    "silently", "eternally"
};

/// Ordered slice of articles (for `pick_function_word`'s min-recency selection).
static ARTICLES_SLICE: &[&str] = &["the", "a", "an"];
static PREPOSITIONS_SLICE: &[&str] = &[
    "in", "of", "through", "beyond", "within", "across", "beneath", "among", "into", "from",
    "toward", "over", "between", "after", "before", "along", "behind", "around", "upon",
];
static CONJUNCTIONS_SLICE: &[&str] = &[
    "and", "yet", "or", "while", "because", "where", "that", "which", "though", "whereas", "since",
    "as",
];
static COPULAS_SLICE: &[&str] = &[
    "is", "was", "are", "were", "have", "had", "would", "could", "may", "must", "shall", "should",
    "can",
];
static COMMON_NOUNS_SLICE: &[&str] = &[
    "world",
    "life",
    "time",
    "mind",
    "heart",
    "soul",
    "truth",
    "light",
    "force",
    "power",
    "knowledge",
    "form",
    "space",
    "nature",
    "order",
    "energy",
    "field",
    "depth",
    "path",
    "wave",
    "core",
    "edge",
    "flow",
    "source",
    "point",
    "thought",
    "reason",
    "vision",
    "sense",
    "voice",
    "code",
    "pattern",
    "motion",
    "change",
    "state",
    "law",
    "ground",
    "center",
    "realm",
    "layer",
    "base",
    "frame",
    "structure",
    "system",
    "process",
    "signal",
    "curve",
    "node",
    "origin",
    "principle",
    "concept",
    "theory",
    "equation",
    "formula",
    "rhythm",
    "harmony",
    "logic",
    "relation",
    "measure",
    "value",
    "symbol",
    "meaning",
    "wisdom",
    "insight",
    "clarity",
    "beauty",
    "shape",
    "dimension",
    "symmetry",
    "balance",
    "proportion",
    "ratio",
    "scale",
    "scope",
    "proof",
    "creation",
    "design",
    "purpose",
    "legacy",
    "cosmos",
    "matter",
    "element",
    "charge",
    "pulse",
    "cycle",
    "basis",
    "language",
    "message",
    "craft",
    "substance",
    "presence",
    "silence",
    "movement",
    "boundary",
    "horizon",
    "current",
    "gravity",
    "tension",
    "density",
    "volume",
    "surface",
    "fabric",
    "memory",
    "future",
    "history",
    "moment",
    "age",
    "epoch",
    "dawn",
];
static COMMON_ADJECTIVES_SLICE: &[&str] = &[
    "ancient", "deep", "vast", "pure", "clear", "bright", "true", "real", "great", "strong",
    "long", "wide", "high", "free", "open", "dark", "light", "still", "calm", "bold", "new", "old",
    "known", "hidden", "sacred", "cosmic", "eternal", "infinite", "divine", "natural", "logical",
    "formal", "primal", "human", "living", "moving", "rising", "central", "vital", "basic",
    "complex", "simple", "ordered", "precise", "exact", "linear", "dynamic", "static", "global",
    "total", "subtle", "dense", "outer", "inner", "silent", "finite", "higher", "noble", "fluid",
    "solid", "curved", "broken", "woven", "resonant", "latent",
];
static COMMON_VERBS_SLICE: &[&str] = &[
    "reveal",
    "encode",
    "create",
    "form",
    "hold",
    "reach",
    "grow",
    "rise",
    "flows",
    "moves",
    "builds",
    "shapes",
    "contains",
    "reflects",
    "extends",
    "emerges",
    "unfolds",
    "expands",
    "compresses",
    "expresses",
    "represents",
    "models",
    "captures",
    "defines",
    "describes",
    "governs",
    "enables",
    "transforms",
    "generates",
    "solves",
    "proves",
    "measures",
    "encodes",
    "connects",
    "binds",
    "weaves",
    "follows",
    "guides",
    "aligns",
];
static COMMON_ADVERBS_SLICE: &[&str] = &[
    "deeply",
    "clearly",
    "truly",
    "freely",
    "greatly",
    "widely",
    "highly",
    "fully",
    "still",
    "always",
    "often",
    "never",
    "only",
    "merely",
    "precisely",
    "exactly",
    "naturally",
    "silently",
    "eternally",
];

/// Coarse part-of-speech categories.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum CoarsePos {
    Article,
    Preposition,
    Conjunction,
    Verb,
    Noun,
    Adjective,
    Adverb,
    Unknown,
}

/// Tags `word` with a coarse POS category using PHF set lookups (O(1) each).
fn detect_pos(word: &str) -> CoarsePos {
    let w = word.to_ascii_lowercase();
    let w = w.as_str();
    if ARTICLES.contains(w) {
        return CoarsePos::Article;
    }
    if PREPOSITIONS.contains(w) {
        return CoarsePos::Preposition;
    }
    if CONJUNCTIONS.contains(w) {
        return CoarsePos::Conjunction;
    }
    if COPULAS.contains(w) {
        return CoarsePos::Verb;
    }
    if COMMON_NOUNS.contains(w) {
        return CoarsePos::Noun;
    }
    if COMMON_ADJECTIVES.contains(w) {
        return CoarsePos::Adjective;
    }
    if COMMON_VERBS.contains(w) {
        return CoarsePos::Verb;
    }
    if COMMON_ADVERBS.contains(w) || w.ends_with("ly") {
        return CoarsePos::Adverb;
    }
    if w.ends_with("tion")
        || w.ends_with("ness")
        || w.ends_with("ity")
        || w.ends_with("ment")
        || w.ends_with("ics")
        || w.ends_with("ism")
        || w.ends_with("ence")
        || w.ends_with("ance")
        || w.ends_with("dom")
        || w.ends_with("hood")
    {
        return CoarsePos::Noun;
    }
    if w.ends_with("ful")
        || w.ends_with("less")
        || w.ends_with("ical")
        || w.ends_with("ous")
        || w.ends_with("ive")
        || w.ends_with("ible")
        || w.ends_with("able")
        || w.ends_with("ic")
        || w.ends_with("al")
    {
        return CoarsePos::Adjective;
    }
    if w.ends_with("ing")
        || w.ends_with("ize")
        || w.ends_with("ise")
        || w.ends_with("es")
        || w.ends_with("ed")
    {
        return CoarsePos::Verb;
    }
    CoarsePos::Unknown
}

/// Returns the most probable next POS given the recent history and current step counter.
fn expected_next_pos(history: &[CoarsePos], step: usize) -> CoarsePos {
    match history.last().copied() {
        Some(CoarsePos::Article) => CoarsePos::Adjective,
        Some(CoarsePos::Adjective) => CoarsePos::Noun,
        Some(CoarsePos::Noun) => match step % 5 {
            0 => CoarsePos::Verb,
            1 => CoarsePos::Preposition,
            2 => CoarsePos::Conjunction,
            3 => CoarsePos::Verb,
            _ => CoarsePos::Adverb,
        },
        Some(CoarsePos::Verb) => match step % 2 {
            0 => CoarsePos::Article,
            _ => CoarsePos::Preposition,
        },
        Some(CoarsePos::Preposition) => CoarsePos::Article,
        Some(CoarsePos::Conjunction) => CoarsePos::Article,
        Some(CoarsePos::Adverb) => CoarsePos::Adjective,
        Some(CoarsePos::Unknown) => CoarsePos::Preposition,
        None => CoarsePos::Article,
    }
}

fn is_function_pos(pos: CoarsePos) -> bool {
    matches!(
        pos,
        CoarsePos::Article | CoarsePos::Preposition | CoarsePos::Conjunction | CoarsePos::Verb
    )
}

fn pick_function_word(pos: CoarsePos, recency: &HashMap<String, usize>) -> &'static str {
    let candidates: &[&str] = match pos {
        CoarsePos::Article => ARTICLES_SLICE,
        CoarsePos::Preposition => PREPOSITIONS_SLICE,
        CoarsePos::Conjunction => CONJUNCTIONS_SLICE,
        CoarsePos::Verb => COPULAS_SLICE,
        _ => ARTICLES_SLICE,
    };
    candidates
        .iter()
        .min_by_key(|&&w| recency.get(w).copied().unwrap_or(0))
        .copied()
        .unwrap_or("the")
}

fn suffix_match(window: &[String], recent: &[String], max_suffix: usize) -> Option<String> {
    for len in (2..=max_suffix.min(recent.len())).rev() {
        let suffix = &recent[recent.len() - len..];
        for pos in 0..window.len().saturating_sub(len) {
            if window[pos..pos + len] == *suffix && pos + len < window.len() {
                return Some(window[pos + len].clone());
            }
        }
    }
    None
}

fn evaluate_at(eq: &Expression, pos: f64) -> f64 {
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), pos);
    eq.evaluate(&vars).unwrap_or(107.0)
}

fn stable_target_tone(eq: &Expression, pos: f64, context_tones: &[f64]) -> f64 {
    let mean = context_tones.iter().sum::<f64>() / context_tones.len().max(1) as f64;
    let gp = evaluate_at(eq, pos);
    (if !gp.is_finite() || (gp - mean).abs() > 6.0 {
        mean
    } else {
        gp
    })
    .clamp(97.0, 122.0)
}

fn stable_target_length(eq: &Expression, pos: f64) -> usize {
    let raw = evaluate_at(eq, pos).round();
    if raw.is_finite() && (3.0..=10.0).contains(&raw) {
        raw as usize
    } else {
        5
    }
}

fn score_word(
    word: &str,
    target_tone: f64,
    target_length: usize,
    recency: &HashMap<String, usize>,
) -> f64 {
    let r = recency.get(word).copied().unwrap_or(0) as f64 * 6.0;
    let tone_diff = (word_tone(word) - target_tone).abs();
    let len_diff = (word.len() as f64 - target_length as f64).abs() * 0.8;
    r + tone_diff + len_diff
}

fn best_from_pool<'a>(
    pool: &[&'a str],
    target_tone: f64,
    target_length: usize,
    recency: &HashMap<String, usize>,
) -> Option<&'a str> {
    pool.iter().copied().min_by(|&a, &b| {
        let sa = score_word(a, target_tone, target_length, recency);
        let sb = score_word(b, target_tone, target_length, recency);
        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// A symbolic text continuation engine.
///
/// `TextPredictor` uses symbolic regression to fit a tone trajectory and a rhythm
/// trajectory over the input window, then generates tokens token-by-token, selecting
/// words from compile-time lexical pools that best match the predicted tone and length.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::predict::TextPredictor;
///
/// let predictor = TextPredictor::new(20, 30, 3);
/// let result = predictor.predict_continuation("the universe reveals its form", 40);
/// assert!(result.is_ok());
/// let cont = result.unwrap();
/// assert!(!cont.continuation.trim().is_empty());
/// ```
pub struct TextPredictor {
    /// Number of input tokens used as context.
    pub window_size: usize,
    /// Symbolic regression iterations.
    pub iterations: usize,
    /// Maximum expression tree depth.
    pub depth: usize,
    /// Optional system dictionary for extended vocabulary.
    pub lexicon: Option<Lexicon>,
}

impl TextPredictor {
    /// Creates a new [`TextPredictor`] without a lexicon.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of context tokens.
    /// * `iterations` - Symbolic regression iterations.
    /// * `depth` - Maximum expression depth.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::predict::TextPredictor;
    /// let p = TextPredictor::new(10, 20, 3);
    /// assert_eq!(p.window_size, 10);
    /// ```
    pub fn new(window_size: usize, iterations: usize, depth: usize) -> Self {
        Self {
            window_size,
            iterations,
            depth,
            lexicon: None,
        }
    }

    /// Attaches a [`Lexicon`] for extended vocabulary during content-word selection.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use lmm::predict::TextPredictor;
    /// use lmm::lexicon::Lexicon;
    ///
    /// let lex = Lexicon::load_system().unwrap();
    /// let p = TextPredictor::new(20, 30, 3).with_lexicon(lex);
    /// assert!(p.lexicon.is_some());
    /// ```
    pub fn with_lexicon(mut self, lexicon: Lexicon) -> Self {
        self.lexicon = Some(lexicon);
        self
    }

    fn fit_trajectory(&self, positions: &[f64], tones: &[f64]) -> Result<Expression> {
        if positions.len() < 2 {
            return Err(LmmError::Discovery("Need at least 2 tokens".into()));
        }
        let inputs: Vec<Vec<f64>> = positions.iter().map(|&p| vec![p]).collect();
        SymbolicRegression::new(self.depth, self.iterations)
            .with_variables(vec!["x".into()])
            .with_population(60)
            .fit(&inputs, tones)
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

    fn curated_pool_for_pos(next_pos: CoarsePos) -> &'static [&'static str] {
        match next_pos {
            CoarsePos::Noun => COMMON_NOUNS_SLICE,
            CoarsePos::Adjective => COMMON_ADJECTIVES_SLICE,
            CoarsePos::Verb => COMMON_VERBS_SLICE,
            CoarsePos::Adverb => COMMON_ADVERBS_SLICE,
            _ => COMMON_NOUNS_SLICE,
        }
    }

    fn select_content_word(
        &self,
        next_pos: CoarsePos,
        trajectory_eq: &Expression,
        rhythm_eq: &Expression,
        pos: f64,
        recency: &HashMap<String, usize>,
        context_tones: &[f64],
    ) -> String {
        let target_tone = stable_target_tone(trajectory_eq, pos, context_tones);
        let target_length = stable_target_length(rhythm_eq, pos);
        let curated = Self::curated_pool_for_pos(next_pos);
        if let Some(w) = best_from_pool(curated, target_tone, target_length, recency) {
            return w.to_string();
        }
        if let Some(lexicon) = &self.lexicon {
            let candidates = lexicon.candidates(target_length, target_tone, 2, 8.0, 30);
            if let Some(w) = best_from_pool(&candidates, target_tone, target_length, recency) {
                return w.to_string();
            }
        }
        "world".to_string()
    }

    fn select_from_context(
        &self,
        window_tokens: &[String],
        trajectory_eq: &Expression,
        rhythm_eq: &Expression,
        pos: f64,
        recency: &HashMap<String, usize>,
        context_tones: &[f64],
    ) -> String {
        let target_tone = stable_target_tone(trajectory_eq, pos, context_tones);
        let target_len = stable_target_length(rhythm_eq, pos);
        let refs: Vec<&str> = window_tokens.iter().map(String::as_str).collect();
        best_from_pool(&refs, target_tone, target_len, recency)
            .unwrap_or("world")
            .to_string()
    }

    /// Predicts a continuation of `text` up to `predict_length` characters.
    ///
    /// # Arguments
    ///
    /// * `text` - Input prompt (at least 2 whitespace-delimited words).
    /// * `predict_length` - Target continuation character count.
    ///
    /// # Returns
    ///
    /// (`Result<PredictedContinuation>`): The generated continuation, fitted equations,
    /// and metadata.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::Perception`] for empty/too-short input, or
    /// [`LmmError::Discovery`] if symbolic regression fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::predict::TextPredictor;
    ///
    /// let p = TextPredictor::new(20, 30, 3);
    /// let c = p.predict_continuation("mathematics reveals the truth of existence", 50).unwrap();
    /// assert!(!c.continuation.trim().is_empty());
    /// ```
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
        let window_start = all_tokens.len().saturating_sub(self.window_size);
        let window_tokens = &all_tokens[window_start..];

        let positions: Vec<f64> = (0..window_tokens.len()).map(|i| i as f64).collect();
        let tones: Vec<f64> = window_tokens
            .iter()
            .map(|t| word_tone(&t.to_ascii_lowercase()))
            .collect();
        let lengths: Vec<f64> = window_tokens.iter().map(|t| t.len() as f64).collect();

        let trajectory_eq = self.fit_trajectory(&positions, &tones)?;
        let rhythm_eq = self.fit_rhythm(&positions, &lengths)?;

        let mut pos_history: Vec<CoarsePos> = window_tokens.iter().map(|t| detect_pos(t)).collect();
        let mut suffix_context: Vec<String> =
            window_tokens[window_tokens.len().saturating_sub(3)..].to_vec();
        let mut continuation = String::new();
        let mut pos = window_tokens.len() as f64;
        let mut recency: HashMap<String, usize> = HashMap::new();
        let mut step: usize = 0;

        while continuation.len() < predict_length {
            let suffix_result =
                suffix_match(window_tokens, &suffix_context, 2.min(suffix_context.len()));
            let chosen = match suffix_result {
                Some(w) => w,
                None => {
                    let next_pos = expected_next_pos(&pos_history, step);
                    if is_function_pos(next_pos) {
                        pick_function_word(next_pos, &recency).to_string()
                    } else if self.lexicon.is_some() {
                        self.select_content_word(
                            next_pos,
                            &trajectory_eq,
                            &rhythm_eq,
                            pos,
                            &recency,
                            &tones,
                        )
                    } else {
                        self.select_from_context(
                            window_tokens,
                            &trajectory_eq,
                            &rhythm_eq,
                            pos,
                            &recency,
                            &tones,
                        )
                    }
                }
            };

            continuation.push(' ');
            continuation.push_str(&chosen);
            *recency.entry(chosen.clone()).or_insert(0) += 1;
            pos_history.push(detect_pos(&chosen));
            if pos_history.len() > 8 {
                pos_history.remove(0);
            }
            suffix_context.push(chosen);
            if suffix_context.len() > 3 {
                suffix_context.remove(0);
            }
            pos += 1.0;
            step += 1;
        }

        Ok(PredictedContinuation {
            trajectory_equation: trajectory_eq,
            rhythm_equation: rhythm_eq,
            window_used: window_tokens.len(),
            continuation,
        })
    }
}

/// The output of a text prediction pass.
///
/// # Fields
///
/// - `trajectory_equation` - symbolic expression mapping token position → tone.
/// - `rhythm_equation` - symbolic expression mapping token position → word length.
/// - `window_used` - number of input tokens used as context.
/// - `continuation` - the generated text continuation (prefixed with a space).
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::predict::TextPredictor;
///
/// let p = TextPredictor::new(10, 20, 2);
/// let c = p.predict_continuation("mathematics reveals the truth", 30).unwrap();
/// println!("Equation: {}", c.trajectory_equation);
/// println!("Window: {}", c.window_used);
/// ```
pub struct PredictedContinuation {
    /// Symbolic expression fitting tone over token positions.
    pub trajectory_equation: Expression,
    /// Symbolic expression fitting word length over token positions.
    pub rhythm_equation: Expression,
    /// Number of input tokens used as context.
    pub window_used: usize,
    /// The generated continuation text.
    pub continuation: String,
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
