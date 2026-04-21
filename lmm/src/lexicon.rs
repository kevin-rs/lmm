// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # System Dictionary Lexicon
//!
//! This module provides [`Lexicon`], a word index loaded from the operating system's
//! dictionary file (e.g. `/usr/share/dict/words`). It supports fuzzy candidate lookup
//! by word length and acoustic tone (mean byte value), enabling the stochastic text
//! generation pipeline to substitute syllabically similar words without an external
//! neural network or embedding model.

#[cfg(not(target_arch = "wasm32"))]
use crate::error::{LmmError, Result};
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

/// System dictionary search order. Tried in sequence; the first readable file wins.
#[cfg(not(target_arch = "wasm32"))]
static SYSTEM_DICT_PATHS: &[&str] = &[
    "/usr/share/dict/american-english",
    "/usr/share/dict/english",
    "/usr/share/dict/words",
    "/usr/dict/words",
];

/// An indexed word list that supports tone-based fuzzy candidate lookup.
///
/// Words are loaded once from a newline-separated dictionary file, filtered to
/// 3-15 character all-lowercase ASCII sequences, and indexed by length for fast
/// retrieval.
///
/// # Examples
///
/// ```no_run
/// use lmm::lexicon::Lexicon;
///
/// // Requires a system dictionary - skip in CI
/// let lex = Lexicon::load_system().unwrap();
/// assert!(lex.word_count() > 1000);
/// ```
pub struct Lexicon {
    words: Vec<String>,
    by_length: HashMap<usize, Vec<usize>>,
}

/// Computes the *acoustic tone* of a word as the mean of its byte values.
///
/// The tone is used as a proxy for phonetic similarity: words with close mean byte
/// values are assumed to sound similar when the source word is replaced.
///
/// Returns `110.0` (close to `'n'`) for empty strings.
///
/// # Arguments
///
/// * `word` - Any UTF-8 string (preferably ASCII).
///
/// # Returns
///
/// (`f64`): Mean byte value ∈ [0.0, 255.0].
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::lexicon::word_tone;
///
/// // All bytes equal 65 ('A') → tone = 65.0
/// assert_eq!(word_tone("AAA"), 65.0);
/// assert_eq!(word_tone(""), 110.0);
/// ```
pub fn word_tone(word: &str) -> f64 {
    if word.is_empty() {
        return 110.0;
    }
    word.bytes().map(|b| b as f64).sum::<f64>() / word.len() as f64
}

impl Lexicon {
    /// Loads a [`Lexicon`] from the first available system dictionary file.
    ///
    /// Tries `SYSTEM_DICT_PATHS` in order and returns the first successfully loaded file.
    ///
    /// # Returns
    ///
    /// (`Result<Lexicon>`): A populated lexicon, or an error if no dictionary is found.
    ///
    /// # Errors
    ///
    /// Returns [`LmmError::Perception`] when no system dictionary is readable.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use lmm::lexicon::Lexicon;
    /// let lex = Lexicon::load_system().unwrap();
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_system() -> Result<Self> {
        for path in SYSTEM_DICT_PATHS {
            if let Ok(lexicon) = Self::load_from(Path::new(path)) {
                return Ok(lexicon);
            }
        }
        Err(LmmError::Perception(
            "No system dictionary found; install a word list or pass --dictionary".into(),
        ))
    }

    /// Loads a [`Lexicon`] from a custom file path.
    ///
    /// The file must be a newline-separated list of words. Lines are filtered to
    /// 3-15 character all-lowercase ASCII strings. Words are stored sorted by an
    /// FNV-1a hash for deterministic ordering.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the dictionary file.
    ///
    /// # Returns
    ///
    /// (`Result<Lexicon>`): The loaded and indexed lexicon.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use lmm::lexicon::Lexicon;
    /// use std::path::Path;
    ///
    /// let lex = Lexicon::load_from(Path::new("/usr/share/dict/words")).unwrap();
    /// assert!(lex.word_count() > 0);
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path).map_err(|e| LmmError::Perception(e.to_string()))?;
        let mut words: Vec<String> = content
            .lines()
            .map(str::trim)
            .filter(|line| {
                line.len() >= 3
                    && line.len() <= 15
                    && line.chars().all(|c| c.is_ascii_alphabetic())
                    && line
                        .chars()
                        .next()
                        .map(|c| c.is_ascii_lowercase())
                        .unwrap_or(false)
            })
            .map(|w| w.to_string())
            .collect();
        words.sort_by_key(|w| {
            w.bytes().fold(0xcbf29ce484222325_u64, |acc, b| {
                acc.wrapping_mul(0x100000001b3).wrapping_add(b as u64)
            })
        });
        let mut by_length: HashMap<usize, Vec<usize>> = HashMap::new();
        for (index, word) in words.iter().enumerate() {
            by_length.entry(word.len()).or_default().push(index);
        }
        Ok(Self { words, by_length })
    }

    /// Returns candidate words near `(target_length, target_tone)`.
    ///
    /// Candidates are selected from the length band
    /// `[target_length - length_tolerance, target_length + length_tolerance]` (min 3)
    /// that also satisfy `|word_tone(word) - target_tone| ≤ tone_tolerance`.
    ///
    /// Results are sorted by ascending tone distance and capped at `limit`.
    ///
    /// # Arguments
    ///
    /// * `target_length` - Desired word length.
    /// * `target_tone` - Desired acoustic tone (mean byte value).
    /// * `length_tolerance` - ±length search radius.
    /// * `tone_tolerance` - Maximum tone deviation.
    /// * `limit` - Maximum number of results.
    ///
    /// # Returns
    ///
    /// (`Vec<&str>`): Up to `limit` candidate words, sorted by tone distance.
    ///
    /// # Time Complexity
    ///
    /// O(k · log k) where k is the number of words in the searched length band.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use lmm::lexicon::{Lexicon, word_tone};
    ///
    /// let lex = Lexicon::load_system().unwrap();
    /// let candidates = lex.candidates(5, word_tone("hello"), 1, 5.0, 10);
    /// assert!(!candidates.is_empty());
    /// ```
    pub fn candidates(
        &self,
        target_length: usize,
        target_tone: f64,
        length_tolerance: usize,
        tone_tolerance: f64,
        limit: usize,
    ) -> Vec<&str> {
        let min_len = target_length.saturating_sub(length_tolerance).max(3);
        let max_len = (target_length + length_tolerance).min(15);
        let mut scored: Vec<(&str, f64)> = (min_len..=max_len)
            .flat_map(|len| {
                self.by_length
                    .get(&len)
                    .map(Vec::as_slice)
                    .unwrap_or_default()
            })
            .filter_map(|&idx| {
                let word = self.words[idx].as_str();
                let diff = (word_tone(word) - target_tone).abs();
                if diff <= tone_tolerance {
                    Some((word, diff))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(limit).map(|(w, _)| w).collect()
    }

    /// Returns the total number of words in the lexicon.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use lmm::lexicon::Lexicon;
    /// let lex = Lexicon::load_system().unwrap();
    /// assert!(lex.word_count() > 0);
    /// ```
    pub fn word_count(&self) -> usize {
        self.words.len()
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
