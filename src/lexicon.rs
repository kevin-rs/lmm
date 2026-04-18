#[cfg(not(target_arch = "wasm32"))]
use crate::error::{LmmError, Result};
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
static SYSTEM_DICT_PATHS: &[&str] = &[
    "/usr/share/dict/american-english",
    "/usr/share/dict/english",
    "/usr/share/dict/words",
    "/usr/dict/words",
];

pub struct Lexicon {
    words: Vec<String>,
    by_length: HashMap<usize, Vec<usize>>,
}

pub fn word_tone(word: &str) -> f64 {
    if word.is_empty() {
        return 110.0;
    }
    word.bytes().map(|b| b as f64).sum::<f64>() / word.len() as f64
}

impl Lexicon {
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

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path).map_err(|e| LmmError::Perception(e.to_string()))?;
        let mut words: Vec<String> = content
            .lines()
            .map(str::trim)
            .filter(|line| {
                line.len() >= 3
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

    pub fn word_count(&self) -> usize {
        self.words.len()
    }
}
