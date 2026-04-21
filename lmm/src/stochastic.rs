// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Stochastic Text Enhancement
//!
//! This module provides [`StochasticEnhancer`] and [`SynonymBank`], which together
//! implement controllable word-level variation for generated text.
//!
//! ## Key Optimisation: Compile-time PHF Synonym Table
//!
//! The curated synonym table is stored as a `phf::Map` with `&'static [&'static str]`
//! values, compiled into the binary at build time. This replaces the previous
//! `build_curated_table()` function - a runtime `HashMap` that was rebuilt on every
//! call to `SynonymBank::new()` - with a **zero-allocation, O(1)** lookup.
//!
//! ## Stop-word Detection
//!
//! Stop words are stored as a compile-time [`phf::Set`], replacing the previous O(n)
//! linear scan over a static `&[&str]`.
//!
//! ## Enhancement Pipeline
//!
//! For each non-stop word in the input text, [`StochasticEnhancer`] samples a uniform
//! Bernoulli draw with probability `p` (default 0.5). On a "hit", the curated table
//! is consulted first; if no entry exists, a same-length word from the system wordlist
//! is substituted. Case is preserved.

use phf::{Map, Set, phf_map, phf_set};
use rand::{Rng, RngExt};
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;

/// Default synonym-substitution probability (50%).
const DEFAULT_REPLACEMENT_PROBABILITY: f64 = 0.5;

/// Minimum word length to include from the system dictionary.
#[cfg(not(target_arch = "wasm32"))]
const WORDLIST_MIN_WORD_LEN: usize = 5;

/// Maximum word length to include from the system dictionary.
#[cfg(not(target_arch = "wasm32"))]
const WORDLIST_MAX_WORD_LEN: usize = 14;

/// System dictionary file paths tried in order.
#[cfg(not(target_arch = "wasm32"))]
static SYSTEM_DICT_PATHS: &[&str] = &[
    "/usr/share/dict/american-english",
    "/usr/share/dict/english",
    "/usr/share/dict/words",
    "/usr/dict/words",
];

/// A compile-time perfect-hash set of English stop words.
///
/// Membership is tested in O(1) via a generated perfect hash - no heap allocation.
static STOP_WORDS: Set<&'static str> = phf_set! {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall", "its", "it",
    "this", "that", "these", "those", "not", "no", "nor", "so", "yet", "both", "also",
    "as", "if", "than", "then", "from", "into", "through", "during", "before", "after",
    "above", "below", "between", "each", "every", "any"
};

/// The curated synonym table compiled into the binary as a perfect hash map.
///
/// Each verb/noun maps to a `&'static [&'static str]` slice of semantically related
/// alternatives drawn from the LMM conceptual vocabulary.
///
/// Lookups are O(1) and require no heap allocation.
static CURATED_SYNONYMS: Map<&'static str, &'static [&'static str]> = phf_map! {
    "enables"       => &["allows", "permits", "empowers", "facilitates", "supports"],
    "represents"    => &["embodies", "symbolizes", "denotes", "signifies", "captures"],
    "describes"     => &["illustrates", "portrays", "articulates", "characterizes"],
    "manifests"     => &["reveals", "expresses", "demonstrates", "exhibits"],
    "connects"      => &["links", "unifies", "bridges", "binds", "joins"],
    "encodes"       => &["captures", "compresses", "maps", "stores", "embeds"],
    "defines"       => &["specifies", "determines", "establishes", "delineates"],
    "produces"      => &["generates", "yields", "creates", "constructs", "forms"],
    "transforms"    => &["converts", "shifts", "alters", "reconfigures", "reshapes"],
    "reveals"       => &["uncovers", "exposes", "discloses", "illuminates", "shows"],
    "governs"       => &["controls", "regulates", "directs", "commands", "shapes"],
    "expresses"     => &["articulates", "conveys", "communicates", "transmits"],
    "unveils"       => &["discloses", "reveals", "exposes", "uncovers", "presents"],
    "illuminates"   => &["clarifies", "enlightens", "elucidates", "reveals", "shows"],
    "shapes"        => &["molds", "forms", "structures", "defines", "guides"],
    "compresses"    => &["condenses", "distills", "reduces", "encapsulates"],
    "captures"      => &["encompasses", "embodies", "encapsulates", "reflects"],
    "generates"     => &["produces", "creates", "yields", "synthesizes", "builds"],
    "determines"    => &["establishes", "dictates", "defines", "resolves", "fixes"],
    "remains"       => &["persists", "endures", "abides", "continues", "stands"],
    "reflects"      => &["embodies", "mirrors", "represents", "signifies", "echoes"],
    "underlies"     => &["supports", "grounds", "anchors", "sustains", "informs"],
    "emerges"       => &["arises", "surfaces", "appears", "originates", "unfolds"],
    "constrains"    => &["limits", "bounds", "restricts", "confines", "regulates"],
    "encapsulates"  => &["contains", "embodies", "summarizes", "condenses"],
    "preserves"     => &["maintains", "sustains", "conserves", "upholds", "keeps"],
    "separates"     => &["divides", "partitions", "distinguishes", "isolates"],
    "truth"         => &["reality", "fact", "knowledge", "verity", "actuality"],
    "reality"       => &["existence", "actuality", "world", "nature", "truth"],
    "knowledge"     => &["understanding", "wisdom", "insight", "cognition", "learning"],
    "pattern"       => &["structure", "design", "configuration", "arrangement", "form"],
    "structure"     => &["framework", "organization", "architecture", "arrangement"],
    "symmetry"      => &["balance", "harmony", "proportion", "regularity", "order"],
    "entropy"       => &["disorder", "chaos", "uncertainty", "complexity", "randomness"],
    "energy"        => &["power", "force", "dynamics", "vitality", "potential"],
    "complexity"    => &["intricacy", "depth", "sophistication", "richness"],
    "harmony"       => &["balance", "coherence", "unity", "symmetry", "accord"],
    "balance"       => &["equilibrium", "harmony", "poise", "stability", "proportion"],
    "motion"        => &["movement", "dynamics", "flow", "trajectory", "progression"],
    "order"         => &["structure", "organization", "coherence", "arrangement"],
    "chaos"         => &["disorder", "turbulence", "randomness", "entropy", "flux"],
    "dimension"     => &["axis", "aspect", "realm", "domain", "magnitude"],
    "infinity"      => &["boundlessness", "endlessness", "vastness", "eternity"],
    "perception"    => &["awareness", "insight", "observation", "cognition", "sense"],
    "meaning"       => &["significance", "substance", "essence", "purpose", "value"],
    "existence"     => &["being", "reality", "presence", "life", "manifestation"],
    "universe"      => &["cosmos", "world", "reality", "existence", "totality"],
    "mathematics"   => &["algebra", "geometry", "calculus", "arithmetic", "analysis"],
    "equation"      => &["formula", "expression", "relationship", "model", "identity"],
    "frequency"     => &["resonance", "oscillation", "wavelength", "rhythm", "rate"],
    "resonance"     => &["harmony", "vibration", "coherence", "synchrony", "accord"],
    "evolution"     => &["transformation", "progression", "development", "dynamics"],
    "boundary"      => &["limit", "threshold", "barrier", "edge", "frontier"],
    "trajectory"    => &["path", "course", "orbit", "direction", "arc"],
    "causality"     => &["determinism", "consequence", "mechanism", "logic", "reason"],
    "logic"         => &["reasoning", "rationality", "inference", "deduction", "thought"],
    "force"         => &["energy", "power", "influence", "dynamics", "pressure"],
    "space"         => &["realm", "domain", "expanse", "field", "region"],
    "time"          => &["moment", "epoch", "duration", "continuity", "period"],
    "field"         => &["domain", "region", "space", "realm", "expanse"],
    "wave"          => &["oscillation", "vibration", "ripple", "undulation", "pulse"],
    "signal"        => &["indicator", "marker", "pattern", "trace", "message"],
    "computation"   => &["calculation", "processing", "evaluation", "analysis"],
    "simulation"    => &["modeling", "emulation", "representation", "approximation"],
    "prediction"    => &["forecast", "projection", "estimation", "inference"],
    "discovery"     => &["revelation", "finding", "insight", "breakthrough", "perception"],
    "foundation"    => &["basis", "grounding", "core", "bedrock", "principle"],
    "principle"     => &["law", "rule", "axiom", "tenet", "doctrine"],
    "intelligence"  => &["cognition", "awareness", "reasoning", "understanding"],
    "consciousness" => &["awareness", "perception", "cognition", "sentience"],
    "mathematical"  => &["symbolic", "geometric", "algebraic", "analytical", "formal"],
    "deterministic" => &["predictable", "systematic", "causal", "precise", "exact"],
    "probabilistic" => &["stochastic", "statistical", "uncertain", "random", "variable"],
    "infinite"      => &["boundless", "endless", "vast", "immeasurable", "limitless"],
    "fundamental"   => &["essential", "core", "primary", "foundational", "elemental"],
    "dynamic"       => &["evolving", "fluid", "active", "continuous", "adaptive"],
    "abstract"      => &["symbolic", "conceptual", "theoretical", "pure", "ideal"],
    "continuous"    => &["unbroken", "flowing", "perpetual", "sustained", "smooth"],
    "discrete"      => &["distinct", "separate", "finite", "quantized", "isolated"],
    "invariant"     => &["constant", "stable", "fixed", "unchanging", "conserved"],
    "coherent"      => &["unified", "consistent", "structured", "harmonious", "ordered"],
    "structural"    => &["architectural", "organizational", "systematic", "formal"],
    "axiomatic"     => &["foundational", "self-evident", "primary", "elemental"],
    "bounded"       => &["finite", "constrained", "limited", "contained", "restricted"],
    "symmetric"     => &["balanced", "regular", "uniform", "proportional", "equal"],
    "elegant"       => &["refined", "sophisticated", "beautiful", "graceful", "pure"],
    "precise"       => &["exact", "accurate", "rigorous", "meticulous", "definite"],
    "universal"     => &["general", "global", "absolute", "total", "pervasive"],
    "recursive"     => &["iterative", "self-referential", "repetitive", "cyclic"],
    "emergent"      => &["arising", "evolving", "developing", "unfolding", "appearing"],
    "complex"       => &["intricate", "sophisticated", "multifaceted", "rich", "deep"],
    "simple"        => &["elementary", "basic", "pure", "direct", "minimal"],
    "ancient"       => &["primordial", "prehistoric", "archaic", "classical", "timeless"],
    "sacred"        => &["divine", "revered", "hallowed", "eternal", "transcendent"],
    "cosmic"        => &["universal", "celestial", "infinite", "vast", "transcendent"],
    "hidden"        => &["concealed", "latent", "underlying", "subtle", "implicit"],
    "deeper"        => &["profound", "fundamental", "underlying", "essential", "core"],
    "new"           => &["novel", "emerging", "fresh", "modern", "innovative"],
    "pure"          => &["exact", "unadulterated", "precise", "fundamental", "essential"],
    "true"          => &["genuine", "authentic", "real", "valid", "accurate"],
    "great"         => &["profound", "vast", "significant", "remarkable", "extraordinary"],
    "vast"          => &["immense", "expansive", "boundless", "infinite", "enormous"],
    "known"         => &["established", "recognized", "understood", "observed", "verified"],
    "seen"          => &["observed", "perceived", "recognized", "witnessed", "noted"],
    "made"          => &["constructed", "formed", "created", "built", "composed"],
    "called"        => &["named", "termed", "labeled", "designated", "referred"],
};

/// Loads the system word list, grouped by word length, for same-length substitution.
///
/// Falls back to an empty map on WASM targets or when no dictionary file is found.
#[cfg(not(target_arch = "wasm32"))]
fn load_wordlist_by_length() -> HashMap<usize, Vec<String>> {
    let mut by_length: HashMap<usize, Vec<String>> = HashMap::new();
    for path in SYSTEM_DICT_PATHS {
        if let Ok(content) = fs::read_to_string(path) {
            for word in content.lines() {
                let w = word.trim().to_lowercase();
                if w.len() >= WORDLIST_MIN_WORD_LEN
                    && w.len() <= WORDLIST_MAX_WORD_LEN
                    && w.chars().all(|c| c.is_ascii_alphabetic())
                {
                    by_length.entry(w.len()).or_default().push(w);
                }
            }
            break;
        }
    }
    by_length
}

#[cfg(target_arch = "wasm32")]
fn load_wordlist_by_length() -> HashMap<usize, Vec<String>> {
    HashMap::new()
}

/// A two-tier synonym lookup bank.
///
/// - **Tier 1**: Compile-time `CURATED_SYNONYMS` PHF map - O(1), zero allocation.
/// - **Tier 2**: Runtime wordlist grouped by word length - O(1) bucket, O(k) sample.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::stochastic::SynonymBank;
///
/// let bank = SynonymBank::new();
/// assert!(bank.curated_count() > 0);
/// ```
pub struct SynonymBank {
    by_length: HashMap<usize, Vec<String>>,
}

impl SynonymBank {
    /// Creates a new [`SynonymBank`], loading the system wordlist at construction time.
    ///
    /// The curated synonyms come from a compile-time PHF map (no runtime build cost).
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::SynonymBank;
    /// let bank = SynonymBank::new();
    /// assert!(bank.curated_count() > 50);
    /// ```
    pub fn new() -> Self {
        Self {
            by_length: load_wordlist_by_length(),
        }
    }

    /// Samples a random synonym for `word`.
    ///
    /// Tier 1 (curated PHF map) is tried first; if the word is not in the curated table,
    /// Tier 2 (same-length wordlist) is used.
    ///
    /// # Arguments
    ///
    /// * `word` - The word to replace (case-insensitive lookup).
    /// * `rng` - A mutable random number generator.
    ///
    /// # Returns
    ///
    /// (`Option<String>`): A synonym, or `None` when no alternative is available.
    ///
    /// # Time Complexity
    ///
    /// O(1) amortised for the PHF lookup; O(k) for sampling from the wordlist bucket.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::SynonymBank;
    ///
    /// let bank = SynonymBank::new();
    /// let mut rng = rand::rng();
    /// let syn = bank.candidate("chaos", &mut rng);
    /// assert!(syn.is_some()); // "chaos" is in curated table
    /// ```
    pub fn candidate<R: Rng>(&self, word: &str, rng: &mut R) -> Option<String> {
        let lower = word.to_lowercase();

        if let Some(syns) = CURATED_SYNONYMS.get(lower.as_str())
            && !syns.is_empty()
        {
            let idx = rng.random_range(0..syns.len());
            return Some(syns[idx].to_string());
        }

        if let Some(bucket) = self.by_length.get(&lower.len()) {
            let candidates: Vec<&String> = bucket.iter().filter(|w| w.as_str() != lower).collect();
            if !candidates.is_empty() {
                let idx = rng.random_range(0..candidates.len());
                return Some(candidates[idx].clone());
            }
        }
        None
    }

    /// Returns the number of entries in the curated synonym table.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::SynonymBank;
    /// assert!(SynonymBank::new().curated_count() > 50);
    /// ```
    pub fn curated_count(&self) -> usize {
        CURATED_SYNONYMS.len()
    }

    /// Returns the total number of words in the loaded system wordlist.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::SynonymBank;
    /// let _ = SynonymBank::new().wordlist_len(); // may be 0 in CI
    /// ```
    pub fn wordlist_len(&self) -> usize {
        self.by_length.values().map(|v| v.len()).sum()
    }
}

impl Default for SynonymBank {
    fn default() -> Self {
        Self::new()
    }
}

/// A probabilistic word-substitution engine for stochastic text variation.
///
/// At each token, a Bernoulli trial with probability `p` determines whether to
/// substitute the word. Stop words (from the compile-time `STOP_WORDS` PHF set)
/// are always preserved. Case style (ALL_CAPS, Capitalised, lowercase) is mirror-copied
/// to the substituted word.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::stochastic::StochasticEnhancer;
///
/// let enhancer = StochasticEnhancer::with_default_probability();
/// let output = enhancer.enhance("chaos governs the universe");
/// assert!(!output.is_empty());
/// ```
pub struct StochasticEnhancer {
    bank: SynonymBank,
    probability: f64,
}

impl StochasticEnhancer {
    /// Creates a [`StochasticEnhancer`] with a custom substitution probability.
    ///
    /// `probability` is clamped to `[0.0, 1.0]`.
    ///
    /// # Arguments
    ///
    /// * `probability` - Per-word substitution probability.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::StochasticEnhancer;
    /// let e = StochasticEnhancer::new(0.3);
    /// assert_eq!(e.probability(), 0.3);
    /// ```
    pub fn new(probability: f64) -> Self {
        Self {
            bank: SynonymBank::new(),
            probability: probability.clamp(0.0, 1.0),
        }
    }

    /// Creates a [`StochasticEnhancer`] with `p = 0.5`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::StochasticEnhancer;
    /// assert_eq!(StochasticEnhancer::with_default_probability().probability(), 0.5);
    /// ```
    pub fn with_default_probability() -> Self {
        Self::new(DEFAULT_REPLACEMENT_PROBABILITY)
    }

    /// Returns the configured substitution probability.
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Enhances all lines of `text` by stochastic synonym substitution.
    ///
    /// Line breaks are preserved. Each line is processed independently.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text (may contain newlines).
    ///
    /// # Returns
    ///
    /// (`String`): Enhanced text with the same line structure.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::stochastic::StochasticEnhancer;
    ///
    /// let e = StochasticEnhancer::new(1.0); // always substitute when possible
    /// let out = e.enhance("chaos\nenergy");
    /// // Line structure preserved
    /// assert_eq!(out.lines().count(), 2);
    /// ```
    pub fn enhance(&self, text: &str) -> String {
        let mut rng = rand::rng();
        text.split('\n')
            .map(|line| self.enhance_line(line, &mut rng))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Enhances a single line by processing tokens left-to-right.
    fn enhance_line<R: Rng>(&self, line: &str, rng: &mut R) -> String {
        let mut result: Vec<String> = Vec::new();
        let mut is_first = true;

        for raw in line.split_whitespace() {
            let (prefix, core, suffix) = split_token(raw);
            let lower = core.to_lowercase();
            let is_stop = STOP_WORDS.contains(lower.as_str());
            let was_capitalized = core
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false);
            let was_all_caps = !core.is_empty() && core.chars().all(|c| c.is_uppercase());

            let replacement = if !is_stop && !core.is_empty() && rng.random_bool(self.probability) {
                match self.bank.candidate(&lower, rng) {
                    Some(syn) => {
                        if was_all_caps {
                            syn.to_uppercase()
                        } else if was_capitalized || is_first {
                            capitalize(&syn)
                        } else {
                            syn
                        }
                    }
                    None => core.to_string(),
                }
            } else {
                core.to_string()
            };

            result.push(format!("{}{}{}", prefix, replacement, suffix));
            is_first = false;
        }

        result.join(" ")
    }
}

/// Splits a raw token `"(word,"` into `("(", "word", ",")`.
///
/// Strips leading punctuation/symbols into `prefix` and trailing punctuation into `suffix`.
///
/// # Arguments
///
/// * `raw` - A whitespace-separated token.
///
/// # Returns
///
/// (`(&str, &str, &str)`): `(prefix, core, suffix)` where `core` is the alphabetic run.
fn split_token(raw: &str) -> (&str, &str, &str) {
    let prefix_end = raw.find(|c: char| c.is_alphabetic()).unwrap_or(raw.len());
    let content_start = prefix_end;
    if content_start >= raw.len() {
        return ("", raw, "");
    }
    let suffix_start = raw[content_start..]
        .rfind(|c: char| c.is_alphabetic())
        .map(|i| {
            content_start
                + i
                + raw[content_start + i..]
                    .chars()
                    .next()
                    .map(|ch| ch.len_utf8())
                    .unwrap_or(1)
        })
        .unwrap_or(raw.len());
    (
        &raw[..content_start],
        &raw[content_start..suffix_start],
        &raw[suffix_start..],
    )
}

/// Uppercases the first character and leaves the rest unchanged.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::stochastic::capitalize;
///
/// assert_eq!(capitalize("hello"), "Hello");
/// assert_eq!(capitalize(""), "");
/// ```
pub fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Returns `true` when `word` is a stop word according to the compile-time PHF set.
///
/// The lookup is O(1) with no heap allocation.
///
/// # Arguments
///
/// * `word` - A lowercase word (case-sensitive lookup against the PHF set).
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::stochastic::is_stop_word;
///
/// assert!(is_stop_word("the"));
/// assert!(!is_stop_word("chaos"));
/// ```
pub fn is_stop_word(word: &str) -> bool {
    STOP_WORDS.contains(word)
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
