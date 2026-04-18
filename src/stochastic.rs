use rand::{Rng, RngExt};
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;

#[cfg(not(target_arch = "wasm32"))]
static SYSTEM_DICT_PATHS: &[&str] = &[
    "/usr/share/dict/american-english",
    "/usr/share/dict/english",
    "/usr/share/dict/words",
    "/usr/dict/words",
];
const DEFAULT_REPLACEMENT_PROBABILITY: f64 = 0.5;
#[cfg(not(target_arch = "wasm32"))]
const WORDLIST_MIN_WORD_LEN: usize = 5;
#[cfg(not(target_arch = "wasm32"))]
const WORDLIST_MAX_WORD_LEN: usize = 14;

const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is",
    "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "its", "it", "this", "that", "these", "those",
    "not", "no", "nor", "so", "yet", "both", "also", "as", "if", "than", "then", "from", "into",
    "through", "during", "before", "after", "above", "below", "between", "each", "every", "any",
];

fn build_curated_table() -> HashMap<String, Vec<String>> {
    let mut m: HashMap<String, Vec<String>> = HashMap::new();

    let entries: &[(&str, &[&str])] = &[
        (
            "enables",
            &["allows", "permits", "empowers", "facilitates", "supports"],
        ),
        (
            "represents",
            &["embodies", "symbolizes", "denotes", "signifies", "captures"],
        ),
        (
            "describes",
            &["illustrates", "portrays", "articulates", "characterizes"],
        ),
        (
            "manifests",
            &["reveals", "expresses", "demonstrates", "exhibits"],
        ),
        (
            "connects",
            &["links", "unifies", "bridges", "binds", "joins"],
        ),
        (
            "encodes",
            &["captures", "compresses", "maps", "stores", "embeds"],
        ),
        (
            "defines",
            &["specifies", "determines", "establishes", "delineates"],
        ),
        (
            "produces",
            &["generates", "yields", "creates", "constructs", "forms"],
        ),
        (
            "transforms",
            &["converts", "shifts", "alters", "reconfigures", "reshapes"],
        ),
        (
            "reveals",
            &["uncovers", "exposes", "discloses", "illuminates", "shows"],
        ),
        (
            "governs",
            &["controls", "regulates", "directs", "commands", "shapes"],
        ),
        (
            "expresses",
            &["articulates", "conveys", "communicates", "transmits"],
        ),
        (
            "unveils",
            &["discloses", "reveals", "exposes", "uncovers", "presents"],
        ),
        (
            "illuminates",
            &["clarifies", "enlightens", "elucidates", "reveals", "shows"],
        ),
        (
            "shapes",
            &["molds", "forms", "structures", "defines", "guides"],
        ),
        (
            "compresses",
            &["condenses", "distills", "reduces", "encapsulates"],
        ),
        (
            "captures",
            &["encompasses", "embodies", "encapsulates", "reflects"],
        ),
        (
            "generates",
            &["produces", "creates", "yields", "synthesizes", "builds"],
        ),
        (
            "determines",
            &["establishes", "dictates", "defines", "resolves", "fixes"],
        ),
        (
            "remains",
            &["persists", "endures", "abides", "continues", "stands"],
        ),
        (
            "reflects",
            &["embodies", "mirrors", "represents", "signifies", "echoes"],
        ),
        (
            "underlies",
            &["supports", "grounds", "anchors", "sustains", "informs"],
        ),
        (
            "emerges",
            &["arises", "surfaces", "appears", "originates", "unfolds"],
        ),
        (
            "constrains",
            &["limits", "bounds", "restricts", "confines", "regulates"],
        ),
        (
            "encapsulates",
            &["contains", "embodies", "summarizes", "condenses"],
        ),
        (
            "preserves",
            &["maintains", "sustains", "conserves", "upholds", "keeps"],
        ),
        (
            "separates",
            &["divides", "partitions", "distinguishes", "isolates"],
        ),
        (
            "truth",
            &["reality", "fact", "knowledge", "verity", "actuality"],
        ),
        (
            "reality",
            &["existence", "actuality", "world", "nature", "truth"],
        ),
        (
            "knowledge",
            &[
                "understanding",
                "wisdom",
                "insight",
                "cognition",
                "learning",
            ],
        ),
        (
            "pattern",
            &[
                "structure",
                "design",
                "configuration",
                "arrangement",
                "form",
            ],
        ),
        (
            "structure",
            &["framework", "organization", "architecture", "arrangement"],
        ),
        (
            "symmetry",
            &["balance", "harmony", "proportion", "regularity", "order"],
        ),
        (
            "entropy",
            &[
                "disorder",
                "chaos",
                "uncertainty",
                "complexity",
                "randomness",
            ],
        ),
        (
            "energy",
            &["power", "force", "dynamics", "vitality", "potential"],
        ),
        (
            "complexity",
            &["intricacy", "depth", "sophistication", "richness"],
        ),
        (
            "harmony",
            &["balance", "coherence", "unity", "symmetry", "accord"],
        ),
        (
            "balance",
            &["equilibrium", "harmony", "poise", "stability", "proportion"],
        ),
        (
            "motion",
            &["movement", "dynamics", "flow", "trajectory", "progression"],
        ),
        (
            "order",
            &["structure", "organization", "coherence", "arrangement"],
        ),
        (
            "chaos",
            &["disorder", "turbulence", "randomness", "entropy", "flux"],
        ),
        (
            "dimension",
            &["axis", "aspect", "realm", "domain", "magnitude"],
        ),
        (
            "infinity",
            &["boundlessness", "endlessness", "vastness", "eternity"],
        ),
        (
            "perception",
            &["awareness", "insight", "observation", "cognition", "sense"],
        ),
        (
            "meaning",
            &["significance", "substance", "essence", "purpose", "value"],
        ),
        (
            "existence",
            &["being", "reality", "presence", "life", "manifestation"],
        ),
        (
            "universe",
            &["cosmos", "world", "reality", "existence", "totality"],
        ),
        (
            "mathematics",
            &["algebra", "geometry", "calculus", "arithmetic", "analysis"],
        ),
        (
            "equation",
            &["formula", "expression", "relationship", "model", "identity"],
        ),
        (
            "frequency",
            &["resonance", "oscillation", "wavelength", "rhythm", "rate"],
        ),
        (
            "resonance",
            &["harmony", "vibration", "coherence", "synchrony", "accord"],
        ),
        (
            "evolution",
            &["transformation", "progression", "development", "dynamics"],
        ),
        (
            "boundary",
            &["limit", "threshold", "barrier", "edge", "frontier"],
        ),
        (
            "trajectory",
            &["path", "course", "orbit", "direction", "arc"],
        ),
        (
            "causality",
            &["determinism", "consequence", "mechanism", "logic", "reason"],
        ),
        (
            "logic",
            &[
                "reasoning",
                "rationality",
                "inference",
                "deduction",
                "thought",
            ],
        ),
        (
            "force",
            &["energy", "power", "influence", "dynamics", "pressure"],
        ),
        ("space", &["realm", "domain", "expanse", "field", "region"]),
        (
            "time",
            &["moment", "epoch", "duration", "continuity", "period"],
        ),
        ("field", &["domain", "region", "space", "realm", "expanse"]),
        (
            "wave",
            &["oscillation", "vibration", "ripple", "undulation", "pulse"],
        ),
        (
            "signal",
            &["indicator", "marker", "pattern", "trace", "message"],
        ),
        (
            "computation",
            &["calculation", "processing", "evaluation", "analysis"],
        ),
        (
            "simulation",
            &["modeling", "emulation", "representation", "approximation"],
        ),
        (
            "prediction",
            &["forecast", "projection", "estimation", "inference"],
        ),
        (
            "discovery",
            &[
                "revelation",
                "finding",
                "insight",
                "breakthrough",
                "perception",
            ],
        ),
        (
            "foundation",
            &["basis", "grounding", "core", "bedrock", "principle"],
        ),
        ("principle", &["law", "rule", "axiom", "tenet", "doctrine"]),
        (
            "intelligence",
            &["cognition", "awareness", "reasoning", "understanding"],
        ),
        (
            "consciousness",
            &["awareness", "perception", "cognition", "sentience"],
        ),
        (
            "mathematical",
            &["symbolic", "geometric", "algebraic", "analytical", "formal"],
        ),
        (
            "deterministic",
            &["predictable", "systematic", "causal", "precise", "exact"],
        ),
        (
            "probabilistic",
            &[
                "stochastic",
                "statistical",
                "uncertain",
                "random",
                "variable",
            ],
        ),
        (
            "infinite",
            &["boundless", "endless", "vast", "immeasurable", "limitless"],
        ),
        (
            "fundamental",
            &["essential", "core", "primary", "foundational", "elemental"],
        ),
        (
            "dynamic",
            &["evolving", "fluid", "active", "continuous", "adaptive"],
        ),
        (
            "abstract",
            &["symbolic", "conceptual", "theoretical", "pure", "ideal"],
        ),
        (
            "continuous",
            &["unbroken", "flowing", "perpetual", "sustained", "smooth"],
        ),
        (
            "discrete",
            &["distinct", "separate", "finite", "quantized", "isolated"],
        ),
        (
            "invariant",
            &["constant", "stable", "fixed", "unchanging", "conserved"],
        ),
        (
            "coherent",
            &[
                "unified",
                "consistent",
                "structured",
                "harmonious",
                "ordered",
            ],
        ),
        (
            "structural",
            &["architectural", "organizational", "systematic", "formal"],
        ),
        (
            "axiomatic",
            &["foundational", "self-evident", "primary", "elemental"],
        ),
        (
            "bounded",
            &[
                "finite",
                "constrained",
                "limited",
                "contained",
                "restricted",
            ],
        ),
        (
            "symmetric",
            &["balanced", "regular", "uniform", "proportional", "equal"],
        ),
        (
            "elegant",
            &["refined", "sophisticated", "beautiful", "graceful", "pure"],
        ),
        (
            "precise",
            &["exact", "accurate", "rigorous", "meticulous", "definite"],
        ),
        (
            "universal",
            &["general", "global", "absolute", "total", "pervasive"],
        ),
        (
            "recursive",
            &["iterative", "self-referential", "repetitive", "cyclic"],
        ),
        (
            "emergent",
            &[
                "arising",
                "evolving",
                "developing",
                "unfolding",
                "appearing",
            ],
        ),
        (
            "complex",
            &["intricate", "sophisticated", "multifaceted", "rich", "deep"],
        ),
        (
            "simple",
            &["elementary", "basic", "pure", "direct", "minimal"],
        ),
        (
            "ancient",
            &[
                "primordial",
                "prehistoric",
                "archaic",
                "classical",
                "timeless",
            ],
        ),
        (
            "sacred",
            &["divine", "revered", "hallowed", "eternal", "transcendent"],
        ),
        (
            "cosmic",
            &["universal", "celestial", "infinite", "vast", "transcendent"],
        ),
        (
            "hidden",
            &["concealed", "latent", "underlying", "subtle", "implicit"],
        ),
        (
            "deeper",
            &["profound", "fundamental", "underlying", "essential", "core"],
        ),
        (
            "new",
            &["novel", "emerging", "fresh", "modern", "innovative"],
        ),
        (
            "pure",
            &[
                "exact",
                "unadulterated",
                "precise",
                "fundamental",
                "essential",
            ],
        ),
        (
            "true",
            &["genuine", "authentic", "real", "valid", "accurate"],
        ),
        (
            "great",
            &[
                "profound",
                "vast",
                "significant",
                "remarkable",
                "extraordinary",
            ],
        ),
        (
            "vast",
            &["immense", "expansive", "boundless", "infinite", "enormous"],
        ),
        (
            "known",
            &[
                "established",
                "recognized",
                "understood",
                "observed",
                "verified",
            ],
        ),
        (
            "seen",
            &["observed", "perceived", "recognized", "witnessed", "noted"],
        ),
        (
            "made",
            &["constructed", "formed", "created", "built", "composed"],
        ),
        (
            "called",
            &["named", "termed", "labeled", "designated", "referred"],
        ),
    ];

    for (word, synonyms) in entries {
        m.insert(
            word.to_string(),
            synonyms.iter().map(|s| s.to_string()).collect(),
        );
    }

    m
}

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

pub struct SynonymBank {
    curated: HashMap<String, Vec<String>>,
    by_length: HashMap<usize, Vec<String>>,
}

impl SynonymBank {
    pub fn new() -> Self {
        Self {
            curated: build_curated_table(),
            by_length: load_wordlist_by_length(),
        }
    }

    pub fn candidate<R: Rng>(&self, word: &str, rng: &mut R) -> Option<String> {
        let lower = word.to_lowercase();
        if let Some(syns) = self.curated.get(&lower) {
            let idx = rng.random_range(0..syns.len());
            return Some(syns[idx].clone());
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

    pub fn curated_count(&self) -> usize {
        self.curated.len()
    }

    pub fn wordlist_len(&self) -> usize {
        self.by_length.values().map(|v| v.len()).sum()
    }
}

impl Default for SynonymBank {
    fn default() -> Self {
        Self::new()
    }
}

pub struct StochasticEnhancer {
    bank: SynonymBank,
    probability: f64,
}

impl StochasticEnhancer {
    pub fn new(probability: f64) -> Self {
        Self {
            bank: SynonymBank::new(),
            probability: probability.clamp(0.0, 1.0),
        }
    }

    pub fn with_default_probability() -> Self {
        Self::new(DEFAULT_REPLACEMENT_PROBABILITY)
    }

    pub fn enhance(&self, text: &str) -> String {
        let mut rng = rand::rng();
        text.split('\n')
            .map(|line| self.enhance_line(line, &mut rng))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn enhance_line<R: Rng>(&self, line: &str, rng: &mut R) -> String {
        let words = line.split_whitespace().peekable();
        let mut result: Vec<String> = Vec::new();
        let mut is_first = true;

        for raw in words {
            let (prefix, core, suffix) = split_token(raw);
            let lower = core.to_lowercase();
            let is_stop = STOP_WORDS.contains(&lower.as_str());
            let was_capitalized = core
                .chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false);
            let was_all_caps = core.chars().all(|c| c.is_uppercase());

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

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}
