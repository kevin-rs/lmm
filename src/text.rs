use crate::error::{LmmError, Result};
use crate::lexicon::word_tone;

static TRANSITIVE_VERBS: &[(&str, f64)] = &[
    ("reveals", 113.1),
    ("encodes", 107.7),
    ("governs", 111.3),
    ("shapes", 106.8),
    ("defines", 107.8),
    ("captures", 109.3),
    ("reflects", 109.6),
    ("transforms", 115.8),
    ("determines", 113.5),
    ("expresses", 116.4),
    ("describes", 109.6),
    ("manifests", 112.8),
    ("illuminates", 115.3),
    ("compresses", 112.5),
    ("represents", 116.9),
    ("unveils", 109.8),
    ("generates", 114.5),
    ("produces", 112.0),
    ("enables", 106.8),
    ("connects", 109.0),
];

static LINKING_VERBS: &[(&str, f64)] = &[
    ("is", 109.5),
    ("are", 107.7),
    ("forms", 108.3),
    ("becomes", 111.9),
    ("remains", 111.5),
    ("holds", 107.0),
];

static SUBJECT_NOUNS: &[(&str, f64)] = &[
    ("mathematics", 112.4),
    ("geometry", 113.2),
    ("physics", 108.5),
    ("logic", 107.3),
    ("algebra", 105.3),
    ("calculus", 108.8),
    ("symmetry", 116.7),
    ("entropy", 113.7),
    ("topology", 116.8),
    ("probability", 114.2),
    ("computation", 116.8),
    ("information", 115.7),
    ("simulation", 116.0),
    ("equation", 111.0),
    ("analysis", 110.4),
    ("recursion", 113.3),
    ("resonance", 113.3),
    ("frequency", 115.4),
    ("wavelength", 113.8),
    ("dimension", 113.8),
    ("structure", 115.3),
    ("pattern", 110.3),
    ("gradient", 110.3),
    ("divergence", 113.0),
    ("integration", 115.4),
    ("transformation", 119.5),
];

static OBJECT_NOUNS: &[(&str, f64)] = &[
    ("reality", 112.5),
    ("truth", 110.6),
    ("complexity", 116.3),
    ("order", 108.2),
    ("chaos", 103.5),
    ("harmony", 109.0),
    ("existence", 114.0),
    ("nature", 109.2),
    ("matter", 108.2),
    ("energy", 107.2),
    ("time", 108.2),
    ("space", 107.5),
    ("motion", 107.5),
    ("change", 107.2),
    ("balance", 106.3),
    ("infinity", 111.9),
    ("symmetry", 116.7),
    ("unity", 111.7),
    ("identity", 112.6),
    ("causality", 113.0),
    ("meaning", 108.7),
    ("knowledge", 112.0),
    ("perception", 113.2),
    ("boundaries", 115.1),
    ("limits", 107.5),
    ("foundations", 117.7),
];

static ADJECTIVES: &[(&str, f64)] = &[
    ("fundamental", 116.1),
    ("mathematical", 114.5),
    ("universal", 115.0),
    ("infinite", 110.8),
    ("precise", 111.8),
    ("elegant", 109.6),
    ("structural", 116.0),
    ("invariant", 113.0),
    ("dynamic", 109.7),
    ("recursive", 113.8),
    ("continuous", 115.4),
    ("discrete", 112.3),
    ("deterministic", 120.0),
    ("probabilistic", 117.5),
    ("axiomatic", 116.0),
    ("abstract", 109.6),
    ("emergent", 111.0),
    ("coherent", 112.3),
    ("symmetric", 117.8),
    ("bounded", 109.8),
];

static SENTENCE_CONNECTORS: &[&str] = &[
    "Furthermore,",
    "Moreover,",
    "Indeed,",
    "Consequently,",
    "In essence,",
    "At its core,",
    "As a result,",
    "Fundamentally,",
    "More precisely,",
    "By extension,",
    "Through this lens,",
    "In this framework,",
];

static PREPOSITIONS_RICH: &[&str] = &[
    "of",
    "within",
    "beyond",
    "through",
    "across",
    "beneath",
    "inside",
    "underlying",
    "pervading",
    "governing",
];

fn text_tone(s: &str) -> f64 {
    let bytes: Vec<u8> = s.bytes().filter(|b| b.is_ascii_alphabetic()).collect();
    if bytes.is_empty() {
        return 110.0;
    }
    bytes.iter().map(|&b| b as f64).sum::<f64>() / bytes.len() as f64
}

fn offset_by_tone<'a>(pool: &'a [(&'a str, f64)], target: f64, offset: usize) -> &'a str {
    let mut scored: Vec<(f64, &str)> = pool
        .iter()
        .map(|(w, t)| ((*t - target).abs(), *w))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .get(offset % scored.len().max(1))
        .map(|(_, w)| *w)
        .unwrap_or("")
}

fn offset_by_tone_not<'a>(
    pool: &'a [(&'a str, f64)],
    target: f64,
    offset: usize,
    exclude: &str,
) -> &'a str {
    let mut scored: Vec<(f64, &str)> = pool
        .iter()
        .filter(|(w, _)| *w != exclude)
        .map(|(w, t)| ((*t - target).abs(), *w))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .get(offset % scored.len().max(1))
        .map(|(_, w)| *w)
        .unwrap_or("")
}

fn split_into_sentences(text: &str) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    for i in 0..len {
        let ch = chars[i];
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' {
            let prev_is_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_is_digit = i + 1 < len && chars[i + 1].is_ascii_digit();
            if ch == '.' && prev_is_digit && next_is_digit {
                continue;
            }
            let next_is_lower = i + 1 < len && chars[i + 1].is_ascii_lowercase();
            if ch == '.' && next_is_lower {
                continue;
            }
            let s = current.trim().to_string();
            if s.split_whitespace().count() >= 5 {
                sentences.push(s);
            }
            current.clear();
        }
    }
    let tail = current.trim().to_string();
    if tail.split_whitespace().count() >= 5 {
        sentences.push(tail);
    }
    sentences
}

fn keywords_from_text(text: &str, n: usize) -> Vec<String> {
    let verb_set: std::collections::HashSet<&str> = TRANSITIVE_VERBS
        .iter()
        .map(|(w, _)| *w)
        .chain(LINKING_VERBS.iter().map(|(w, _)| *w))
        .chain(ADJECTIVES.iter().map(|(w, _)| *w))
        .chain(
            [
                "this", "that", "their", "also", "with", "from", "into", "will", "have", "been",
                "more", "very", "most", "such", "than", "when", "early", "first", "some", "only",
                "both", "each", "many", "other", "modern", "ancient",
            ]
            .iter()
            .copied(),
        )
        .collect();

    let mut seen = std::collections::HashSet::new();
    let mut word_tones: Vec<(f64, String)> = text
        .split_whitespace()
        .filter_map(|w| {
            let clean: String = w
                .chars()
                .filter(|c| c.is_ascii_alphabetic())
                .collect::<String>()
                .to_ascii_lowercase();
            if clean.len() >= 4 && !verb_set.contains(clean.as_str()) && seen.insert(clean.clone())
            {
                Some((word_tone(&clean), clean))
            } else {
                None
            }
        })
        .collect();
    word_tones.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    word_tones.into_iter().take(n).map(|(_, w)| w).collect()
}

fn seed_hash(s: &str) -> usize {
    s.bytes().enumerate().fold(0usize, |acc, (i, b)| {
        acc.wrapping_add((b as usize).wrapping_mul(i.wrapping_add(31)))
    })
}

pub struct SentenceGenerator {
    pub iterations: usize,
    pub depth: usize,
}

impl SentenceGenerator {
    pub fn new(iterations: usize, depth: usize) -> Self {
        Self { iterations, depth }
    }
    pub fn generate_variant(&self, seed: &str, variant: usize) -> Result<String> {
        let tone = text_tone(seed);
        let sh = seed_hash(seed);

        let keywords = keywords_from_text(seed, 3);
        let topic = keywords.first().map(|s| s.as_str()).unwrap_or("reality");

        let v = variant.wrapping_add(sh);

        let sentence = match variant % 6 {
            0 => {
                let subj = offset_by_tone(SUBJECT_NOUNS, tone, v);
                let verb = offset_by_tone(TRANSITIVE_VERBS, tone, v + 1);
                let adj = offset_by_tone(ADJECTIVES, tone - 2.0, v + 2);
                let obj = offset_by_tone(OBJECT_NOUNS, tone + 1.0, v + 3);
                format!(
                    "{} {} the {} {} of the {}.",
                    capitalize(subj),
                    verb,
                    adj,
                    obj,
                    topic
                )
            }
            1 => {
                let adj = offset_by_tone(ADJECTIVES, tone, v);
                let subj = offset_by_tone(SUBJECT_NOUNS, tone + 2.0, v + 1);
                let verb = offset_by_tone(TRANSITIVE_VERBS, tone + 1.0, v + 2);
                let obj = offset_by_tone(OBJECT_NOUNS, tone - 2.0, v + 3);
                format!("The {} {} {} {}.", adj, subj, verb, obj)
            }
            2 => {
                let subj = offset_by_tone(SUBJECT_NOUNS, tone + 1.5, v);
                let link = offset_by_tone(LINKING_VERBS, tone, v + 1);
                let adj = offset_by_tone(ADJECTIVES, tone + 3.0, v + 2);
                let prep = PREPOSITIONS_RICH[v % PREPOSITIONS_RICH.len()];
                let obj = offset_by_tone_not(OBJECT_NOUNS, tone - 1.0, v + 3, topic);
                format!(
                    "{} {} the {} {} {} {}.",
                    capitalize(subj),
                    link,
                    adj,
                    topic,
                    prep,
                    obj
                )
            }
            3 => {
                let subj = offset_by_tone(SUBJECT_NOUNS, tone, v + 1);
                let verb = offset_by_tone(TRANSITIVE_VERBS, tone + 2.0, v + 2);
                let obj = offset_by_tone(OBJECT_NOUNS, tone, v + 3);
                format!("The {} of {} {} {}.", subj, topic, verb, obj)
            }
            4 => {
                let connector = SENTENCE_CONNECTORS[v % SENTENCE_CONNECTORS.len()];
                let adj = offset_by_tone(ADJECTIVES, tone - 1.0, v + 2);
                let subj = offset_by_tone(SUBJECT_NOUNS, tone + 1.0, v + 3);
                let verb = offset_by_tone(TRANSITIVE_VERBS, tone, v + 4);
                let obj = offset_by_tone(OBJECT_NOUNS, tone + 2.0, v + 5);
                format!("{} the {} {} {} {}.", connector, adj, subj, verb, obj)
            }
            _ => {
                let subj = offset_by_tone(SUBJECT_NOUNS, tone + 3.0, v + 1);
                let verb = offset_by_tone(TRANSITIVE_VERBS, tone + 1.5, v + 2);
                let adj = offset_by_tone(ADJECTIVES, tone - 3.0, v + 3);
                let obj = offset_by_tone(OBJECT_NOUNS, tone, v + 4);
                let prep = PREPOSITIONS_RICH[(v + 2) % PREPOSITIONS_RICH.len()];
                format!(
                    "{} {} {} {} {} {}.",
                    capitalize(subj),
                    verb,
                    adj,
                    obj,
                    prep,
                    topic
                )
            }
        };

        Ok(sentence)
    }

    pub fn generate(&self, seed: &str) -> Result<String> {
        self.generate_variant(seed, 0)
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => {
            let upper: String = f.to_uppercase().collect();
            upper + c.as_str()
        }
    }
}

fn sentence_has_verb(lower: &str) -> bool {
    let present_tense = [
        " is ",
        " are ",
        " was ",
        " were ",
        " has ",
        " have ",
        " supports ",
        " provides ",
        " enables ",
        " uses ",
        " describes ",
        " represents ",
        " includes ",
        " can ",
        " will ",
        " does ",
        " do ",
        " spans ",
        " emphasizes ",
        " creates ",
        " allows ",
        " helps ",
        " makes ",
        " runs ",
        " enforces ",
        " prevents ",
        " reduces ",
        " increases ",
        " remains ",
        " requires ",
        " relies ",
        " acts ",
        " forms ",
    ];
    if present_tense.iter().any(|&v| lower.contains(v)) {
        return true;
    }
    let irregulars = [
        " built ",
        " made ",
        " went ",
        " came ",
        " gave ",
        " took ",
        " found ",
        " knew ",
        " thought ",
        " brought ",
        " left ",
        " became ",
        " saw ",
        " led ",
        " kept ",
        " held ",
        " told ",
        " got ",
        " set ",
        " put ",
        " let ",
        " cut ",
        " hit ",
        " ran ",
        " stood ",
        " lost ",
        " won ",
        " fell ",
        " grew ",
        " bore ",
        " drew ",
        " rose ",
        " wore ",
        " spoke ",
        " wrote ",
        " chose ",
        " drove ",
        " began ",
        " swam ",
        " flew ",
        " threw ",
        " struggled ",
        " struggle ",
        " replicate ",
        " align ",
    ];
    if irregulars.iter().any(|&v| lower.contains(v)) {
        return true;
    }
    for word in lower.split_whitespace() {
        let w = word.trim_matches(|c: char| !c.is_ascii_alphabetic());
        if w.len() > 4 && w.ends_with("ed") {
            return true;
        }
        if w.len() > 6 && w.ends_with("ing") {
            return true;
        }
    }
    false
}

pub struct TextSummarizer {
    pub sentence_count: usize,
    pub iterations: usize,
    pub depth: usize,
}

impl TextSummarizer {
    pub fn new(sentence_count: usize, iterations: usize, depth: usize) -> Self {
        Self {
            sentence_count,
            iterations,
            depth,
        }
    }

    pub fn summarize(&self, text: &str) -> Result<Vec<String>> {
        self.summarize_with_query(text, "")
    }

    pub fn summarize_with_query(&self, text: &str, query: &str) -> Result<Vec<String>> {
        let sentences = split_into_sentences(text);
        if sentences.is_empty() {
            return Err(LmmError::Perception(
                "No complete sentences found in input.".into(),
            ));
        }
        if sentences.len() <= self.sentence_count {
            return Ok(deduplicate_sentences(sentences));
        }

        let global_tone = text_tone(text);
        let avg_len =
            sentences.iter().map(|s| s.len()).sum::<usize>() as f64 / sentences.len() as f64;
        let want = self.sentence_count;

        let query_keywords: Vec<String> = query
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .map(|w| w.to_lowercase())
            .collect();

        let mut scored: Vec<(usize, f64)> = sentences
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let lower = s.to_lowercase();
                let tone = text_tone(s);
                let tone_delta = (tone - global_tone).abs();

                let len_ratio = s.len() as f64 / avg_len.max(1.0);
                let len_score = len_ratio.min(2.5);

                let early_bonus = if i == 0 { 2.5 } else { 0.0 };

                let comma_count = s.matches(',').count();
                let comma_penalty = if comma_count > 3 {
                    (comma_count as f64 - 3.0) * 1.8
                } else {
                    0.0
                };

                let has_verb = sentence_has_verb(&lower);
                let verb_bonus = if has_verb { 1.2 } else { -3.0 };

                let relevance_bonus: f64 = if query_keywords.is_empty() {
                    0.0
                } else {
                    let matches = query_keywords
                        .iter()
                        .filter(|kw| lower.contains(kw.as_str()))
                        .count();
                    matches as f64 * 0.8
                };

                let total = len_score * 1.0 + early_bonus - tone_delta * 0.3 - comma_penalty
                    + verb_bonus
                    + relevance_bonus;
                (i, total)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected: Vec<usize> = scored.into_iter().take(want).map(|(i, _)| i).collect();
        selected.sort_unstable();
        selected.dedup();

        let result: Vec<String> =
            deduplicate_sentences(selected.into_iter().map(|i| sentences[i].clone()).collect());
        Ok(result)
    }
}

fn deduplicate_sentences(sentences: Vec<String>) -> Vec<String> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    sentences
        .into_iter()
        .filter(|s| {
            let key = s
                .split_whitespace()
                .take(6)
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase();
            seen.insert(key)
        })
        .collect()
}

pub struct ParagraphGenerator {
    pub sentence_count: usize,
    pub iterations: usize,
    pub depth: usize,
}

impl ParagraphGenerator {
    pub fn new(sentence_count: usize, iterations: usize, depth: usize) -> Self {
        Self {
            sentence_count,
            iterations,
            depth,
        }
    }

    pub fn generate(&self, seed: &str) -> Result<String> {
        let base_gen = SentenceGenerator::new(self.iterations, self.depth);
        let keywords = keywords_from_text(seed, self.sentence_count.max(3));
        let mut sentences: Vec<String> = Vec::with_capacity(self.sentence_count);

        for i in 0..self.sentence_count {
            let sub_seed = if i == 0 {
                seed.to_string()
            } else if let Some(kw) = keywords.get(i) {
                format!("{} {}", seed, kw)
            } else {
                seed.to_string()
            };
            sentences.push(base_gen.generate_variant(&sub_seed, i)?);
        }

        Ok(sentences.join(" "))
    }
}

pub struct EssayGenerator {
    pub paragraph_count: usize,
    pub sentence_count: usize,
    pub iterations: usize,
    pub depth: usize,
}

impl EssayGenerator {
    pub fn new(
        paragraph_count: usize,
        sentence_count: usize,
        iterations: usize,
        depth: usize,
    ) -> Self {
        Self {
            paragraph_count,
            sentence_count,
            iterations,
            depth,
        }
    }

    pub fn generate(&self, topic: &str) -> Result<EssayOutput> {
        let title = topic
            .split_whitespace()
            .take(7)
            .map(capitalize)
            .collect::<Vec<_>>()
            .join(" ");

        let keywords = keywords_from_text(topic, self.paragraph_count + 2);
        let para_gen = ParagraphGenerator::new(self.sentence_count, self.iterations, self.depth);
        let single_gen = SentenceGenerator::new(self.iterations, self.depth);

        let mut paragraphs: Vec<String> = Vec::new();

        let intro_seed = format!("{} fundamental truth", topic);
        paragraphs.push(para_gen.generate(&intro_seed)?);

        for (i, kw) in keywords.iter().take(self.paragraph_count).enumerate() {
            let body_seed = format!("{} {}", topic, kw);
            let mut body_sentences: Vec<String> = Vec::new();
            for j in 0..self.sentence_count {
                body_sentences.push(
                    single_gen.generate_variant(&body_seed, i * self.sentence_count + j + 4)?,
                );
            }
            paragraphs.push(body_sentences.join(" "));
        }

        let concl_seed = format!("{} coherence understanding", topic);
        paragraphs.push(para_gen.generate(&concl_seed)?);

        Ok(EssayOutput { title, paragraphs })
    }
}

pub struct EssayOutput {
    pub title: String,
    pub paragraphs: Vec<String>,
}
