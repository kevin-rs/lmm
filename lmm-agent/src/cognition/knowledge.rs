// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `knowledge` - Knowledge Acquisition for Agents
//!
//! This module enables agents to build a queryable knowledge base from diverse
//! local and remote sources, then retrieve grounded, extractive answers to
//! natural-language questions, entirely offline and without any LLM.
//!
//! ## Sources
//!
//! | Variant | Description |
//! |---|---|
//! | `KnowledgeSource::File(path)` | A single `.txt`, `.md`, or `.pdf` file |
//! | `KnowledgeSource::Dir(path)` | All parseable files within a directory |
//! | `KnowledgeSource::Url(url)` | Fetch and parse a web URL (`net` feature) |
//! | `KnowledgeSource::RawText(text)` | Inline text string |
//!
//! ## Retrieval algorithm
//!
//! 1. Tokenise the question into lowercase content words (≥ 3 chars, stop-words removed).
//! 2. Score every indexed [`DocumentChunk`] with IDF-weighted token overlap.
//! 3. Take the top-k chunks and concatenate their text.
//! 4. Feed the concatenated corpus + the original question into
//!    [`lmm::text::TextSummarizer::summarize_with_query`] to produce a short
//!    extractive answer.
//!
//! ## Quick example
//!
//! ```rust
//! use lmm_agent::cognition::knowledge::{KnowledgeIndex, KnowledgeSource};
//!
//! let mut index = KnowledgeIndex::new();
//! index.ingest_text("my-doc", "Rust gives you control over memory without a garbage collector. \
//!                              The borrow checker enforces safety at compile time.");
//! let answer = index.answer("How does Rust handle memory?", 3);
//! assert!(answer.is_some());
//! ```

use anyhow::{Result, anyhow};
use lmm::text::TextSummarizer;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs;
#[cfg(feature = "knowledge")]
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// A source from which an agent can acquire knowledge.
#[derive(Debug, Clone)]
pub enum KnowledgeSource {
    /// A local file.  Supported extensions: `.txt`, `.md`, and `.pdf` (requires `knowledge`).
    File(PathBuf),
    /// All parseable files found directly inside this directory (non-recursive).
    Dir(PathBuf),
    /// Fetch plain text from a URL. Requires the `net` feature.
    Url(String),
    /// Inline text supplied directly by the caller.
    RawText(String),
}

/// A sentence-sized unit of ingested knowledge.
#[derive(Debug, Clone)]
pub struct DocumentChunk {
    /// Human-readable label for the source (filename or URL).
    pub source: String,
    /// The raw text of this chunk.
    pub text: String,
    /// Pre-tokenised, lowercase content words used for fast scoring.
    pub tokens: Vec<String>,
}

impl DocumentChunk {
    pub fn new(source: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        let tokens = tokenise(&text);
        Self {
            source: source.into(),
            text,
            tokens,
        }
    }
}

/// In-memory, term-indexed knowledge base built from ingested documents.
///
/// Retrieval is based on IDF-weighted token overlap: content words that appear
/// rarely across the corpus are weighted more heavily, focusing results on
/// discriminative evidence.
///
/// # Examples
///
/// ```rust
/// use lmm_agent::cognition::knowledge::KnowledgeIndex;
///
/// let mut idx = KnowledgeIndex::new();
/// idx.ingest_text(
///     "rust-book",
///     "Rust prevents data races at compile time through its ownership model.",
/// );
/// let hits = idx.query("What prevents data races in Rust?", 3);
/// assert!(!hits.is_empty());
/// ```
#[derive(Debug, Clone, Default)]
pub struct KnowledgeIndex {
    chunks: Vec<DocumentChunk>,
    term_index: HashMap<String, Vec<usize>>,
    doc_freq: HashMap<String, usize>,
}

impl KnowledgeIndex {
    /// Creates an empty `KnowledgeIndex`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if no documents have been ingested yet.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Returns the total number of indexed chunks.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Ingests a raw text string under the given `source` label.
    ///
    /// The text is split into sentence-level chunks via a lightweight sentence
    /// splitter before being indexed.  Returns the number of chunks created.
    pub fn ingest_text(&mut self, source: &str, text: &str) -> usize {
        let sentences = split_sentences(text);
        let start = self.chunks.len();
        for sentence in sentences {
            if sentence.split_whitespace().count() < 4 {
                continue;
            }
            let chunk = DocumentChunk::new(source, &sentence);
            let idx = self.chunks.len();
            for token in &chunk.tokens {
                self.term_index.entry(token.clone()).or_default().push(idx);
            }
            self.chunks.push(chunk);
        }
        let added = self.chunks.len() - start;
        self.rebuild_doc_freq();
        added
    }

    /// Returns the `top_k` most relevant [`DocumentChunk`]s for `question`.
    ///
    /// Chunks are scored by the sum of IDF weights for each question token that
    /// appears in the chunk.
    pub fn query(&self, question: &str, top_k: usize) -> Vec<&DocumentChunk> {
        if self.chunks.is_empty() {
            return Vec::new();
        }
        let q_tokens = tokenise(question);
        let n = self.chunks.len() as f64;

        let mut scores: Vec<(usize, f64)> = (0..self.chunks.len())
            .map(|i| {
                let chunk = &self.chunks[i];
                let score: f64 = q_tokens
                    .iter()
                    .filter(|t| chunk.tokens.contains(t))
                    .map(|t| {
                        let df = *self.doc_freq.get(t).unwrap_or(&1) as f64;
                        (n / df).ln() + 1.0
                    })
                    .sum();
                (i, score)
            })
            .filter(|(_, s)| *s > 0.0)
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scores
            .into_iter()
            .take(top_k)
            .map(|(i, _)| &self.chunks[i])
            .collect()
    }

    /// Returns an extractive answer to `question` from the knowledge base,
    /// or `None` if no relevant chunks are found.
    ///
    /// The top-`top_k` chunks are concatenated and passed to
    /// [`lmm::text::TextSummarizer`] with the original question as a relevance
    /// hint. The summariser selects the sentences most likely to answer the question.
    pub fn answer(&self, question: &str, top_k: usize) -> Option<String> {
        let hits = self.query(question, top_k);
        if hits.is_empty() {
            return None;
        }
        let corpus: String = hits
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let summariser = TextSummarizer::new(3, 4, 2);
        summariser
            .summarize_with_query(&corpus, question)
            .ok()
            .map(|sentences| sentences.join(" "))
    }

    fn rebuild_doc_freq(&mut self) {
        self.doc_freq.clear();
        for chunk in &self.chunks {
            let mut seen = HashSet::new();
            for token in &chunk.tokens {
                if seen.insert(token) {
                    *self.doc_freq.entry(token.clone()).or_insert(0) += 1;
                }
            }
        }
    }
}

/// Trait for pluggable document parsers.
pub trait DocumentParser: Send + Sync {
    /// Returns `true` when this parser can handle the given file extension.
    fn supports_extension(&self, ext: &str) -> bool;
    /// Parses `bytes` into a plain-text string.
    fn parse_bytes(&self, bytes: &[u8]) -> Result<String>;
}

/// Parses plain `.txt` files (UTF-8 assumed, with a lossy fallback).
#[derive(Debug, Default, Clone)]
pub struct PlainTextParser;

impl DocumentParser for PlainTextParser {
    fn supports_extension(&self, ext: &str) -> bool {
        ext.eq_ignore_ascii_case("txt")
    }

    fn parse_bytes(&self, bytes: &[u8]) -> Result<String> {
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
}

/// Parses `.md` (Markdown) files by stripping formatting markers and yielding plain text.
#[derive(Debug, Default, Clone)]
pub struct MarkdownParser;

impl DocumentParser for MarkdownParser {
    fn supports_extension(&self, ext: &str) -> bool {
        matches!(ext.to_ascii_lowercase().as_str(), "md" | "markdown")
    }

    fn parse_bytes(&self, bytes: &[u8]) -> Result<String> {
        let raw = String::from_utf8_lossy(bytes);
        Ok(strip_markdown(&raw))
    }
}

/// Parses `.pdf` files using `lopdf`. Requires the `knowledge` feature.
#[cfg(feature = "knowledge")]
#[derive(Debug, Default, Clone)]
pub struct PdfParser;

#[cfg(feature = "knowledge")]
impl DocumentParser for PdfParser {
    fn supports_extension(&self, ext: &str) -> bool {
        ext.eq_ignore_ascii_case("pdf")
    }

    fn parse_bytes(&self, bytes: &[u8]) -> Result<String> {
        #[cfg(feature = "knowledge")]
        use lopdf::Document;

        let doc =
            Document::load_from(Cursor::new(bytes)).map_err(|e| anyhow!("lopdf error: {e}"))?;
        let mut out = String::new();
        for page_num in 1..=doc.get_pages().len() as u32 {
            if let Ok(texts) = doc.extract_text(&[page_num]) {
                out.push_str(&texts);
                out.push('\n');
            }
        }
        Ok(out)
    }
}

/// Returns a list of parsers active for the current feature set.
pub fn default_parsers() -> Vec<Box<dyn DocumentParser>> {
    #[allow(unused_mut)]
    let mut parsers: Vec<Box<dyn DocumentParser>> =
        vec![Box::new(PlainTextParser), Box::new(MarkdownParser)];
    #[cfg(feature = "knowledge")]
    parsers.push(Box::new(PdfParser));
    parsers
}

/// Ingest a [`KnowledgeSource`] into `index`, returning the number of new chunks.
///
/// This function is the top-level entry point used by [`LmmAgent::ingest`].
pub async fn ingest(index: &mut KnowledgeIndex, source: KnowledgeSource) -> Result<usize> {
    match source {
        KnowledgeSource::RawText(text) => Ok(index.ingest_text("inline", &text)),

        KnowledgeSource::File(path) => ingest_file(index, &path),

        KnowledgeSource::Dir(dir) => {
            let mut total = 0;
            let entries = fs::read_dir(&dir)
                .map_err(|e| anyhow!("Cannot read dir {}: {e}", dir.display()))?;
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() {
                    total += ingest_file(index, &p).unwrap_or(0);
                }
            }
            Ok(total)
        }

        KnowledgeSource::Url(url) => ingest_url(index, &url).await,
    }
}

fn ingest_file(index: &mut KnowledgeIndex, path: &Path) -> Result<usize> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let parsers = default_parsers();
    let parser = parsers
        .iter()
        .find(|p| p.supports_extension(&ext))
        .ok_or_else(|| anyhow!("No parser for '{}' (extension: .{})", path.display(), ext))?;

    let bytes = fs::read(path).map_err(|e| anyhow!("Cannot read {}: {e}", path.display()))?;
    let text = parser.parse_bytes(&bytes)?;
    let label = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    Ok(index.ingest_text(label, &text))
}

#[cfg(feature = "net")]
async fn ingest_url(index: &mut KnowledgeIndex, url: &str) -> Result<usize> {
    let body = reqwest::get(url)
        .await
        .map_err(|e| anyhow!("HTTP error for {url}: {e}"))?
        .text()
        .await
        .map_err(|e| anyhow!("Body error for {url}: {e}"))?;
    let plain = strip_html(&body);
    Ok(index.ingest_text(url, &plain))
}

#[cfg(not(feature = "net"))]
async fn ingest_url(_index: &mut KnowledgeIndex, url: &str) -> Result<usize> {
    Err(anyhow!(
        "URL ingestion requires the `net` feature. Enable it with --features net. URL: {url}"
    ))
}

static STOP_WORDS: phf::Set<&'static str> = phf::phf_set! {
    "the","and","for","are","was","that","this","with","from","have",
    "not","but","had","has","its","were","they","will","been","their",
    "all","one","can","her","his","him","she","who","which","what",
    "into","then","than","when","also","more","some","out","about",
    "said","would","could","should","each","other","there","these",
    "those","such","any","our","you","your","very","just","now","may"
};

fn tokenise(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphabetic())
        .filter_map(|w| {
            let w = w.to_ascii_lowercase();
            if w.len() >= 3 && !STOP_WORDS.contains(w.as_str()) {
                Some(w)
            } else {
                None
            }
        })
        .collect()
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &ch) in chars.iter().enumerate() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let next_lower = chars
                .get(i + 1)
                .map(|c| c.is_ascii_lowercase())
                .unwrap_or(false);
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_digit = chars
                .get(i + 1)
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false);
            if ch == '.' && (next_lower || (prev_digit && next_digit)) {
                continue;
            }
            let s = cur.trim().to_string();
            if !s.is_empty() {
                out.push(s);
            }
            cur.clear();
        }
    }
    let tail = cur.trim().to_string();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

fn strip_markdown(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_code = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_code = !in_code;
            continue;
        }
        if in_code {
            continue;
        }
        if trimmed.starts_with('#') || trimmed.starts_with("---") || trimmed.starts_with("===") {
            continue;
        }
        let clean: String = trimmed
            .chars()
            .filter(|&c| c != '*' && c != '_' && c != '`' && c != '|')
            .collect();
        let clean = clean.trim();
        if !clean.is_empty() {
            out.push_str(clean);
            out.push(' ');
        }
    }
    out
}

#[allow(dead_code)]
fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let mut in_tag = false;
    let mut buf = String::new();
    for ch in html.chars() {
        match ch {
            '<' => {
                if !buf.trim().is_empty() {
                    out.push_str(buf.trim());
                    out.push(' ');
                }
                buf.clear();
                in_tag = true;
            }
            '>' => {
                in_tag = false;
            }
            _ if !in_tag => {
                buf.push(ch);
            }
            _ => {}
        }
    }
    if !buf.trim().is_empty() {
        out.push_str(buf.trim());
    }
    out
}
