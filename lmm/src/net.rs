// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Web Search Integration
//!
//! This module wraps the [`duckduckgo`] crate's client into a higher-level
//! [`SearchAggregator`] and provides utilities for converting raw search results
//! into clean text corpora for use as seeds in the text-generation pipeline.
//!
//! ## Key Changes
//!
//! - **`get_response`** now URL-encodes the query via a minimal percent-encoding
//!   implementation, so queries like `"E = mc²"` are correctly transmitted.
//! - **`SearchAggregator`** derives [`Default`] instead of maintaining a manual
//!   `Default` impl.

use anyhow::Result;
use duckduckgo::browser::Browser;
pub use duckduckgo::response::{LiteSearchResult, Response, ResultFormat};
use duckduckgo::user_agents::get as agent;

/// Percent-encodes a query string for safe inclusion in a URL query parameter.
///
/// Encodes all characters that are not unreserved (ASCII letters, digits, `-`, `_`,
/// `.`, `~`). Spaces become `%20` (not `+`) for maximum compatibility.
///
/// # Arguments
///
/// * `s` - The raw query string.
///
/// # Returns
///
/// (`String`): The percent-encoded string.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::net::url_encode;
///
/// assert_eq!(url_encode("hello world"), "hello%20world");
/// assert_eq!(url_encode("E=mc²"), "E%3Dmc%C2%B2");
/// assert_eq!(url_encode("safe-string_1.0~"), "safe-string_1.0~");
/// ```
pub fn url_encode(s: &str) -> String {
    let mut encoded = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            b => {
                encoded.push('%');
                encoded.push(
                    char::from_digit((b >> 4) as u32, 16)
                        .unwrap_or('0')
                        .to_ascii_uppercase(),
                );
                encoded.push(
                    char::from_digit((b & 0xf) as u32, 16)
                        .unwrap_or('0')
                        .to_ascii_uppercase(),
                );
            }
        }
    }
    encoded
}

/// A DuckDuckGo search aggregator that wraps the `duckduckgo` client.
///
/// # Examples
///
/// ```rust,ignore
/// use lmm::net::SearchAggregator;
///
/// #[tokio::main]
/// async fn main() {
///     let agg = SearchAggregator::new().with_region("us-en");
///     let results = agg.fetch("Rust programming language", 5).await.unwrap();
///     println!("{} results", results.len());
/// }
/// ```
/// use lmm::traits::Simulatable;
#[derive(Default)]
pub struct SearchAggregator {
    browser: Browser,
    /// BCP-47 / DDG region code, e.g. `"wt-wt"` (world-wide) or `"us-en"`.
    pub region: String,
}

impl SearchAggregator {
    /// Creates a new [`SearchAggregator`] with the world-wide region `"wt-wt"`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::net::SearchAggregator;
    /// let agg = SearchAggregator::new();
    /// assert_eq!(agg.region, "wt-wt");
    /// ```
    pub fn new() -> Self {
        Self {
            browser: Browser::new(),
            region: "wt-wt".to_string(),
        }
    }

    /// Sets the DuckDuckGo region code.
    ///
    /// # Arguments
    ///
    /// * `region` - A BCP-47 or DDG region string (e.g. `"us-en"`, `"de-de"`).
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::net::SearchAggregator;
    /// let agg = SearchAggregator::new().with_region("us-en");
    /// assert_eq!(agg.region, "us-en");
    /// ```
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    /// Executes a search and displays the results to stdout.
    ///
    /// This is a thin wrapper around the `duckduckgo` crate's Detailed mode.
    pub async fn search_and_display(&self, query: &str, limit: usize) -> Result<()> {
        self.browser
            .search(query, false, ResultFormat::Detailed, Some(limit), None)
            .await
    }

    /// Fetches `limit` lite search results for `query`.
    ///
    /// # Returns
    ///
    /// (`Result<Vec<LiteSearchResult>>`): A list of title + snippet pairs.
    pub async fn fetch(&self, query: &str, limit: usize) -> Result<Vec<LiteSearchResult>> {
        let ua = agent("firefox").unwrap_or("Mozilla/5.0");
        self.browser
            .lite_search(query, &self.region, Some(limit), ua)
            .await
    }

    /// Fetches the DuckDuckGo Instant Answer API response for `query`.
    ///
    /// The query is percent-encoded before being appended to the URL path so that
    /// special characters (spaces, `=`, `+`, Unicode) are correctly transmitted.
    ///
    /// # Returns
    ///
    /// (`Result<Response>`): The parsed API response struct.
    pub async fn get_response(&self, query: &str) -> Result<Response> {
        self.browser
            .get_api_response(&format!("?q={}", url_encode(query)), None)
            .await
    }
}

/// Appends a `.` to `text` if it doesn't already end with sentence-final punctuation.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::net::ensure_terminal_punct;
///
/// assert_eq!(ensure_terminal_punct("hello"), "hello.");
/// assert_eq!(ensure_terminal_punct("hello."), "hello.");
/// assert_eq!(ensure_terminal_punct("really?"), "really?");
/// ```
pub fn ensure_terminal_punct(text: &str) -> String {
    let t = text.trim();
    if t.ends_with('.') || t.ends_with('!') || t.ends_with('?') {
        t.to_string()
    } else {
        format!("{t}.")
    }
}

/// Collapses internal `"__###newline###__"` placeholders and normalises whitespace.
fn sanitize(text: &str) -> String {
    text.replace("__###newline###__", " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Heuristically determines whether `text` is a category label rather than a sentence.
///
/// Returns `true` when the text is very short (<5 words) or lacks common finite verbs.
fn is_category_label(text: &str) -> bool {
    let lower = text.to_lowercase();
    let word_count = text.split_whitespace().count();
    if word_count < 5 {
        return true;
    }
    let verb_indicators = [
        " is ",
        " are ",
        " was ",
        " were ",
        " has ",
        " have ",
        " can ",
        " will ",
        " does ",
        " do ",
        " provides ",
        " supports ",
        " describes ",
        " represents ",
        " enables ",
        " includes ",
        " spans ",
        " emphasizing ",
        " provided ",
    ];
    let has_verb = verb_indicators.iter().any(|&v| lower.contains(v));
    if !has_verb {
        return true;
    }
    let category_patterns = [
        "programming languages",
        "software using",
        "free software",
        "license",
        "category",
    ];
    category_patterns.iter().any(|&p| lower.contains(p))
}

/// Strips a `"Topic - "` prefix when the remainder has at least 5 words.
fn strip_topic_prefix(text: &str) -> String {
    if let Some(dash_pos) = text.find(" - ") {
        let after = text[dash_pos + 3..].trim();
        if after.split_whitespace().count() >= 5 {
            return after.to_string();
        }
    }
    text.to_string()
}

/// Builds a clean text corpus from lite search results for use as a generation seed.
///
/// Includes titles (≥ 3 words, no `|`) and snippets (≥ 7 words, no `|`).
///
/// # Arguments
///
/// * `results` - Slice of [`LiteSearchResult`] values.
///
/// # Returns
///
/// (`String`): Space-joined sentences with terminal punctuation.
pub fn corpus_from_results(results: &[LiteSearchResult]) -> String {
    results
        .iter()
        .filter_map(|r| {
            let mut parts: Vec<String> = Vec::new();
            let title = r.title.trim();
            if !title.is_empty() && !title.contains('|') && title.split_whitespace().count() >= 3 {
                parts.push(ensure_terminal_punct(title));
            }
            let snippet = r.snippet.trim();
            if !snippet.is_empty()
                && !snippet.contains('|')
                && snippet.split_whitespace().count() >= 7
            {
                parts.push(ensure_terminal_punct(snippet));
            }
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(" "))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Builds a raw corpus from lite search results, including only snippets.
///
/// Falls back to the title when the snippet is empty.
pub fn corpus_from_results_raw(results: &[LiteSearchResult]) -> String {
    results
        .iter()
        .filter_map(|r| {
            let mut parts: Vec<String> = Vec::new();
            let snippet = r.snippet.trim();
            if !snippet.is_empty() && !snippet.contains('|') {
                parts.push(ensure_terminal_punct(snippet));
            }
            let title = r.title.trim();
            if !title.is_empty() && !title.contains('|') && parts.is_empty() {
                parts.push(ensure_terminal_punct(title));
            }
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(" "))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Builds a corpus from a DuckDuckGo Instant Answer [`Response`].
///
/// Includes the abstract, answer, definition, and up to 15 related-topic sentences.
pub fn corpus_from_response(resp: &Response) -> String {
    let mut parts: Vec<String> = Vec::new();

    for text in [&resp.abstract_text, &resp.answer, &resp.definition]
        .iter()
        .filter_map(|o| o.as_ref())
    {
        let t = sanitize(text);
        if !t.is_empty() {
            parts.push(ensure_terminal_punct(&t));
        }
    }

    for topic in resp.related_topics.iter().take(15) {
        if let Some(raw_text) = &topic.text {
            let cleaned = strip_topic_prefix(&sanitize(raw_text));
            if !is_category_label(&cleaned) {
                parts.push(ensure_terminal_punct(&cleaned));
            }
        }
    }

    parts.join(" ")
}

/// Extracts topic-representative words from search results and appends them to `query`.
///
/// Stop words and short words (≤ 3 chars) are filtered. Up to 6 topic words are kept.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::net::{seed_from_results, LiteSearchResult};
/// let results: Vec<LiteSearchResult> = vec![];
/// assert_eq!(seed_from_results("my query", &results), "my query");
/// ```
pub fn seed_from_results(query: &str, results: &[LiteSearchResult]) -> String {
    const STOPWORDS: &[&str] = &[
        "the", "and", "for", "with", "that", "this", "from", "what", "how", "are", "was", "were",
        "will", "have", "been", "they",
    ];
    let topic_words: Vec<String> = results
        .iter()
        .flat_map(|r| r.title.split_whitespace().map(str::to_string))
        .filter(|w| {
            let low = w.to_lowercase();
            w.len() > 3 && !STOPWORDS.contains(&low.as_str())
        })
        .take(6)
        .collect();

    if topic_words.is_empty() {
        return query.to_string();
    }
    format!("{} {}", query, topic_words.join(" "))
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
