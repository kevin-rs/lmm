use anyhow::Result;
use duckduckgo::browser::Browser;
use duckduckgo::response::{LiteSearchResult, Response, ResultFormat};
use duckduckgo::user_agents::get as agent;

pub struct SearchAggregator {
    browser: Browser,
    pub region: String,
}

impl SearchAggregator {
    pub fn new() -> Self {
        Self {
            browser: Browser::new(),
            region: "wt-wt".to_string(),
        }
    }

    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    pub async fn search_and_display(&self, query: &str, limit: usize) -> Result<()> {
        self.browser
            .search(query, false, ResultFormat::Detailed, Some(limit), None)
            .await
    }

    pub async fn fetch(&self, query: &str, limit: usize) -> Result<Vec<LiteSearchResult>> {
        let ua = agent("firefox").unwrap_or("Mozilla/5.0");
        self.browser
            .lite_search(query, &self.region, Some(limit), ua)
            .await
    }

    pub async fn get_response(&self, query: &str) -> Result<Response> {
        self.browser
            .get_api_response(&format!("?q={}", query), None)
            .await
    }
}

impl Default for SearchAggregator {
    fn default() -> Self {
        Self::new()
    }
}

fn ensure_terminal_punct(text: &str) -> String {
    let t = text.trim();
    if t.ends_with('.') || t.ends_with('!') || t.ends_with('?') {
        t.to_string()
    } else {
        format!("{}.", t)
    }
}

fn sanitize(text: &str) -> String {
    text.replace("__###newline###__", " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

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

fn strip_topic_prefix(text: &str) -> String {
    if let Some(dash_pos) = text.find(" - ") {
        let after = text[dash_pos + 3..].trim();
        if after.split_whitespace().count() >= 5 {
            return after.to_string();
        }
    }
    text.to_string()
}

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

pub fn corpus_from_response(resp: &Response) -> String {
    let mut parts: Vec<String> = Vec::new();

    if let Some(abstract_text) = &resp.abstract_text {
        let t = sanitize(abstract_text);
        if !t.is_empty() {
            parts.push(ensure_terminal_punct(&t));
        }
    }

    if let Some(answer) = &resp.answer {
        let t = sanitize(answer);
        if !t.is_empty() {
            parts.push(ensure_terminal_punct(&t));
        }
    }

    if let Some(definition) = &resp.definition {
        let t = sanitize(definition);
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

pub fn seed_from_results(query: &str, results: &[LiteSearchResult]) -> String {
    let stopwords = [
        "the", "and", "for", "with", "that", "this", "from", "what", "how", "are", "was", "were",
        "will", "have", "been", "they",
    ];
    let topic_words: Vec<String> = results
        .iter()
        .flat_map(|r| r.title.split_whitespace().map(str::to_string))
        .filter(|w| {
            let low = w.to_lowercase();
            w.len() > 3 && !stopwords.contains(&low.as_str())
        })
        .take(6)
        .collect();

    if topic_words.is_empty() {
        return query.to_string();
    }
    format!("{} {}", query, topic_words.join(" "))
}
