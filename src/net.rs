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

pub fn corpus_from_results(results: &[LiteSearchResult]) -> String {
    results
        .iter()
        .map(|r| {
            let mut parts: Vec<String> = Vec::new();
            if !r.title.is_empty() {
                let mut title = r.title.trim().to_string();
                if !title.ends_with('.') && !title.ends_with('!') && !title.ends_with('?') {
                    title.push('.');
                }
                parts.push(title);
            }
            if !r.snippet.is_empty() {
                let mut snippet = r.snippet.trim().to_string();
                if !snippet.ends_with('.') && !snippet.ends_with('!') && !snippet.ends_with('?') {
                    snippet.push('.');
                }
                parts.push(snippet);
            }
            parts.join(" ")
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn corpus_from_response(resp: &Response) -> String {
    let mut parts = Vec::new();

    let mut add_part = |text: &str| {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            let mut content = trimmed.to_string();
            if !content.ends_with('.') && !content.ends_with('!') && !content.ends_with('?') {
                content.push('.');
            }
            parts.push(content);
        }
    };

    if let Some(abstract_text) = &resp.abstract_text {
        add_part(abstract_text);
    }
    if let Some(answer) = &resp.answer {
        add_part(answer);
    }
    if let Some(definition) = &resp.definition {
        add_part(definition);
    }

    for topic in resp.related_topics.iter().take(10) {
        if let Some(text) = &topic.text
            && text.split_whitespace().count() >= 5
        {
            add_part(text);
        }
    }

    parts.join(" ")
}

pub fn seed_from_results(query: &str, results: &[LiteSearchResult]) -> String {
    let topic_words: Vec<String> = results
        .iter()
        .flat_map(|r| r.title.split_whitespace().map(str::to_string))
        .filter(|w| {
            let low = w.to_lowercase();
            w.len() > 3
                && !matches!(
                    low.as_str(),
                    "the"
                        | "and"
                        | "for"
                        | "with"
                        | "that"
                        | "this"
                        | "from"
                        | "what"
                        | "how"
                        | "are"
                        | "was"
                        | "were"
                        | "will"
                )
        })
        .take(6)
        .collect();

    if topic_words.is_empty() {
        return query.to_string();
    }
    format!("{} {}", query, topic_words.join(" "))
}
