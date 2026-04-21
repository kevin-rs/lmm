// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `SearchOracle` - DuckDuckGo query planner with in-process cache.
//!
//! Transforms a query string into a raw-text observation that the controller
//! feeds back as the error measurement.
//!
//! * **In-process cache** - duplicate queries never make a second network request.
//! * **Graceful offline degradation** - when `net` feature is disabled, `fetch`
//!   returns an empty string immediately.
//!
//! ## Examples
//!
//! ```rust
//! #[tokio::main]
//! async fn main() {
//!     use lmm_agent::cognition::search::SearchOracle;
//!     let mut oracle = SearchOracle::new(5);
//!     let result = oracle.fetch("Rust memory safety").await;
//!     println!("{result}");
//! }
//! ```
//!
//! ## See Also
//!
//! * [Information retrieval - Wikipedia](https://en.wikipedia.org/wiki/Information_retrieval)
//! * [DuckDuckGo - Wikipedia](https://en.wikipedia.org/wiki/DuckDuckGo)

use std::collections::HashMap;

/// DuckDuckGo search client with an in-process query cache.
///
/// # Examples
///
/// ```rust
/// #[tokio::main]
/// async fn main() {
///     use lmm_agent::cognition::search::SearchOracle;
///     let mut oracle = SearchOracle::new(3);
///     let obs = oracle.fetch("Rust ownership").await;
///     println!("{obs}");
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SearchOracle {
    /// In-process cache: query → observation corpus.
    cache: HashMap<String, String>,
    /// Maximum number of DDG result snippets to concatenate.
    pub limit: usize,
}

impl SearchOracle {
    /// Constructs a new `SearchOracle`.
    ///
    /// * `limit` - number of DDG result snippets to concatenate (≥ 1).
    pub fn new(limit: usize) -> Self {
        Self {
            cache: HashMap::new(),
            limit: limit.max(1),
        }
    }

    /// Fetches an observation for `query`, returning a cached result when available.
    ///
    /// When the `net` feature is disabled this always returns `String::new()`.
    /// Network errors are silently discarded so the ThinkLoop continues safely.
    pub async fn fetch(&mut self, query: &str) -> String {
        if let Some(cached) = self.cache.get(query) {
            return cached.clone();
        }
        let observation = self.fetch_live(query).await;
        self.cache.insert(query.to_string(), observation.clone());
        observation
    }

    /// Returns the number of cached queries.
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Clears the in-process cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Issues the actual DuckDuckGo request (only compiled when `net` is on).
    #[cfg(feature = "net")]
    async fn fetch_live(&self, query: &str) -> String {
        use duckduckgo::browser::Browser;
        use duckduckgo::user_agents::get as get_ua;

        let browser = Browser::new();
        let ua = get_ua("firefox").unwrap_or("Mozilla/5.0");
        match browser
            .lite_search(query, "wt-wt", Some(self.limit), ua)
            .await
        {
            Ok(results) => results
                .iter()
                .filter_map(|r| {
                    let snippet = r.snippet.trim();
                    if !snippet.is_empty() {
                        Some(snippet.to_string())
                    } else if !r.title.trim().is_empty() {
                        Some(r.title.trim().to_string())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(" "),
            Err(_) => String::new(),
        }
    }

    #[cfg(not(feature = "net"))]
    async fn fetch_live(&self, _query: &str) -> String {
        String::new()
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
