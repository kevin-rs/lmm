use crate::types::SearchLink;
use lmm::net::{
    SearchAggregator, corpus_from_response, corpus_from_results, corpus_from_results_raw,
};
use lmm::predict::TextPredictor;
use lmm::stochastic::StochasticEnhancer;
use lmm::text::{EssayGenerator, ParagraphGenerator, SentenceGenerator, TextSummarizer};

fn enhance(text: &str, stochastic: bool, prob: f64) -> String {
    if stochastic {
        StochasticEnhancer::new(prob).enhance(text)
    } else {
        text.to_string()
    }
}

pub fn generate_sentence(seed: &str, stochastic: bool, prob: f64) -> String {
    match SentenceGenerator::new(50, 5).generate(seed) {
        Ok(s) => enhance(&s, stochastic, prob),
        Err(e) => format!("Generation error: {e}"),
    }
}

pub fn generate_paragraph(seed: &str, sentences: usize, stochastic: bool, prob: f64) -> String {
    match ParagraphGenerator::new(sentences.max(2), 50, 5).generate(seed) {
        Ok(p) => enhance(&p, stochastic, prob),
        Err(e) => format!("Generation error: {e}"),
    }
}

pub fn generate_essay(
    topic: &str,
    paragraphs: usize,
    sentences: usize,
    stochastic: bool,
    prob: f64,
) -> String {
    match EssayGenerator::new(paragraphs.max(2), sentences.max(2), 50, 5).generate(topic) {
        Ok(essay) => {
            let mut out = format!("# {}\n\n", essay.title);
            for para in &essay.paragraphs {
                out.push_str(&enhance(para, stochastic, prob));
                out.push_str("\n\n");
            }
            out.trim_end().to_string()
        }
        Err(e) => format!("Generation error: {e}"),
    }
}

pub fn summarize(text: &str, count: usize, stochastic: bool, prob: f64) -> String {
    match TextSummarizer::new(count.max(1), 50, 5).summarize(text) {
        Ok(sentences) => sentences
            .iter()
            .map(|s| enhance(s, stochastic, prob))
            .collect::<Vec<_>>()
            .join(" "),
        Err(_) => {
            "Input needs complete sentences (ending with . ! ?) of at least 5 words.".to_string()
        }
    }
}

pub fn predict(text: &str, length: usize, stochastic: bool, prob: f64) -> String {
    match TextPredictor::new(10, 50, 5).predict_continuation(text, length) {
        Ok(result) => {
            let cont = enhance(&result.continuation, stochastic, prob);
            format!("{}{}", text, cont)
        }
        Err(_) => "Prediction requires at least 2 words of input.".to_string(),
    }
}

pub async fn ask(
    prompt: &str,
    sentences: usize,
    stochastic: bool,
    prob: f64,
) -> (String, Vec<SearchLink>) {
    let aggregator = SearchAggregator::new();
    let api_result = aggregator.get_response(prompt).await;

    let corpus = api_result
        .as_ref()
        .map(corpus_from_response)
        .unwrap_or_default();

    let mut links: Vec<SearchLink> = api_result
        .as_ref()
        .map(|resp| {
            resp.related_topics
                .iter()
                .filter_map(|t| {
                    let url = t.first_url.clone()?;
                    if url.is_empty() {
                        return None;
                    }
                    let text = t.text.clone().unwrap_or_default();
                    let title: String = text
                        .split_whitespace()
                        .take(6)
                        .collect::<Vec<_>>()
                        .join(" ");
                    let title = if title.is_empty() {
                        url.clone()
                    } else {
                        format!("{}...", title)
                    };
                    Some(SearchLink { title, url })
                })
                .take(10)
                .collect()
        })
        .unwrap_or_default();

    let final_corpus = if corpus.trim().is_empty() {
        #[cfg(not(target_arch = "wasm32"))]
        {
            match aggregator.fetch(prompt, 6).await {
                Ok(results) => {
                    if links.is_empty() {
                        links = results
                            .iter()
                            .filter(|r| !r.url.is_empty())
                            .map(|r| SearchLink {
                                title: r.title.clone(),
                                url: r.url.clone(),
                            })
                            .take(10)
                            .collect();
                    }
                    let quality = corpus_from_results(&results);
                    if quality.trim().is_empty() {
                        corpus_from_results_raw(&results)
                    } else {
                        quality
                    }
                }
                Err(_) => String::new(),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            String::new()
        }
    } else {
        corpus
    };

    if final_corpus.trim().is_empty() {
        return (
            "The DuckDuckGo Instant Answer API returned no content for this query. \
             Try a more specific well-known topic."
                .to_string(),
            links,
        );
    }

    let summary = match TextSummarizer::new(sentences.max(2), 50, 5)
        .summarize_with_query(&final_corpus, prompt)
    {
        Ok(results) => results
            .iter()
            .map(|s| enhance(s, stochastic, prob))
            .collect::<Vec<_>>()
            .join(" "),
        Err(_) => "Could not summarize search results.".to_string(),
    };

    (summary, links)
}
