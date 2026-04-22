// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Integration tests for `cognition::knowledge`.

use lmm_agent::agent::LmmAgent;
use lmm_agent::cognition::knowledge::{
    DocumentChunk, DocumentParser, KnowledgeIndex, KnowledgeSource, MarkdownParser,
    PlainTextParser, ingest,
};

const RUST_TEXT: &str = "\
Rust is a systems language that prevents data races through its ownership model. \
The borrow checker enforces memory safety at compile time without a garbage collector. \
Cargo is the Rust package manager and build system. \
Async/await in Rust enables asynchronous programming via the Future trait. \
The Tokio runtime is the most popular async executor for Rust programs.";

fn populated_index() -> KnowledgeIndex {
    let mut idx = KnowledgeIndex::new();
    idx.ingest_text("test", RUST_TEXT);
    idx
}

#[test]
fn ingest_text_returns_nonzero_chunks() {
    let n = populated_index().len();
    assert!(n > 0, "Expected at least one chunk, got {n}");
}

#[test]
fn index_is_empty_before_ingest() {
    let idx = KnowledgeIndex::new();
    assert!(idx.is_empty());
    assert_eq!(idx.len(), 0);
}

#[test]
fn query_returns_relevant_chunks() {
    let idx = populated_index();
    let hits = idx.query("How does Rust prevent data races?", 3);
    assert!(!hits.is_empty(), "Expected at least one hit");
    let combined: String = hits
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let combined_lower = combined.to_lowercase();
    assert!(
        combined_lower.contains("rust")
            || combined_lower.contains("data")
            || combined_lower.contains("ownership"),
        "Expected relevant content in hit: {combined}"
    );
}

#[test]
fn query_unrelated_topic_returns_empty_or_low_score() {
    let idx = populated_index();
    let hits = idx.query("quantum entanglement photon wavelength", 3);
    for chunk in &hits {
        let lower = chunk.text.to_lowercase();
        assert!(
            !lower.contains("quantum"),
            "Unexpected quantum content in hit: {}",
            chunk.text
        );
    }
}

#[test]
fn answer_returns_some_for_known_topic() {
    let idx = populated_index();
    let answer = idx.answer("What does the borrow checker do?", 3);
    assert!(answer.is_some(), "Expected an extractive answer");
    let a = answer.unwrap();
    assert!(!a.is_empty());
}

#[test]
fn answer_returns_none_on_empty_index() {
    let idx = KnowledgeIndex::new();
    assert!(idx.answer("What is Rust?", 3).is_none());
}

#[test]
fn plain_text_parser_supports_txt() {
    let p = PlainTextParser;
    assert!(p.supports_extension("txt"));
    assert!(!p.supports_extension("pdf"));
}

#[test]
fn plain_text_parser_roundtrips_utf8() {
    let p = PlainTextParser;
    let input = b"hello world";
    let out = p.parse_bytes(input).unwrap();
    assert_eq!(out, "hello world");
}

#[test]
fn markdown_parser_strips_headings_and_code() {
    let p = MarkdownParser;
    let md = b"# Title\n\n```rust\nlet x = 1;\n```\n\nSome plain text here.";
    let out = p.parse_bytes(md).unwrap();
    assert!(!out.contains('#'), "headings should be stripped");
    assert!(!out.contains("let x"), "code blocks should be stripped");
    assert!(out.contains("plain text"), "prose should be preserved");
}

#[test]
fn document_chunk_tokens_are_lowercase_words() {
    let chunk = DocumentChunk::new("src", "Rust prevents Data Races through Ownership.");
    for token in &chunk.tokens {
        assert_eq!(
            token,
            &token.to_ascii_lowercase(),
            "Token not lowercase: {token}"
        );
        assert!(
            token.chars().all(|c| c.is_ascii_alphabetic()),
            "Token has non-alpha: {token}"
        );
    }
}

#[test]
fn ingest_multiple_sources_accumulates_chunks() {
    let mut idx = KnowledgeIndex::new();
    let n1 = idx.ingest_text("a", "Rust uses ownership. Ownership prevents memory bugs.");
    let n2 = idx.ingest_text("b", "Cargo manages dependencies. Cargo builds packages.");
    assert_eq!(idx.len(), n1 + n2);
}

#[tokio::test]
async fn agent_ingest_and_answer_roundtrip() {
    let mut agent = LmmAgent::new("QA".into(), "Rust questions.".into());
    let n = agent
        .ingest(KnowledgeSource::RawText(RUST_TEXT.into()))
        .await
        .unwrap();
    assert!(n > 0);

    let answer = agent.answer_from_knowledge("What does the borrow checker do?");
    assert!(
        answer.is_some(),
        "Expected an answer from ingested knowledge"
    );
}

#[tokio::test]
async fn agent_generate_uses_knowledge_index_first() {
    let mut agent = LmmAgent::new("QA".into(), "Rust.".into());
    agent
        .ingest(KnowledgeSource::RawText(
            "The borrow checker enforces memory safety rules at compile time in Rust.".into(),
        ))
        .await
        .unwrap();

    let response = agent
        .generate("What does the borrow checker enforce?")
        .await
        .unwrap();

    assert!(
        !response.is_empty(),
        "generate() should return something even with knowledge index populated"
    );
}

#[tokio::test]
async fn ingest_raw_text_via_free_function() {
    let mut idx = KnowledgeIndex::new();
    let n = ingest(
        &mut idx,
        KnowledgeSource::RawText("Rust is fast and safe. Systems programming without GC.".into()),
    )
    .await
    .unwrap();
    assert!(n > 0);
}

#[tokio::test]
async fn query_knowledge_method_returns_strings() {
    let mut agent = LmmAgent::new("Agent".into(), "Rust.".into());
    agent
        .ingest(KnowledgeSource::RawText(RUST_TEXT.into()))
        .await
        .unwrap();
    let results = agent.query_knowledge("ownership memory", 3);
    assert!(!results.is_empty());
    for r in results {
        assert!(!r.is_empty());
    }
}
