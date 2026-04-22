// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Knowledge Agent Example
//!
//! Demonstrates the full **knowledge acquisition lifecycle**:
//!
//! ```text
//! Source (File / URL / RawText)
//!   → ingest()         (parse + sentence-chunk + index)
//!   → query_knowledge() (IDF-weighted retrieval)
//!   → answer_from_knowledge() (TextSummarizer extractive answer)
//!   → generate()       (returns grounded text if index has relevant material)
//! ```
//!
//! This example runs entirely offline with no network access and no LLM API.
//! It uses built-in `KnowledgeSource::RawText` so it works out-of-the box.
//!
//! To ingest a real file use:
//! ```bash
//! # plain-text file
//! cargo run --example knowledge_agent
//!
//! # PDF requires the knowledge feature
//! cargo run --example knowledge_agent --features knowledge
//!
//! # URL ingestion also requires the net feature
//! cargo run --example knowledge_agent --features net
//! ```

use lmm_agent::prelude::*;

const CORPUS: &str = "\
Rust is a systems programming language focused on three goals: safety, speed, \
and concurrency. It accomplishes these goals without a garbage collector, making \
it useful for a number of use cases other languages are not good at: embedding in \
other languages, programs with specific space and time requirements, and writing \
low-level code like device drivers and operating systems. Rust improves on current \
languages targeting this space by having a number of compile-time safety checks with \
no runtime overhead, while eliminating all data races.

The ownership model is Rust's most unique and compelling feature. It gives Rust the \
ability to make memory safety guarantees without needing a garbage collector. Each \
value in Rust has a variable that's called its owner. There can only be one owner at \
a time. When the owner goes out of scope, the value will be dropped automatically.

The borrow checker enforces these ownership rules at compile time. References allow \
you to refer to some value without taking ownership of it. Unlike a pointer, a \
reference is guaranteed to point to a valid value for the life of that reference. \
Rust uses lifetimes to track how long references are valid.

Cargo is Rust's build system and package manager. Most Rustaceans use Cargo to manage \
their Rust projects because Cargo handles many tasks: building code, downloading \
dependencies and building them. Cargo comes installed with Rust.

Async/await in Rust enables writing asynchronous code that looks like synchronous code. \
The async keyword turns a function into one that returns a Future. Calling .await on a \
Future suspends execution until the future is ready. Tokio is the most popular async \
runtime for Rust, providing the executor, timers, I/O primitives, and task scheduling.";

#[derive(Debug, Default, Auto)]
pub struct KnowledgeAgent {
    pub agent: LmmAgent,
}

#[async_trait]
impl Executor for KnowledgeAgent {
    async fn execute<'a>(
        &'a mut self,
        _task: &'a mut Task,
        _execute: bool,
        _browse: bool,
        _max_tries: u64,
    ) -> Result<()> {
        println!("Persona   : {}", self.agent.persona);
        println!("Status    : {} → Active", self.agent.status);
        self.agent.update(Status::Active);

        let n = self
            .agent
            .ingest(KnowledgeSource::RawText(CORPUS.into()))
            .await?;
        println!("Indexed   : {n} knowledge chunks");
        println!(
            "Index size: {} chunks total\n",
            self.agent.knowledge_index.len()
        );

        let questions = [
            "What is Rust's ownership model?",
            "How does the borrow checker work?",
            "What is Cargo used for?",
            "How does async/await work in Rust?",
        ];

        for question in &questions {
            println!("Q: {question}");
            let answer = self
                .generate(question)
                .await
                .unwrap_or_else(|_| "[no answer]".into());
            println!("A: {answer}\n");
        }

        if let Some(reflection) = &self.agent.reflection {
            println!("Reflection:\n{}", (reflection.evaluation_fn)(&self.agent));
        }

        self.agent.update(Status::Completed);
        println!("Status: {}", self.agent.status);
        Ok(())
    }
}

#[tokio::main]
async fn main() {
    let agent = KnowledgeAgent::new(
        "Knowledge Agent".into(),
        "Answer questions about Rust programming.".into(),
    );

    match AutoAgent::default()
        .with(agents![agent])
        .max_tries(1)
        .build()
        .expect("Failed to build AutoAgent")
        .run()
        .await
    {
        Ok(msg) => println!("\n[AutoAgent] {msg}"),
        Err(err) => eprintln!("\n[AutoAgent] Error: {err:?}"),
    }
}
