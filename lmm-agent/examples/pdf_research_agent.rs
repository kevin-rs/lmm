// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # PDF Research Agent Example
//!
//! Demonstrates how an agent can ingest and query a local PDF document
//! using the **knowledge** feature.
//!
//! In this example, the agent ingests a paper from the `papers` directory
//! and answers specialized technical questions based on its content.
//!
//! To run this example:
//! ```bash
//! cargo run --example pdf_research_agent --features knowledge
//! ```

use lmm_agent::prelude::*;
use std::path::PathBuf;

#[derive(Debug, Default, Auto)]
pub struct PdfResearchAgent {
    pub agent: LmmAgent,
}

#[async_trait]
impl Executor for PdfResearchAgent {
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

        let paper_path = PathBuf::from("papers/lmm.pdf");

        if !paper_path.exists() {
            return Err(anyhow::anyhow!("Paper not found at {:?}", paper_path));
        }

        println!("Ingesting paper: {:?}", paper_path);
        let n = self.agent.ingest(KnowledgeSource::File(paper_path)).await?;
        println!("Indexed   : {n} chunks from PDF\n");

        let queries = [
            "What is the core architecture of LMM?",
            "How does the symbolic engine handle text generation?",
            "What are the main advantages of equation-based agents?",
        ];

        for query in &queries {
            println!("Query: {query}");
            if let Some(answer) = self.agent.answer_from_knowledge(query) {
                println!("Answer: {answer}\n");
            } else {
                let response = self.agent.generate(query).await?;
                println!("Response: {response}\n");
            }
        }

        self.agent.update(Status::Completed);
        println!("Status    : {}", self.agent.status);
        Ok(())
    }
}

#[tokio::main]
async fn main() {
    let agent = PdfResearchAgent::new(
        "PDF Researcher".into(),
        "Analyze technical papers and extract key insights.".into(),
    );

    match AutoAgent::default()
        .with(agents![agent])
        .max_tries(1)
        .build()
        .expect("Failed to build AutoAgent")
        .run()
        .await
    {
        Ok(msg) => println!("\n[AutoAgent] Success: {msg}"),
        Err(err) => eprintln!("\n[AutoAgent] Error: {err:?}"),
    }
}
