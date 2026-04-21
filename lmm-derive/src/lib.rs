// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `lmm-derive`
//!
//! Proc-macro crates for the `lmm` agent framework.
//!
//! ## Derives
//!
//! - [`Auto`] - Generates `impl Agent`, `impl Functions`, and
//!   `#[async_trait] impl AsyncFunctions` for any struct that contains an
//!   `agent: LmmAgent` field. The struct must also `impl Executor`.
//!
//! ## Macros
//!
//! See the `agents!` macro re-exported from `lmm_agent`.
//!
//! ## Attribution
//!
//! The `Auto` derive is adapted from the `autogpt` crate's `auto-derive`:
//! <https://github.com/wiseaidotdev/autogpt/blob/main/auto-derive/src/lib.rs>

extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

/// Derives `Agent`, `Functions`, and `AsyncFunctions` for a struct that:
///
/// - Contains a field named `agent` of type `LmmAgent`.
/// - Implements the `Executor` trait.
///
/// This macro is adapted from the `autogpt` project's `auto-derive` crate:
/// <https://github.com/wiseaidotdev/autogpt/blob/main/auto-derive/src/lib.rs>
///
/// # Example
///
/// ```rust,ignore
/// extern crate lmm_agent;
/// use lmm_derive::{Auto, Executor};
/// use lmm_agent::prelude::*;
/// use lmm_agent::types::{Task, Status, Message};
/// use lmm_agent::agent::LmmAgent;
/// use std::borrow::Cow;
/// use async_trait::async_trait;
/// use anyhow::Result;
///
/// #[derive(Debug, Default, Auto)]
/// pub struct MyAgent {
///     pub persona:  Cow<'static, str>,
///     pub behavior: Cow<'static, str>,
///     pub status:   Status,
///     pub agent:    LmmAgent,
///     pub memory:   Vec<Message>,
/// }
///
/// #[async_trait]
/// impl Executor for MyAgent {
///     async fn execute<'a>(
///         &'a mut self,
///         tasks: &'a mut Task,
///         _execute: bool,
///         _browse: bool,
///         _max_tries: u64,
///     ) -> Result<()> {
///         // Custom logic here...
///         Ok(())
///     }
/// }
/// ```
#[proc_macro_derive(Auto)]
pub fn derive_auto(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let expanded = quote! {
        impl ::lmm_agent::traits::agent::Agent for #name {
            fn new(
                persona: ::std::borrow::Cow<'static, str>,
                behavior: ::std::borrow::Cow<'static, str>,
            ) -> Self {
                let mut agent = Self::default();
                agent.agent = ::lmm_agent::agent::LmmAgent::new(persona, behavior.clone());
                agent.persona = agent.agent.persona.clone().into();
                agent.behavior = agent.agent.behavior.clone().into();
                agent
            }

            fn update(&mut self, status: ::lmm_agent::types::Status) {
                self.agent.update(status);
            }

            fn persona(&self) -> &str {
                &self.agent.persona
            }

            fn behavior(&self) -> &str {
                &self.agent.behavior
            }

            fn status(&self) -> &::lmm_agent::types::Status {
                &self.agent.status
            }

            fn memory(&self) -> &::std::vec::Vec<::lmm_agent::types::Message> {
                &self.agent.memory
            }

            fn tools(&self) -> &::std::vec::Vec<::lmm_agent::types::Tool> {
                &self.agent.tools
            }

            fn knowledge(&self) -> &::lmm_agent::types::Knowledge {
                &self.agent.knowledge
            }

            fn planner(&self) -> ::std::option::Option<&::lmm_agent::types::Planner> {
                self.agent.planner.as_ref()
            }

            fn profile(&self) -> &::lmm_agent::types::Profile {
                &self.agent.profile
            }

            fn reflection(&self) -> ::std::option::Option<&::lmm_agent::types::Reflection> {
                self.agent.reflection.as_ref()
            }

            fn scheduler(&self) -> ::std::option::Option<&::lmm_agent::types::TaskScheduler> {
                self.agent.scheduler.as_ref()
            }

            fn capabilities(&self) -> &::std::collections::HashSet<::lmm_agent::types::Capability> {
                &self.agent.capabilities
            }

            fn context(&self) -> &::lmm_agent::types::ContextManager {
                &self.agent.context
            }

            fn tasks(&self) -> &::std::vec::Vec<::lmm_agent::types::Task> {
                &self.agent.tasks
            }

            fn memory_mut(&mut self) -> &mut ::std::vec::Vec<::lmm_agent::types::Message> {
                &mut self.agent.memory
            }

            fn planner_mut(&mut self) -> ::std::option::Option<&mut ::lmm_agent::types::Planner> {
                self.agent.planner.as_mut()
            }

            fn context_mut(&mut self) -> &mut ::lmm_agent::types::ContextManager {
                &mut self.agent.context
            }
        }

        impl ::lmm_agent::traits::functions::Functions for #name {
            fn get_agent(&self) -> &::lmm_agent::agent::LmmAgent {
                &self.agent
            }
        }

        #[async_trait]
        impl ::lmm_agent::traits::functions::AsyncFunctions for #name {
            async fn execute<'a>(
                &'a mut self,
                tasks: &'a mut Task,
                execute: bool,
                browse: bool,
                max_tries: u64,
            ) -> Result<()> {
                <#name as ::lmm_agent::traits::functions::Executor>::execute(
                    self, tasks, execute, browse, max_tries,
                )
                .await
            }

            async fn save_ltm(
                &mut self,
                communication: Message,
            ) -> Result<()> {
                self.agent.long_term_memory.push(communication);
                Ok(())
            }

            async fn get_ltm(&self) -> Result<Vec<Message>> {
                Ok(self.agent.long_term_memory.clone())
            }

            async fn ltm_context(&self) -> String {
                self.agent
                    .long_term_memory
                    .iter()
                    .map(|c| format!("{}: {}", c.role, c.content))
                    .collect::<Vec<_>>()
                    .join("\n")
            }

            async fn generate(&mut self, request: &str) -> Result<String> {
                self.agent.generate(request).await
            }

            async fn search(
                &self,
                query: &str,
                limit: usize,
            ) -> Result<String> {
                self.agent.search(query, limit).await
            }
        }
    };

    TokenStream::from(expanded)
}
