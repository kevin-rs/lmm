#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!("../README.md")]

pub mod causal;
#[cfg(feature = "cli")]
pub mod cli;
pub mod compression;
pub mod consciousness;
pub mod discovery;
pub mod encode;
pub mod equation;
pub mod error;
pub mod field;
pub mod lexicon;
pub mod models;
pub mod operator;
pub mod perception;
pub mod physics;
pub mod predict;
pub mod prelude;
pub mod simulation;
pub mod symbolic;
pub mod tensor;
pub mod text;
pub mod traits;
pub mod world;
