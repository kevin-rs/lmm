// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `LearningStore` - persistent serialisation of HELM learning state.
//!
//! `LearningStore` serialises and deserialises the complete `LearningEngine`
//! state (Q-table, meta-prototypes, distiller fingerprints, elastic guard
//! activation counts, PMI pair counts) to/from a JSON file on disk.
//!
//! This provides **long-term persistence** across process restarts: an agent
//! that has been saved can be restored and continue learning from where it
//! left off, fulfilling the *smart lifelong learning* requirement.
//!
//! ## Examples
//!
//! ```rust
//! use lmm_agent::cognition::learning::store::LearningStore;
//! use lmm_agent::cognition::learning::engine::LearningEngine;
//! use lmm_agent::cognition::learning::config::LearningConfig;
//!
//! let engine = LearningEngine::new(LearningConfig::default());
//! let path = std::env::temp_dir().join(format!("test_helm_store_{}.json", uuid::Uuid::new_v4()));
//!
//! LearningStore::save(&engine, &path).unwrap();
//! let loaded = LearningStore::load(&path).unwrap();
//! assert_eq!(loaded.q_table().state_count(), engine.q_table().state_count());
//! ```

use crate::cognition::learning::engine::LearningEngine;
use anyhow::{Result, anyhow};
use std::fs;
use std::path::Path;

/// Round-trip serialisation helper for [`LearningEngine`].
pub struct LearningStore;

impl LearningStore {
    /// Serialises `engine` to a JSON file at `path`.
    ///
    /// Creates the file (and all parent directories) if they do not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if serialisation fails or the file cannot be created.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::store::LearningStore;
    /// use lmm_agent::cognition::learning::engine::LearningEngine;
    /// use lmm_agent::cognition::learning::config::LearningConfig;
    ///
    /// let engine = LearningEngine::new(LearningConfig::default());
    /// let path = std::env::temp_dir().join(format!("helm_doctest_{}.json", uuid::Uuid::new_v4()));
    /// LearningStore::save(&engine, &path).unwrap();
    /// ```
    pub fn save(engine: &LearningEngine, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)
                .map_err(|e| anyhow!("Cannot create dir {:?}: {e}", parent))?;
        }
        let json = serde_json::to_string_pretty(engine)
            .map_err(|e| anyhow!("Serialisation error: {e}"))?;
        fs::write(path, &json).map_err(|e| anyhow!("Cannot write {:?}: {e}", path))
    }

    /// Deserialises a [`LearningEngine`] from the JSON file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if the JSON is malformed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lmm_agent::cognition::learning::store::LearningStore;
    /// use lmm_agent::cognition::learning::engine::LearningEngine;
    /// use lmm_agent::cognition::learning::config::LearningConfig;
    ///
    /// let engine = LearningEngine::new(LearningConfig::default());
    /// let path = std::env::temp_dir().join(format!("helm_load_doctest_{}.json", uuid::Uuid::new_v4()));
    /// LearningStore::save(&engine, &path).unwrap();
    /// let loaded = LearningStore::load(&path).unwrap();
    /// assert_eq!(loaded.config().alpha, engine.config().alpha);
    /// ```
    pub fn load(path: &Path) -> Result<LearningEngine> {
        let json = fs::read_to_string(path).map_err(|e| anyhow!("Cannot read {:?}: {e}", path))?;
        serde_json::from_str(&json).map_err(|e| anyhow!("Deserialisation error in {:?}: {e}", path))
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
