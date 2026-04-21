# LMM Derive Macros ⚙️

The `lmm-derive` crate provides procedural macros that eliminate agent boilerplate. The primary export is `#[derive(Auto)]`.

## 📦 Installation

Pulled in automatically with `lmm-agent`. No manual dependency needed.

## 🚀 The `Auto` Macro

### Required struct shape

```rust
use lmm_agent::prelude::*;
use lmm_agent::types::{Message, Status, Task};
use lmm_agent::agent::LmmAgent;
use std::borrow::Cow;
use async_trait::async_trait;
use anyhow::Result;

#[derive(Debug, Default, Auto)]
pub struct MyAgent {
    pub persona:  Cow<'static, str>,
    pub behavior: Cow<'static, str>,
    pub status:   Status,
    pub agent:    LmmAgent,
    pub memory:   Vec<Message>,
}

#[async_trait]
impl Executor for MyAgent {
    async fn execute<'a>(
        &'a mut self, _tasks: &'a mut Task,
        _execute: bool, _browse: bool, _max_tries: u64,
    ) -> Result<()> { Ok(()) }
}
```

`Auto` inspects the five required fields above and generates three trait implementations automatically.

### Generated traits

#### `Agent`

Delegates all methods to the inner `LmmAgent` field:

```rust,ignore
impl Agent for MyAgent {
    fn new(persona: Cow<'static, str>, behavior: Cow<'static, str>) -> Self {
         unimplemented!()
    }
    fn persona(&self)     -> &str          { &self.agent.persona }
    fn behavior(&self)    -> &str          { &self.agent.behavior }
    fn status(&self)      -> &Status       { &self.agent.status }
    fn memory(&self)      -> &Vec<Message> { &self.agent.memory }
    fn memory_mut(&mut self) -> &mut Vec<Message> { &mut self.agent.memory }
}
```

#### `Functions`

```rust,ignore
impl Functions for MyAgent {
    fn get_agent(&self)         -> &LmmAgent     { &self.agent }
    fn get_agent_mut(&mut self) -> &mut LmmAgent { &mut self.agent }
}
```

#### `AsyncFunctions`

```rust,ignore
#[async_trait]
impl AsyncFunctions for MyAgent {
    async fn generate(&mut self, prompt: &str) -> Result<String>      { unimplemented!() }
    async fn search(&self, query: &str)         -> Result<Vec<LiteSearchResult>>   { unimplemented!() }
    async fn save_ltm(&mut self, msg: Message)  -> Result<()>         { unimplemented!() }
    async fn get_ltm(&self)                     -> Result<Vec<Message>>{ unimplemented!() }
    async fn ltm_context(&self)                 -> Result<String>     { unimplemented!() }
}
```

### Additional fields

Fields beyond the five required ones are ignored by `Auto`. You can freely add domain-specific data:

```rust
extern crate lmm_agent;
use lmm_agent::prelude::*;
use lmm_agent::types::{Message, Status, Task};
use lmm_agent::agent::LmmAgent;
use std::borrow::Cow;
use std::collections::HashMap;
use async_trait::async_trait;
use anyhow::Result;

#[derive(Debug, Default, Auto)]
pub struct DataAgent {
    pub persona:   Cow<'static, str>,
    pub behavior:  Cow<'static, str>,
    pub status:    Status,
    pub agent:     LmmAgent,
    pub memory:    Vec<Message>,
    // custom fields, ignored by the macro
    pub db_url:    String,
    pub cache:     HashMap<String, String>,
}

#[async_trait]
impl Executor for DataAgent {
    async fn execute<'a>(
        &'a mut self, _tasks: &'a mut Task,
        _execute: bool, _browse: bool, _max_tries: u64,
    ) -> Result<()> { Ok(()) }
}
```

## ⚠️ Field Name Contract

| Field      | Type                | Must be named      |
| ---------- | ------------------- | ------------------ |
| `persona`  | `Cow<'static, str>` | exactly `persona`  |
| `behavior` | `Cow<'static, str>` | exactly `behavior` |
| `status`   | `Status`            | exactly `status`   |
| `agent`    | `LmmAgent`          | exactly `agent`    |
| `memory`   | `Vec<Message>`      | exactly `memory`   |

Compile errors will occur if any required field is missing or misnamed.

## 📄 License

Licensed under the [MIT License](../LICENSE).
