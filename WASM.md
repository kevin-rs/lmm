# LMM WebAssembly Guide 🌐

LMM natively targets `wasm32-unknown-unknown`. Because the HTTP client (`reqwest`) automatically switches to the browser `fetch` API on WASM targets, you can deploy LMM inside Rust frontend frameworks without any additional glue code.

## Supported Frameworks

LMM WASM integration is actively used and tested with:

- **[Yew](https://yew.rs/)**: the most popular Rust / WASM frontend framework
- **[Dioxus](https://dioxuslabs.com/)**: cross-platform Rust UI framework
- **[Leptos](https://leptos.dev/)**: fine-grained reactive framework

## 📦 Adding to a WASM Project

Add `lmm` to your `Cargo.toml` with the appropriate feature set. Avoid enabling `rust-binary`, `python`, or `node` on WASM targets.

```toml
[dependencies]
lmm = { version = "0.1.15", default-features = false, features = ["wasm-net"] }
```

Build with Trunk (Yew / Leptos):

```sh
trunk serve
```

## 🌍 CORS Considerations

DuckDuckGo's endpoints set permissive CORS headers on most responses, but the behaviour can vary by endpoint and region. If you encounter CORS errors:

- Use a server-side proxy to forward requests from your own origin.
- Use the `wt-wt` (worldwide) region which tends to have the broadest CORS coverage.
- Consider caching results server-side and serving them from your own API.

## 🔬 Feature Flags for WASM

| Feature       | WASM-safe | Description                                     |
| ------------- | --------- | ----------------------------------------------- |
| (default)     | ✅        | Core symbolic engine (no networking)            |
| `wasm-net`    | ✅        | Enables `reqwest` fetch-based HTTP on WASM      |
| `net`         | ❌        | Native Tokio networking: incompatible with WASM |
| `rust-binary` | ❌        | CLI binary: incompatible with WASM              |
| `python`      | ❌        | pyo3 bindings: incompatible with WASM           |
| `node`        | ❌        | napi bindings: incompatible with WASM           |

## 📖 Example: Yew Component

```rust
use lmm::prelude::*;
use yew::prelude::*;

#[function_component(PredictorDemo)]
fn predictor_demo() -> Html {
    let output = use_state(|| String::from("Click to generate..."));
    let onclick = {
        let output = output.clone();
        Callback::from(move |_| {
            let predictor = TextPredictor::new(20, 30, 3);
            if let Ok(result) = predictor.predict_continuation("Wise AI built the first LMM", 80) {
                output.set(result.continuation);
            }
        })
    };
    html! {
        <div>
            <button {onclick}>{ "Generate" }</button>
            <p>{ (*output).clone() }</p>
        </div>
    }
}
```

## 📄 License

Licensed under the [MIT License](LICENSE).
