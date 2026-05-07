use yew::prelude::*;

#[function_component(Header)]
pub fn header() -> Html {
    html! {
        <header
            class="flex items-center gap-4 px-5 py-3 border-b border-vect-border bg-vect-surface shrink-0"
            role="banner"
        >
            <div class="flex items-center gap-3" aria-label="VECT logo and title">
                <div class="w-8 h-8 rounded-lg bg-vect-accent flex items-center justify-center">
                    <i class="fa-solid fa-bolt text-white text-sm" aria-hidden="true"></i>
                </div>
                <div>
                    <h1 class="text-lg font-bold tracking-tight text-vect-text leading-none">{"VECT"}</h1>
                    <p class="text-xs text-vect-muted leading-none mt-0.5 hidden sm:block">
                        {"Variable Equation Computation Technology"}
                    </p>
                </div>
            </div>

            <div class="ml-auto flex items-center gap-2">
                <a
                    href="https://github.com/wiseaidotdev/lmm"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="vect-btn-ghost h-8 w-8 rounded-lg p-0 flex items-center justify-center"
                    aria-label="LMM on GitHub"
                    title="LMM on GitHub"
                >
                    <i class="fa-brands fa-github text-sm" aria-hidden="true"></i>
                </a>
                <a
                    href="https://docs.rs/lmm"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="vect-btn-ghost h-8 w-8 rounded-lg p-0 flex items-center justify-center"
                    aria-label="LMM documentation on docs.rs"
                    title="LMM docs"
                >
                    <i class="fa-solid fa-book text-sm" aria-hidden="true"></i>
                </a>
                <a
                    href="https://wiseai.dev"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="vect-btn-ghost h-8 w-8 rounded-lg p-0 flex items-center justify-center"
                    aria-label="Wise AI - LMM homepage"
                    title="wiseai.dev"
                >
                    <i class="fa-solid fa-globe text-sm" aria-hidden="true"></i>
                </a>
                <div class="w-px h-5 bg-vect-border mx-1" aria-hidden="true" />
                <span class="badge bg-vect-elevated text-vect-muted text-xs">
                    {"LMM"}
                </span>
                <span class="badge bg-vect-elevated text-vect-muted text-xs">
                    {"WASM"}
                </span>
            </div>
        </header>
    }
}
