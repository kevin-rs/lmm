use yew::prelude::*;

#[function_component(Header)]
pub fn header() -> Html {
    html! {
        <header
            class="flex items-center gap-4 px-5 py-3 border-b border-vect-border bg-vect-surface shrink-0"
            role="banner"
        >
            <div class="flex items-center gap-3" aria-label="VECT logo and title">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="36" height="36" aria-hidden="true">
                    <defs>
                        <linearGradient id="vect-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#7c3aed"/>
                            <stop offset="100%" stop-color="#06b6d4"/>
                        </linearGradient>
                    </defs>
                    <polygon points="24,2 44,14 44,34 24,46 4,34 4,14" fill="none" stroke="url(#vect-grad)" stroke-width="1.5"/>
                    <polygon points="24,10 38,19 38,29 24,38 10,29 10,19" fill="none" stroke="url(#vect-grad)" stroke-width="1" opacity="0.6"/>
                    <line x1="24" y1="2" x2="24" y2="46" stroke="#7c3aed" stroke-width="0.5" opacity="0.35"/>
                    <line x1="4" y1="24" x2="44" y2="24" stroke="#06b6d4" stroke-width="0.5" opacity="0.35"/>
                    <circle cx="24" cy="24" r="3.5" fill="url(#vect-grad)"/>
                </svg>
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
                    <svg class="w-4 h-4" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                        <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
                    </svg>
                </a>
                <a
                    href="https://docs.rs/lmm"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="vect-btn-ghost h-8 w-8 rounded-lg p-0 flex items-center justify-center"
                    aria-label="LMM documentation on docs.rs"
                    title="LMM docs"
                >
                    <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/>
                    </svg>
                </a>
                <a
                    href="https://wiseai.dev"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="vect-btn-ghost h-8 w-8 rounded-lg p-0 flex items-center justify-center"
                    aria-label="Wise AI - LMM homepage"
                    title="wiseai.dev"
                >
                    <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9"/>
                    </svg>
                </a>
                <div class="w-px h-5 bg-vect-border mx-1" aria-hidden="true" />
                <span class="badge bg-vect-violet/20 text-vect-violet-light border border-vect-violet/30 text-xs">
                    {"LMM"}
                </span>
                <span class="badge bg-vect-cyan/20 text-vect-cyan-light border border-vect-cyan/30 text-xs">
                    {"WASM"}
                </span>
            </div>
        </header>
    }
}
