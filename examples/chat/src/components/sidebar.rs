use crate::types::GenerationMode;
use input_rs::yew::Input;
use wasm_bindgen::JsCast;
use web_sys::{HtmlInputElement, InputEvent, MouseEvent};
use yew::prelude::*;

fn validate_count(v: String) -> bool {
    v.parse::<usize>()
        .map(|n| (1..=200).contains(&n))
        .unwrap_or(false)
}

#[derive(Properties, PartialEq)]
pub struct SidebarProps {
    pub mode: GenerationMode,
    pub on_mode_change: Callback<GenerationMode>,
    pub stochastic: bool,
    pub on_stochastic_change: Callback<bool>,
    pub probability: u32,
    pub on_probability_change: Callback<u32>,
    pub web_search: bool,
    pub on_web_search_change: Callback<bool>,
    pub sentence_count_handle: UseStateHandle<String>,
    pub sentence_count_valid: UseStateHandle<bool>,
    pub paragraph_count_handle: UseStateHandle<String>,
    pub paragraph_count_valid: UseStateHandle<bool>,
    pub predict_length_handle: UseStateHandle<String>,
    pub predict_length_valid: UseStateHandle<bool>,
}

#[function_component(Sidebar)]
pub fn sidebar(props: &SidebarProps) -> Html {
    let sentence_ref = use_node_ref();
    let paragraph_ref = use_node_ref();
    let predict_ref = use_node_ref();

    let show_sentences = matches!(
        props.mode,
        GenerationMode::Paragraph | GenerationMode::Essay | GenerationMode::Summarize
    );
    let show_paragraphs = props.mode == GenerationMode::Essay;
    let show_predict_len = props.mode == GenerationMode::Predict;
    let show_web_search = props.mode == GenerationMode::Ask;

    let on_prob = {
        let cb = props.on_probability_change.clone();
        Callback::from(move |e: InputEvent| {
            if let Some(el) = e
                .target()
                .and_then(|t| t.dyn_into::<HtmlInputElement>().ok())
                && let Ok(v) = el.value().parse::<u32>()
            {
                cb.emit(v);
            }
        })
    };

    html! {
        <nav class="flex flex-col gap-5 p-4 h-full" aria-label="VECT configuration">

            <section aria-labelledby="mode-label">
                <h2 id="mode-label" class="sidebar-label">{"Generation Mode"}</h2>
                <div class="flex flex-col gap-0.5" role="radiogroup" aria-label="Generation mode">
                    {for GenerationMode::all().into_iter().map(|m| {
                        let active = m == props.mode;
                        let cb = props.on_mode_change.clone();
                        let mc = m.clone();
                        let onclick = Callback::from(move |_: MouseEvent| cb.emit(mc.clone()));
                        html! {
                            <button
                                key={m.label()}
                                class={classes!("mode-btn", if active { "mode-btn-active" } else { "mode-btn-inactive" })}
                                onclick={onclick}
                                role="radio"
                                aria-checked={active.to_string()}
                                id={format!("mode-{}", m.label().to_lowercase())}
                                title={m.description()}
                            >
                                <span class="text-base shrink-0" aria-hidden="true">{m.icon()}</span>
                                <div class="flex flex-col items-start min-w-0">
                                    <span class="text-sm font-medium leading-none">{m.label()}</span>
                                    <span class="text-[10px] opacity-50 leading-snug mt-0.5 truncate w-full">{m.description()}</span>
                                </div>
                            </button>
                        }
                    })}
                </div>
            </section>

            <section aria-labelledby="stochastic-label">
                <h2 id="stochastic-label" class="sidebar-label">{"Stochasticity"}</h2>
                <div class="flex flex-col gap-3">
                    <div class="control-row">
                        <label for="stochastic-toggle" class="text-sm text-vect-text cursor-pointer select-none">
                            {"Enable synonym replacement"}
                        </label>
                        <button
                            role="switch"
                            aria-checked={props.stochastic.to_string()}
                            id="stochastic-toggle"
                            onclick={{
                                let cb = props.on_stochastic_change.clone();
                                let v = props.stochastic;
                                Callback::from(move |_: MouseEvent| cb.emit(!v))
                            }}
                            class={classes!(
                                "toggle-track",
                                "shrink-0",
                                if props.stochastic { "bg-vect-violet" } else { "bg-vect-border" }
                            )}
                        >
                            <span class={classes!(
                                "toggle-thumb",
                                if props.stochastic { "translate-x-5" } else { "translate-x-0" }
                            )} />
                        </button>
                    </div>
                    if props.stochastic {
                        <div class="flex flex-col gap-1.5 animate-fade-in">
                            <div class="control-row">
                                <label for="prob-slider" class="text-sm text-vect-text">{"Probability"}</label>
                                <span class="text-sm font-mono font-semibold text-vect-violet-light" aria-live="polite">
                                    {format!("{}%", props.probability)}
                                </span>
                            </div>
                            <input
                                type="range"
                                id="prob-slider"
                                min="1"
                                max="100"
                                value={props.probability.to_string()}
                                class="w-full h-1.5 rounded-full cursor-pointer accent-vect-violet bg-vect-border"
                                oninput={on_prob}
                                aria-label="Synonym replacement probability"
                                aria-valuemin="1"
                                aria-valuemax="100"
                                aria-valuenow={props.probability.to_string()}
                            />
                            <p class="text-xs text-vect-subtle">
                                {format!("{}% chance each word is replaced by a synonym.", props.probability)}
                            </p>
                        </div>
                    }
                </div>
            </section>

            if show_web_search || show_sentences || show_paragraphs || show_predict_len {
                <section aria-labelledby="params-label">
                    <h2 id="params-label" class="sidebar-label">{"Mode Parameters"}</h2>
                    <div class="flex flex-col gap-3">

                        if show_web_search {
                            <div class="flex flex-col gap-2">
                                <div class="control-row">
                                    <label for="web-search-toggle" class="text-sm text-vect-text cursor-pointer select-none">
                                        {"Live Web Search"}
                                    </label>
                                    <button
                                        role="switch"
                                        aria-checked={props.web_search.to_string()}
                                        id="web-search-toggle"
                                        onclick={{
                                            let cb = props.on_web_search_change.clone();
                                            let v = props.web_search;
                                            Callback::from(move |_: MouseEvent| cb.emit(!v))
                                        }}
                                        class={classes!(
                                            "toggle-track",
                                            "shrink-0",
                                            if props.web_search { "bg-vect-cyan" } else { "bg-vect-border" }
                                        )}
                                    >
                                        <span class={classes!(
                                            "toggle-thumb",
                                            if props.web_search { "translate-x-5" } else { "translate-x-0" }
                                        )} />
                                    </button>
                                </div>
                                if props.web_search {
                                    <p class="text-xs text-vect-cyan-light animate-fade-in leading-relaxed">
                                        {"🌐 DuckDuckGo search fetches live context for your query."}
                                    </p>
                                } else {
                                    <p class="text-xs text-vect-subtle leading-relaxed">
                                        {"Enable to use live web search (requires network)."}
                                    </p>
                                }
                            </div>
                        }

                        if show_sentences {
                            <Input
                                r#type={"number"}
                                label={"Sentences"}
                                handle={props.sentence_count_handle.clone()}
                                name={"sentence_count"}
                                r#ref={sentence_ref}
                                placeholder={"3"}
                                input_class={"vect-input text-sm py-2"}
                                field_class={"w-full"}
                                label_class={"text-xs text-vect-muted block mb-1"}
                                error_class={"text-red-400 text-xs mt-1"}
                                error_message={"Enter 1 – 200"}
                                valid_handle={props.sentence_count_valid.clone()}
                                validate_function={Callback::from(validate_count)}
                                id={"sentence-count-input"}
                            />
                        }

                        if show_paragraphs {
                            <Input
                                r#type={"number"}
                                label={"Paragraphs"}
                                handle={props.paragraph_count_handle.clone()}
                                name={"paragraph_count"}
                                r#ref={paragraph_ref}
                                placeholder={"3"}
                                input_class={"vect-input text-sm py-2"}
                                field_class={"w-full"}
                                label_class={"text-xs text-vect-muted block mb-1"}
                                error_class={"text-red-400 text-xs mt-1"}
                                error_message={"Enter 1 – 200"}
                                valid_handle={props.paragraph_count_valid.clone()}
                                validate_function={Callback::from(validate_count)}
                                id={"paragraph-count-input"}
                            />
                        }

                        if show_predict_len {
                            <Input
                                r#type={"number"}
                                label={"Predict Length (chars)"}
                                handle={props.predict_length_handle.clone()}
                                name={"predict_length"}
                                r#ref={predict_ref}
                                placeholder={"80"}
                                input_class={"vect-input text-sm py-2"}
                                field_class={"w-full"}
                                label_class={"text-xs text-vect-muted block mb-1"}
                                error_class={"text-red-400 text-xs mt-1"}
                                error_message={"Enter 1 – 200"}
                                valid_handle={props.predict_length_valid.clone()}
                                validate_function={Callback::from(validate_count)}
                                id={"predict-length-input"}
                            />
                        }
                    </div>
                </section>
            }

            <div class="mt-auto pt-4 border-t border-vect-border/40">
                <p class="text-[11px] text-vect-subtle leading-relaxed">
                    {"Powered by "}
                    <span class="text-vect-violet-light font-semibold">{"LMM"}</span>
                    {" - Large Mathematical Model. All text generation runs locally in your browser via WebAssembly."}
                </p>
            </div>
        </nav>
    }
}
