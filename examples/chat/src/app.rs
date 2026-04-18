use crate::components::chat_window::ChatWindow;
use crate::components::header::Header;
use crate::components::input_bar::InputBar;
use crate::components::sidebar::Sidebar;
use crate::lmm_bridge;
use crate::types::{ChatMessage, GenerationMode, MessageRole};
use wasm_bindgen_futures::spawn_local;
use yew::prelude::*;

#[function_component(App)]
pub fn app() -> Html {
    let messages = use_state(Vec::<ChatMessage>::new);
    let mode = use_state(|| GenerationMode::Ask);
    let stochastic = use_state(|| false);
    let probability = use_state(|| 30u32);
    let web_search = use_state(|| true);
    let sentence_count_handle = use_state(|| "3".to_string());
    let sentence_count_valid = use_state(|| true);
    let paragraph_count_handle = use_state(|| "3".to_string());
    let paragraph_count_valid = use_state(|| true);
    let predict_length_handle = use_state(|| "80".to_string());
    let predict_length_valid = use_state(|| true);
    let is_loading = use_state(|| false);
    let next_id = use_state(|| 0usize);
    let sidebar_open = use_state(|| false);

    let on_mode_change = {
        let mode = mode.clone();
        Callback::from(move |m: GenerationMode| mode.set(m))
    };
    let on_stochastic_change = {
        let s = stochastic.clone();
        Callback::from(move |v: bool| s.set(v))
    };
    let on_probability_change = {
        let p = probability.clone();
        Callback::from(move |v: u32| p.set(v))
    };
    let on_web_search_change = {
        let ws = web_search.clone();
        Callback::from(move |v: bool| ws.set(v))
    };

    let on_submit = {
        let messages = messages.clone();
        let mode = mode.clone();
        let stochastic = stochastic.clone();
        let probability = probability.clone();
        let sc_h = sentence_count_handle.clone();
        let pc_h = paragraph_count_handle.clone();
        let pl_h = predict_length_handle.clone();
        let is_loading = is_loading.clone();
        let next_id = next_id.clone();

        Callback::from(move |text: String| {
            let id = *next_id;
            next_id.set(id + 2);

            let current_mode = (*mode).clone();
            let stoch = *stochastic;
            let prob = *probability as f64 / 100.0;
            let sc = (*sc_h).parse::<usize>().unwrap_or(3);
            let pc = (*pc_h).parse::<usize>().unwrap_or(3);
            let pl = (*pl_h).parse::<usize>().unwrap_or(80);

            let user_msg = ChatMessage {
                id,
                role: MessageRole::User,
                content: text.clone(),
                mode: None,
                timestamp: String::new(),
                links: vec![],
            };

            if current_mode == GenerationMode::Ask {
                let mut msgs = (*messages).clone();
                msgs.push(user_msg);
                messages.set(msgs.clone());
                is_loading.set(true);
                let messages = messages.clone();
                let is_loading = is_loading.clone();
                spawn_local(async move {
                    let (response, links) = lmm_bridge::ask(&text, sc, stoch, prob).await;
                    let mut m = msgs;
                    m.push(ChatMessage {
                        id: id + 1,
                        role: MessageRole::Assistant,
                        content: response,
                        mode: Some(GenerationMode::Ask),
                        timestamp: String::new(),
                        links,
                    });
                    messages.set(m);
                    is_loading.set(false);
                });
            } else {
                let response = match &current_mode {
                    GenerationMode::Sentence => lmm_bridge::generate_sentence(&text, stoch, prob),
                    GenerationMode::Paragraph => {
                        lmm_bridge::generate_paragraph(&text, sc, stoch, prob)
                    }
                    GenerationMode::Essay => lmm_bridge::generate_essay(&text, pc, sc, stoch, prob),
                    GenerationMode::Summarize => lmm_bridge::summarize(&text, sc, stoch, prob),
                    GenerationMode::Predict => lmm_bridge::predict(&text, pl, stoch, prob),
                    GenerationMode::Ask => unreachable!(),
                };
                let mut msgs = (*messages).clone();
                msgs.push(user_msg);
                msgs.push(ChatMessage {
                    id: id + 1,
                    role: MessageRole::Assistant,
                    content: response,
                    mode: Some(current_mode),
                    timestamp: String::new(),
                    links: vec![],
                });
                messages.set(msgs);
            }
        })
    };

    let sidebar_content = html! {
        <Sidebar
            mode={(*mode).clone()}
            on_mode_change={on_mode_change.clone()}
            stochastic={*stochastic}
            on_stochastic_change={on_stochastic_change.clone()}
            probability={*probability}
            on_probability_change={on_probability_change.clone()}
            web_search={*web_search}
            on_web_search_change={on_web_search_change.clone()}
            sentence_count_handle={sentence_count_handle.clone()}
            sentence_count_valid={sentence_count_valid.clone()}
            paragraph_count_handle={paragraph_count_handle.clone()}
            paragraph_count_valid={paragraph_count_valid.clone()}
            predict_length_handle={predict_length_handle.clone()}
            predict_length_valid={predict_length_valid.clone()}
        />
    };

    html! {
        <div class="flex flex-col h-screen overflow-hidden bg-vect-bg font-sans">
            <Header />
            <div class="flex flex-1 overflow-hidden">

                <aside
                    class="hidden lg:flex flex-col w-72 border-r border-vect-border bg-vect-surface overflow-y-auto shrink-0"
                    aria-label="VECT configuration panel"
                >
                    {sidebar_content.clone()}
                </aside>

                if *sidebar_open {
                    <div
                        class="lg:hidden fixed inset-0 z-50 flex"
                        role="dialog"
                        aria-modal="true"
                        aria-label="Configuration panel"
                    >
                        <aside class="w-72 bg-vect-surface border-r border-vect-border overflow-y-auto">
                            {sidebar_content}
                        </aside>
                        <div
                            class="flex-1 bg-black/60 cursor-pointer"
                            onclick={{
                                let so = sidebar_open.clone();
                                Callback::from(move |_| so.set(false))
                            }}
                        />
                    </div>
                }

                <main class="flex flex-col flex-1 overflow-hidden" id="main-content">
                    <div class="lg:hidden px-4 pt-3 shrink-0">
                        <button
                            class="vect-btn-ghost text-sm"
                            onclick={{
                                let so = sidebar_open.clone();
                                Callback::from(move |_| so.set(!*so))
                            }}
                            aria-label="Toggle configuration panel"
                            aria-expanded={(*sidebar_open).to_string()}
                        >
                            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                <path stroke-linecap="round" stroke-linejoin="round"
                                    d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/>
                                <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                            </svg>
                            {"Configure"}
                        </button>
                    </div>

                    <ChatWindow messages={(*messages).clone()} is_loading={*is_loading} />
                    <InputBar on_submit={on_submit} is_loading={*is_loading} />
                </main>
            </div>
        </div>
    }
}
