use crate::components::message_bubble::MessageBubble;
use crate::types::{ChatMessage, GenerationMode};
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct ChatWindowProps {
    pub messages: Vec<ChatMessage>,
    pub is_loading: bool,
}

#[function_component(ChatWindow)]
pub fn chat_window(props: &ChatWindowProps) -> Html {
    let bottom_ref = use_node_ref();

    {
        let bottom_ref = bottom_ref.clone();
        let len = props.messages.len();
        use_effect_with(len, move |_| {
            if let Some(el) = bottom_ref.cast::<web_sys::Element>() {
                el.scroll_into_view();
            }
            move || {}
        });
    }

    html! {
        <section
            class="flex-1 overflow-y-auto px-4 md:px-8 py-6"
            aria-label="Chat conversation"
            id="chat-window"
        >
            if props.messages.is_empty() {
                <div class="flex flex-col items-center justify-center h-full gap-8 text-center max-w-2xl mx-auto">
                    <div>
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="56" height="56" class="mx-auto mb-4" aria-hidden="true">
                            <defs>
                                <linearGradient id="cw-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stop-color="#7c3aed"/>
                                    <stop offset="100%" stop-color="#06b6d4"/>
                                </linearGradient>
                            </defs>
                            <polygon points="24,2 44,14 44,34 24,46 4,34 4,14" fill="none" stroke="url(#cw-grad)" stroke-width="1.5"/>
                            <polygon points="24,10 38,19 38,29 24,38 10,29 10,19" fill="none" stroke="url(#cw-grad)" stroke-width="1" opacity="0.6"/>
                            <circle cx="24" cy="24" r="3.5" fill="url(#cw-grad)"/>
                        </svg>
                        <h2 class="text-2xl font-bold text-vect-text mb-2">{"Welcome to VECT"}</h2>
                        <p class="text-vect-muted text-sm leading-relaxed">
                            {"Select a generation mode from the sidebar, then type any text below to begin."}
                        </p>
                    </div>
                    <div class="grid grid-cols-2 md:grid-cols-3 gap-3 w-full" role="list" aria-label="Available modes">
                        {for GenerationMode::all().into_iter().map(|m| html! {
                            <div
                                key={m.label()}
                                class="bg-vect-surface border border-vect-border rounded-xl p-3 text-left hover:border-vect-violet/50 transition-colors"
                                role="listitem"
                            >
                                <span class="text-xl" aria-hidden="true">{m.icon()}</span>
                                <p class="text-sm font-semibold text-vect-text mt-1">{m.label()}</p>
                                <p class="text-xs text-vect-muted mt-0.5 leading-tight">{m.description()}</p>
                            </div>
                        })}
                    </div>
                </div>
            } else {
                <div class="flex flex-col max-w-4xl mx-auto">
                    {for props.messages.iter().map(|msg| html! {
                        <MessageBubble key={msg.id} message={msg.clone()} />
                    })}
                    if props.is_loading {
                        <div class="flex items-start gap-3 mb-5 animate-fade-in" aria-label="Assistant is thinking" aria-live="polite">
                            <div class="w-8 h-8 rounded-full bg-gradient-to-br from-vect-violet to-vect-cyan flex items-center justify-center shrink-0 mt-1 animate-pulse-glow">
                                <span class="text-white text-xs font-bold">{"V"}</span>
                            </div>
                            <div class="message-ai flex items-center gap-2">
                                <span class="text-vect-muted text-sm">{"Searching & generating"}</span>
                                <span class="flex gap-1" aria-hidden="true">
                                    <span class="w-1.5 h-1.5 bg-vect-violet rounded-full animate-bounce" style="animation-delay: 0ms"/>
                                    <span class="w-1.5 h-1.5 bg-vect-violet rounded-full animate-bounce" style="animation-delay: 150ms"/>
                                    <span class="w-1.5 h-1.5 bg-vect-violet rounded-full animate-bounce" style="animation-delay: 300ms"/>
                                </span>
                            </div>
                        </div>
                    }
                    <div ref={bottom_ref} id="chat-bottom" aria-hidden="true" />
                </div>
            }
        </section>
    }
}
