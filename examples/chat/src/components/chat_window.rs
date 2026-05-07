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
                        <div class="w-14 h-14 rounded-2xl bg-vect-elevated flex items-center justify-center mx-auto mb-4">
                            <i class="fa-solid fa-bolt text-vect-accent text-2xl" aria-hidden="true"></i>
                        </div>
                        <h2 class="text-2xl font-bold text-vect-text mb-2">{"Welcome to VECT"}</h2>
                        <p class="text-vect-muted text-sm leading-relaxed">
                            {"Select a generation mode from the sidebar, then type any text below to begin."}
                        </p>
                    </div>
                    <div class="grid grid-cols-2 md:grid-cols-3 gap-3 w-full" role="list" aria-label="Available modes">
                        {for GenerationMode::all().into_iter().map(|m| html! {
                            <div
                                key={m.label()}
                                class="bg-vect-surface border border-vect-border rounded-xl p-3 text-left hover:border-vect-subtle transition-colors"
                                role="listitem"
                            >
                                <i class={classes!(m.icon_class(), "text-lg", "text-vect-muted")} aria-hidden="true"></i>
                                <p class="text-sm font-semibold text-vect-text mt-1.5">{m.label()}</p>
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
                            <div class="w-8 h-8 rounded-full bg-vect-accent flex items-center justify-center shrink-0 mt-1">
                                <i class="fa-solid fa-bolt text-white text-xs" aria-hidden="true"></i>
                            </div>
                            <div class="message-ai flex items-center gap-2">
                                <span class="text-vect-muted text-sm">{"Searching & generating"}</span>
                                <span class="flex gap-1" aria-hidden="true">
                                    <span class="w-1.5 h-1.5 bg-vect-muted rounded-full animate-bounce" style="animation-delay: 0ms"/>
                                    <span class="w-1.5 h-1.5 bg-vect-muted rounded-full animate-bounce" style="animation-delay: 150ms"/>
                                    <span class="w-1.5 h-1.5 bg-vect-muted rounded-full animate-bounce" style="animation-delay: 300ms"/>
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
