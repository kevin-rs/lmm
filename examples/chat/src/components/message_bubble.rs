use crate::components::mode_badge::ModeBadge;
use crate::types::{ChatMessage, MessageRole};
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct MessageBubbleProps {
    pub message: ChatMessage,
}

#[function_component(MessageBubble)]
pub fn message_bubble(props: &MessageBubbleProps) -> Html {
    let is_user = props.message.role == MessageRole::User;

    if is_user {
        html! {
            <article class="flex justify-end mb-5 animate-fade-in" aria-label="Your message">
                <div class="message-user">
                    <p class="text-sm leading-relaxed whitespace-pre-wrap break-words">
                        {&props.message.content}
                    </p>
                </div>
                <div
                    class="ml-3 mt-1 w-8 h-8 rounded-full bg-gradient-to-br from-violet-700 to-vect-violet flex items-center justify-center shrink-0 shadow-glow-violet"
                    aria-hidden="true"
                >
                    <span class="text-white text-xs font-bold">{"U"}</span>
                </div>
            </article>
        }
    } else {
        html! {
            <article class="flex items-start gap-3 mb-5 animate-fade-in" aria-label="Assistant response" aria-live="polite">
                <div
                    class="w-8 h-8 rounded-full bg-gradient-to-br from-vect-violet to-vect-cyan flex items-center justify-center shrink-0 shadow-glow-violet mt-1"
                    aria-hidden="true"
                >
                    <span class="text-white text-xs font-bold">{"V"}</span>
                </div>
                <div class="flex flex-col gap-1.5 min-w-0 flex-1">
                    {if let Some(mode) = &props.message.mode {
                        html! { <div><ModeBadge mode={mode.clone()} /></div> }
                    } else {
                        html! {}
                    }}
                    <div class="message-ai">
                        <p class="text-sm leading-relaxed whitespace-pre-wrap break-words">
                            {&props.message.content}
                        </p>

                        if !props.message.links.is_empty() {
                            <div
                                class="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-vect-border/40"
                                aria-label="Related search links"
                            >
                                {for props.message.links.iter().map(|link| {
                                    let url = link.url.clone();
                                    let title: String = link.title.chars().take(40).collect();
                                    html! {
                                        <a
                                            href={url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            class="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-full \
                                                   bg-vect-violet/10 text-vect-violet-light border border-vect-violet/20 \
                                                   hover:bg-vect-violet/20 hover:border-vect-violet/40 \
                                                   transition-all duration-150 max-w-[180px] truncate"
                                        >
                                            <svg class="w-3 h-3 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
                                                <path stroke-linecap="round" stroke-linejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
                                            </svg>
                                            {title}
                                        </a>
                                    }
                                })}
                            </div>
                        }
                    </div>
                </div>
            </article>
        }
    }
}
