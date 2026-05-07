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
                    class="ml-3 mt-1 w-8 h-8 rounded-full bg-vect-subtle flex items-center justify-center shrink-0"
                    aria-hidden="true"
                >
                    <i class="fa-solid fa-user text-vect-text text-xs" aria-hidden="true"></i>
                </div>
            </article>
        }
    } else {
        html! {
            <article class="flex items-start gap-3 mb-5 animate-fade-in" aria-label="Assistant response" aria-live="polite">
                <div
                    class="w-8 h-8 rounded-full bg-vect-accent flex items-center justify-center shrink-0 mt-1"
                    aria-hidden="true"
                >
                    <i class="fa-solid fa-bolt text-white text-xs" aria-hidden="true"></i>
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
                                            class="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-md \
                                                   bg-vect-elevated text-vect-muted border border-vect-border \
                                                   hover:bg-vect-border hover:text-vect-text \
                                                   transition-all duration-150 max-w-[180px] truncate"
                                        >
                                            <i class="fa-solid fa-arrow-up-right-from-square text-[10px] shrink-0" aria-hidden="true"></i>
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
