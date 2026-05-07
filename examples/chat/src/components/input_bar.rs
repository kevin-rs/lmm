use input_rs::yew::Input;
use web_sys::{HtmlTextAreaElement, KeyboardEvent, MouseEvent};
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct InputBarProps {
    pub on_submit: Callback<String>,
    pub is_loading: bool,
}

fn do_submit(input_ref: &NodeRef, on_submit: &Callback<String>) {
    if let Some(el) = input_ref.cast::<HtmlTextAreaElement>() {
        let val = el.value();
        if !val.trim().is_empty() {
            on_submit.emit(val);
            el.set_value("");
        }
    }
}

#[function_component(InputBar)]
pub fn input_bar(props: &InputBarProps) -> Html {
    let input_ref = use_node_ref();
    let input_handle = use_state(String::default);
    let input_valid = use_state(|| true);

    {
        let r = input_ref.clone();
        use_effect_with((), move |_| {
            if let Some(el) = r.cast::<HtmlTextAreaElement>() {
                el.set_value("What is the Rust programming language?");
            }
            || ()
        });
    }

    let on_send: Callback<MouseEvent> = {
        let r = input_ref.clone();
        let cb = props.on_submit.clone();
        Callback::from(move |_| do_submit(&r, &cb))
    };

    let on_keydown: Callback<KeyboardEvent> = {
        let r = input_ref.clone();
        let cb = props.on_submit.clone();
        Callback::from(move |e: KeyboardEvent| {
            if e.key() == "Enter" && !e.shift_key() {
                e.prevent_default();
                do_submit(&r, &cb);
            }
        })
    };

    html! {
        <div
            class="border-t border-vect-border bg-vect-surface px-4 py-3 shrink-0"
            onkeydown={on_keydown}
        >
            <div class="flex items-end gap-3 max-w-4xl mx-auto">
                <div class="flex-1">
                    <Input
                        r#type={"textarea"}
                        label={""}
                        handle={input_handle}
                        name={"message"}
                        r#ref={input_ref}
                        placeholder={"Ask anything..."}
                        input_class={"vect-textarea"}
                        field_class={"w-full"}
                        error_class={""}
                        valid_handle={input_valid}
                        validate_function={Callback::from(|v: String| !v.trim().is_empty())}
                        id={"message-input"}
                    />
                </div>
                <button
                    class="vect-btn-primary h-10 w-10 rounded-full shrink-0 mb-0.5"
                    onclick={on_send}
                    disabled={props.is_loading}
                    aria-label="Send message"
                    id="send-button"
                    type="button"
                >
                    if props.is_loading {
                        <i class="fa-solid fa-spinner fa-spin text-sm" aria-hidden="true"></i>
                    } else {
                        <i class="fa-solid fa-arrow-up text-sm" aria-hidden="true"></i>
                    }
                </button>
            </div>
            <p class="text-center text-xs text-vect-subtle mt-2">
                {"Enter to send  •  Shift+Enter for newline  •  All generation runs locally in WASM"}
            </p>
        </div>
    }
}
