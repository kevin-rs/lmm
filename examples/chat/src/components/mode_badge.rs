use crate::types::GenerationMode;
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct ModeBadgeProps {
    pub mode: GenerationMode,
}

#[function_component(ModeBadge)]
pub fn mode_badge(props: &ModeBadgeProps) -> Html {
    html! {
        <span class={classes!("badge", props.mode.color_class())} aria-label={format!("Mode: {}", props.mode.label())}>
            <i class={props.mode.icon_class()} aria-hidden="true"></i>
            {" "}{props.mode.label()}
        </span>
    }
}
