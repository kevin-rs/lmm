mod app;
mod components;
mod lmm_bridge;
mod types;

fn main() {
    yew::Renderer::<app::App>::new().render();
}
