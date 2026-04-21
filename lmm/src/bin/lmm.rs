use lmm::app::run_cli_entry;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    run_cli_entry(std::env::args().collect()).await
}
