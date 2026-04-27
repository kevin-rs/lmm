// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

use arc_lmm_agent::config::AgentConfig;
use arc_lmm_agent::display;
use arc_lmm_agent::runner::ArcGameRunner;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .with_target(false)
        .compact()
        .init();

    let config = AgentConfig::parse();

    info!(
        game_id = %config.game_id,
        base_url = %config.base_url,
        max_actions = config.max_actions,
        "Starting ARC-LMM-Agent"
    );

    display::print_banner(&config.game_id, &config.base_url, config.max_actions);

    let mut runner = ArcGameRunner::new(config)?;
    let summary = runner.run().await?;

    info!(
        game_id    = %summary.game_id,
        final_state = %summary.final_state,
        levels_completed = summary.levels_completed,
        win_levels = summary.win_levels,
        steps_taken = summary.steps_taken,
        card_id    = %summary.card_id,
        "Run complete"
    );

    display::print_run_summary(
        &summary.game_id,
        summary.final_state.as_str(),
        summary.levels_completed,
        summary.win_levels,
        summary.steps_taken,
        &summary.card_id,
    );

    Ok(())
}
