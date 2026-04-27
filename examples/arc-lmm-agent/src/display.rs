// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # `display` – production-ready terminal output for arc-lmm-agent.
//!
//! Provides structured, coloured, human-readable prints that surface the agent's
//! inner cognitive mechanics.  All output goes to **stderr** so it does not
//! interfere with structured logs or stdout redirection.

use indicatif::{ProgressBar, ProgressStyle};
use owo_colors::OwoColorize;
use std::time::Duration;

const HEADER_WIDTH: usize = 70;

/// Creates a transient spinner that disappears when [`Spinner::finish`] is called.
///
/// Use for operations with uncertain duration (ThinkLoop, network calls).
pub struct Spinner(ProgressBar);

impl Spinner {
    pub fn new(msg: &str) -> Self {
        let pb = ProgressBar::new_spinner();
        pb.enable_steady_tick(Duration::from_millis(80));
        pb.set_style(
            ProgressStyle::with_template("{spinner:.cyan} {msg}")
                .unwrap()
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        pb.set_message(msg.to_string());
        Self(pb)
    }

    /// Clears the spinner and prints a completion line.
    pub fn finish_ok(self, msg: &str) {
        self.0.finish_and_clear();
        eprintln!("  {} {}", "✔".green().bold(), msg.dimmed());
    }

    /// Clears the spinner silently.
    pub fn finish_silent(self) {
        self.0.finish_and_clear();
    }
}

/// Prints a top-level banner, e.g. at game start.
pub fn print_banner(game_id: &str, base_url: &str, max_actions: usize) {
    let title = format!(
        "  ARC-LMM-Agent  ·  game={}  ·  url={}  ·  budget={}  ",
        game_id, base_url, max_actions
    );
    let width = title.len().max(HEADER_WIDTH);
    let border = "═".repeat(width);
    eprintln!("\n╔{}╗", border.cyan());
    eprintln!("║{}   ║", title.bold().cyan());
    eprintln!("╚{}╝\n", border.cyan());
}

/// Prints a section divider for a new trial.
pub fn print_trial_start(trial: usize, epsilon: f64, q_states: usize) {
    let label = format!(" Trial #{trial}  ε={epsilon:.3}  Q-states={q_states} ");
    let dashes = "─".repeat(HEADER_WIDTH.saturating_sub(label.len()).max(2));
    eprintln!(
        "\n{}{}{}",
        "┌─".yellow(),
        label.yellow().bold(),
        dashes.yellow()
    );
}

/// Prints a trial-end summary line.
pub fn print_trial_end(trial: usize, steps: usize, epsilon: f64, states: usize) {
    eprintln!(
        "{}  trial={} steps={} ε={:.3} states={}",
        "└─ End".yellow(),
        trial.yellow().bold(),
        steps.bright_white(),
        epsilon,
        states
    );
}

/// Prints the compact per-step state digest.
#[allow(clippy::too_many_arguments)]
pub fn print_step_state(
    trial: usize,
    step: usize,
    pos: (i32, i32),
    level: u32,
    win_levels: u32,
    trial_visits: u32,
    global_visits: u32,
    global_states: usize,
    walls: usize,
    passages: usize,
    milestones: usize,
    ui_mods: usize,
    plan_len: usize,
) {
    eprintln!(
        "{} {}  pos={} lvl={}/{} tv={} gv={} Σ={} walls={} pass={} milestones={} mods={} plan={}",
        format!("[t{trial}·s{step}]").bright_cyan().bold(),
        "▸".dimmed(),
        format!("({},{})", pos.0, pos.1).bright_white(),
        level.bright_green(),
        win_levels.dimmed(),
        trial_visits.bright_yellow(),
        global_visits.dimmed(),
        global_states,
        walls.bright_red(),
        passages.green(),
        milestones,
        ui_mods.magenta(),
        if plan_len > 0 {
            format!("{plan_len}").bright_blue().to_string()
        } else {
            "0".dimmed().to_string()
        },
    );
}

/// Prints action direction chosen.
pub fn print_action(action: u32, mode: &str) {
    let dir = match action {
        1 => "↑ UP",
        2 => "↓ DN",
        3 => "← LT",
        4 => "→ RT",
        _ => "· NOP",
    };
    eprintln!(
        "  {} {} {}",
        "→".bright_white().bold(),
        dir.bright_white().bold(),
        format!("[{mode}]").dimmed(),
    );
}

/// Prints the current plan execution status.
pub fn print_plan_step(action: u32, remaining: usize) {
    let dir = match action {
        1 => "↑",
        2 => "↓",
        3 => "←",
        4 => "→",
        _ => "·",
    };
    eprintln!(
        "  {} {} {}",
        "📋".dimmed(),
        format!("PLAN {dir}").bright_blue(),
        format!("({remaining} remaining)").dimmed()
    );
}

/// Prints a plan being loaded.
pub fn print_plan_loaded(label: &str, steps: usize) {
    eprintln!(
        "  {} {} {}",
        "📌".dimmed(),
        format!("Plan loaded: {label}").bright_blue().bold(),
        format!("({steps} steps)").dimmed()
    );
}

/// Prints a routing target decision.
pub fn print_routing(label: &str, target: (usize, usize), path_len: usize, action: u32) {
    let dir = match action {
        1 => "↑",
        2 => "↓",
        3 => "←",
        4 => "→",
        _ => "·",
    };
    eprintln!(
        "  {} {} {} {} {}",
        "🧭".dimmed(),
        format!("Route→{label}").bright_magenta(),
        format!("target=({},{})", target.0, target.1).dimmed(),
        format!("path={path_len}").dimmed(),
        dir.to_string().bright_white().bold(),
    );
}

/// Prints a stuck-escape event.
pub fn print_stuck(mode: &str, action: u32, escape_count: u32) {
    eprintln!(
        "  {} {} {} {}",
        "⚠".bright_yellow().bold(),
        "STUCK".bright_yellow().bold(),
        format!("[{mode}]").bright_yellow(),
        format!("action={action} count={escape_count}").dimmed()
    );
}

/// Prints a modifier (plus sign) reaching event.
pub fn print_modifier_reached(pos: (usize, usize), visit_num: u32) {
    eprintln!(
        "  {} {} {} {}",
        "✚".bright_green().bold(),
        "MODIFIER ACTIVATED".bright_green().bold(),
        format!("at ({},{})", pos.0, pos.1).bright_white(),
        format!("(visit #{visit_num})").dimmed()
    );
}

/// Prints a newly discovered bonus (step booster).
pub fn print_bonus_found(pos: (usize, usize)) {
    eprintln!(
        "  {} {} {}",
        "⭐".yellow(),
        "Booster found".yellow().bold(),
        format!("at ({},{})", pos.0, pos.1).bright_white()
    );
}

/// Prints a bonus collection event.
pub fn print_bonus_collected(pos: (usize, usize), remaining: usize) {
    eprintln!(
        "  {} {} {} {}",
        "🍬".dimmed(),
        "Booster collected".bright_yellow().bold(),
        format!("({},{})", pos.0, pos.1).bright_white(),
        format!("- {remaining} remaining").dimmed()
    );
}

/// Prints a backtrack event.
pub fn print_backtrack(steps: usize) {
    eprintln!(
        "  {} {} {}",
        "↩".bright_cyan().bold(),
        "Backtracking to modifier".bright_cyan().bold(),
        format!("({steps} steps)").dimmed()
    );
}

/// Prints the second modifier pass status.
pub fn print_second_mod_pass(complete: bool) {
    if complete {
        eprintln!(
            "  {} {}",
            "✚✚".bright_green().bold(),
            "Second modifier pass complete".bright_green()
        );
    } else {
        eprintln!(
            "  {} {}",
            "✚".bright_green(),
            "Second modifier pass - routing to modifier".dimmed()
        );
    }
}

/// Prints when all bonuses are consumed and agent marches to target.
pub fn print_all_bonuses_consumed() {
    eprintln!(
        "  {} {}",
        "🏁".dimmed(),
        "All boosters collected - marching to final target ↓"
            .bright_green()
            .bold()
    );
}

/// Prints when the final target is locked.
pub fn print_target_locked(pos: (usize, usize)) {
    eprintln!(
        "  {} {} {}",
        "🔒".dimmed(),
        "Final target locked".bright_magenta().bold(),
        format!("at ({},{})", pos.0, pos.1).bright_white()
    );
}

/// Prints a level completion / progressive learning event.
pub fn print_level_advance(from: u32, to: u32, mod_visits: u32, bonuses: usize) {
    eprintln!(
        "\n  {} {} {}",
        "🎉".dimmed(),
        format!("Level {from} → {to}").bright_green().bold(),
        format!("[mod_visits={mod_visits}  bonuses_collected={bonuses}]").dimmed()
    );
}

/// Prints a milestone recording event.
pub fn print_milestone(trial: usize, step: usize, levels: u32, state: u64) {
    eprintln!(
        "  {} {}",
        "📍".dimmed(),
        format!("Milestone  trial={trial}  step={step}  lvl={levels}  state={state:x}").dimmed()
    );
}

/// Prints a dominant internal drive signal when its magnitude crosses the threshold.
pub fn print_drive(name: &str, magnitude: f64) {
    let bar_len = (magnitude * 10.0) as usize;
    let bar: String = "█".repeat(bar_len) + &"░".repeat(10 - bar_len);
    eprintln!(
        "  {} {} [{}] {:.2}",
        "💡".dimmed(),
        format!("Drive:{name}").bright_magenta(),
        bar.magenta(),
        magnitude
    );
}

/// Returns a spinner to display while the ThinkLoop is running.
pub fn think_spinner() -> Spinner {
    Spinner::new("🧠 Cognitive ThinkLoop ...")
}

/// Prints the end-of-run summary box.
pub fn print_run_summary(
    game_id: &str,
    final_state: &str,
    levels_completed: u32,
    win_levels: u32,
    steps_taken: usize,
    card_id: &str,
) {
    let w = HEADER_WIDTH;
    eprintln!("\n{}", "─".repeat(w).cyan());
    eprintln!(
        "{:^width$}",
        "Run Complete".bold().bright_white(),
        width = w
    );
    eprintln!("{}", "─".repeat(w).cyan());
    eprintln!("  {:20} {}", "Game ID".dimmed(), game_id.bright_white());
    eprintln!(
        "  {:20} {}",
        "Final State".dimmed(),
        final_state.bright_green().bold()
    );
    eprintln!(
        "  {:20} {}",
        "Levels Completed".dimmed(),
        format!("{levels_completed} / {win_levels}").bright_green()
    );
    eprintln!(
        "  {:20} {}",
        "Steps Taken".dimmed(),
        steps_taken.to_string().bright_yellow()
    );
    eprintln!("  {:20} {}", "Scorecard ID".dimmed(), card_id.bright_cyan());
    eprintln!("{}\n", "─".repeat(w).cyan());
}

/// Printed once per step in the runner: choice + server confirmation.
#[allow(clippy::too_many_arguments)]
pub fn print_step_summary(
    trial: usize,
    step: usize,
    total: usize,
    action_id: u32,
    epsilon: f64,
    q_states: usize,
    state: &str,
    levels: u32,
    win_levels: u32,
    server_action: u32,
) {
    let dir = match action_id {
        1 => "↑",
        2 => "↓",
        3 => "←",
        4 => "→",
        _ => "·",
    };
    let ok = if action_id == server_action {
        "✔".green().to_string()
    } else {
        "≠".yellow().to_string()
    };
    eprintln!(
        "  {} t{}·s{} tot={} {dir} {} ε={:.3} Q={} lvl={}/{} state={}",
        ok,
        trial,
        step,
        total,
        format!("[{action_id}→{server_action}]").dimmed(),
        epsilon,
        q_states,
        levels,
        win_levels,
        state.dimmed()
    );
}

/// Plan invalidated notification.
pub fn print_plan_invalidated() {
    eprintln!(
        "  {} {}",
        "✖".bright_red(),
        "Plan invalidated (wall/unavailable)".dimmed()
    );
}
