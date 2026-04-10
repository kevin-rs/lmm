use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "lmm")]
#[command(about = "Large Mathematical Model Framework", version = "0.0.0")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Simulate {
        #[arg(short, long, default_value_t = 0.01)]
        step: f64,
        #[arg(short = 't', long, default_value_t = 100)]
        steps: usize,
    },
    Discover {
        #[arg(short, long)]
        data_path: String,
    },
    Consciousness,
}
