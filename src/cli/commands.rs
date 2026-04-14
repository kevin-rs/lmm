use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "lmm")]
#[command(
    about = "Large Mathematical Model: Equation-Based Intelligence Framework",
    version = "0.0.1"
)]
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
        #[arg(short, long, default_value = "synthetic")]
        data_path: String,
        #[arg(short = 'i', long, default_value_t = 100)]
        iterations: usize,
    },
    Consciousness {
        #[arg(short, long, default_value_t = 3)]
        lookahead: usize,
    },
    Physics {
        #[arg(short, long, default_value = "harmonic")]
        model: String,
        #[arg(short, long, default_value_t = 200)]
        steps: usize,
        #[arg(short = 'z', long, default_value_t = 0.01)]
        step_size: f64,
    },
    Causal {
        #[arg(short = 'n', long, default_value = "x")]
        intervene_node: String,
        #[arg(short = 'v', long, default_value_t = 1.0)]
        intervene_value: f64,
    },
    Field {
        #[arg(short, long, default_value_t = 10)]
        size: usize,
        #[arg(short = 'o', long, default_value = "gradient")]
        operation: String,
    },
    Encode {
        #[arg(short, long, default_value = "-")]
        input: String,
        #[arg(short, long, default_value = "Hello, LMM!")]
        text: String,
        #[arg(long, default_value_t = 80)]
        iterations: usize,
        #[arg(long, default_value_t = 4)]
        depth: usize,
    },
    Decode {
        #[arg(short, long)]
        equation: String,
        #[arg(short, long)]
        length: usize,
        #[arg(short, long, default_value = "", allow_hyphen_values = true)]
        residuals: String,
    },
    Predict {
        #[arg(short, long, default_value = "-")]
        input: String,
        #[arg(short, long, default_value = "The Pharaohs encoded reality in")]
        text: String,
        #[arg(short = 'w', long, default_value_t = 32)]
        window: usize,
        #[arg(short = 'p', long, default_value_t = 16)]
        predict_length: usize,
        #[arg(long, default_value_t = 80)]
        iterations: usize,
        #[arg(long, default_value_t = 4)]
        depth: usize,
    },
}
