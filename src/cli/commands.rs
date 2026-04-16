use clap::builder::styling::{AnsiColor, Effects, Styles};
use clap::{Parser, Subcommand};

fn styles() -> Styles {
    clap::builder::styling::Styles::styled()
        .header(AnsiColor::Red.on_default() | Effects::BOLD)
        .usage(AnsiColor::Red.on_default() | Effects::BOLD)
        .literal(AnsiColor::Blue.on_default() | Effects::BOLD)
        .error(AnsiColor::Red.on_default() | Effects::BOLD)
        .placeholder(AnsiColor::Green.on_default())
}

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    name = "lmm",
    propagate_version = true,
    styles = styles(),
    help_template = r#"{before-help}{name} {version}
{about}
{usage-heading} {usage}
{all-args}{after-help}

AUTHORS:
    {author}
"#,
    about = r#"
  ▄▄▄      ▄▄▄     ▄▄▄   ▄▄▄     ▄▄▄  
 ▀██▀       ███▄ ▄███     ███▄ ▄███   
  ██        ██ ▀█▀ ██     ██ ▀█▀ ██   
  ██        ██     ██     ██     ██   
  ██        ██     ██     ██     ██   
 ████████ ▀██▀     ▀██▄ ▀██▀     ▀██▄

Large Mathematical Model · Equation-Based Intelligence

The `lmm` CLI enables interaction with the Large Mathematical Model (LMM).
It provides advanced equation discovery, physics simulation, causal 
inference, and unified sequence processing features.
"#
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    #[command(about = "Simulate continuous logical pathways")]
    Simulate {
        #[arg(short, long, default_value_t = 0.01)]
        step: f64,
        #[arg(short = 't', long, default_value_t = 100)]
        steps: usize,
    },
    #[command(about = "Discover governing equations from data")]
    Discover {
        #[arg(short, long, default_value = "synthetic")]
        data_path: String,
        #[arg(short = 'i', long, default_value_t = 100)]
        iterations: usize,
    },
    #[command(about = "Evaluate conscious state coherence")]
    Consciousness {
        #[arg(short, long, default_value_t = 3)]
        lookahead: usize,
    },
    #[command(about = "Run harmonic and chaotic physical models")]
    Physics {
        #[arg(short, long, default_value = "harmonic")]
        model: String,
        #[arg(short, long, default_value_t = 200)]
        steps: usize,
        #[arg(short = 'z', long, default_value_t = 0.01)]
        step_size: f64,
    },
    #[command(about = "Perform causal interventions and counterfactuals")]
    Causal {
        #[arg(short = 'n', long, default_value = "x")]
        intervene_node: String,
        #[arg(short = 'v', long, default_value_t = 1.0)]
        intervene_value: f64,
    },
    #[command(about = "Compute tensor field gradients and divergences")]
    Field {
        #[arg(short, long, default_value_t = 10)]
        size: usize,
        #[arg(short = 'o', long, default_value = "gradient")]
        operation: String,
    },
    #[command(about = "Encode continuous truth into discrete text")]
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
    #[command(about = "Decode text back into dynamic equations")]
    Decode {
        #[arg(short, long)]
        equation: String,
        #[arg(short, long)]
        length: usize,
        #[arg(short, long, default_value = "", allow_hyphen_values = true)]
        residuals: String,
    },
    #[command(about = "Predict next sequence based on pattern logic")]
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
        #[arg(long)]
        dictionary: Option<String>,
    },
    #[command(about = "Extract key meaning via GP scoring")]
    Summarize {
        #[arg(short, long, default_value = "-")]
        input: String,
        #[arg(
            short,
            long,
            default_value = "Equations are the universe. Mathematical models compress reality into compact symbolic forms. Simulation is more powerful than description. The world is not made of words but of structure and force."
        )]
        text: String,
        #[arg(short = 'n', long, default_value_t = 2)]
        sentences: usize,
        #[arg(long, default_value_t = 40)]
        iterations: usize,
        #[arg(long, default_value_t = 3)]
        depth: usize,
    },
    #[command(about = "Generate a single structural sentence")]
    Sentence {
        #[arg(short, long, default_value = "-")]
        input: String,
        #[arg(short, long, default_value = "Mathematical equations")]
        text: String,
        #[arg(long, default_value_t = 60)]
        iterations: usize,
        #[arg(long, default_value_t = 3)]
        depth: usize,
    },
    #[command(about = "Generate a cohesive paragraph from a seed")]
    Paragraph {
        #[arg(short, long, default_value = "-")]
        input: String,
        #[arg(
            short,
            long,
            default_value = "Equations encode the structure of reality"
        )]
        text: String,
        #[arg(short = 'n', long, default_value_t = 3)]
        sentences: usize,
        #[arg(long, default_value_t = 60)]
        iterations: usize,
        #[arg(long, default_value_t = 3)]
        depth: usize,
    },
    #[command(about = "Structure a full essay with intro and conclusion")]
    Essay {
        #[arg(short, long, default_value = "-")]
        input: String,
        #[arg(
            short,
            long,
            default_value = "Mathematical models and the structure of reality"
        )]
        text: String,
        #[arg(short = 'n', long, default_value_t = 2)]
        paragraphs: usize,
        #[arg(short = 's', long, default_value_t = 3)]
        sentences: usize,
        #[arg(long, default_value_t = 60)]
        iterations: usize,
        #[arg(long, default_value_t = 3)]
        depth: usize,
    },
}
