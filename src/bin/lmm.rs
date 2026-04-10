use clap::Parser;
use lmm::cli::commands::{
    Cli,
    Commands::{Consciousness as CmdConsciousness, Discover, Simulate},
};
use lmm::consciousness::Consciousness;
use lmm::discovery::SymbolicRegression;
use lmm::error::Result;
use lmm::tensor::Tensor;
use lmm::traits::Discoverable;
use lmm::world::WorldModel;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Simulate { step, steps } => {
            println!("Simulate called with step={} over {} steps", step, steps);
            Ok(())
        }
        Discover { data_path } => {
            println!("Discovering using data from {}", data_path);
            let data = vec![Tensor::zeros(vec![1]), Tensor::zeros(vec![1])];
            let eq = SymbolicRegression::discover(&data)?;
            println!("Discovered Equation: {:?}", eq);
            Ok(())
        }
        CmdConsciousness => {
            let mut consc = Consciousness {
                world_model: WorldModel {
                    current_state: Tensor::zeros(vec![4]),
                },
            };
            let fake_input = vec![128, 64, 32, 255];
            let new_state = consc.tick(&fake_input)?;
            println!("Consciousness ticked. New state: {:?}", new_state.data);
            Ok(())
        }
    }
}
