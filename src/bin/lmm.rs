use clap::Parser;
use lmm::causal::CausalGraph;
use lmm::cli::commands::{
    Cli,
    Commands::{
        Causal, Consciousness as CmdConsciousness, Decode, Discover, Encode, Field as CmdField,
        Physics, Simulate,
    },
};
use lmm::consciousness::Consciousness;
use lmm::discovery::SymbolicRegression;
use lmm::encode::{decode_message, encode_text};
use lmm::equation::Expression;
use lmm::error::Result;
use lmm::field::Field;
use lmm::physics::{HarmonicOscillator, LorenzSystem, Pendulum, SIRModel};
use lmm::simulation::Simulator;
use lmm::tensor::Tensor;
use lmm::traits::{Causal as CausalTrait, Simulatable};
use std::str::FromStr;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Simulate { step, steps } => {
            let osc = HarmonicOscillator::new(1.0, 1.0, 0.0)?;
            let sim = Simulator { step_size: step };
            let trajectory = sim.simulate_trajectory(&osc, osc.state(), steps)?;
            println!(
                "Simulated {} steps with step_size={}",
                trajectory.len() - 1,
                step
            );
            println!("Final state: {:?}", trajectory.last().unwrap().data);
        }
        Discover {
            data_path: _,
            iterations,
        } => {
            let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
            let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
            let inputs: Vec<Vec<f64>> = xs.iter().map(|&x| vec![x]).collect();
            let sr = SymbolicRegression::new(3, iterations).with_variables(vec!["x".into()]);
            let expr = sr.fit(&inputs, &ys)?;
            println!("Discovered equation: {}", expr);
        }
        CmdConsciousness { lookahead } => {
            let mut consc = Consciousness::new(Tensor::zeros(vec![4]), lookahead, 0.01);
            let input = vec![128u8, 64, 32, 255];
            let state = consc.tick(&input)?;
            println!("Consciousness ticked. New state: {:?}", state.data);
            println!("Mean prediction error: {}", consc.mean_prediction_error());
        }
        Physics {
            model,
            steps,
            step_size,
        } => {
            let sim = Simulator { step_size };
            match model.as_str() {
                "lorenz" => {
                    let sys = LorenzSystem::canonical()?;
                    let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
                    println!(
                        "Lorenz: {} steps. Final xyz: {:?}",
                        traj.len() - 1,
                        traj.last().unwrap().data
                    );
                }
                "pendulum" => {
                    let sys = Pendulum::new(9.81, 1.0, 0.3, 0.0)?;
                    let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
                    println!(
                        "Pendulum: {} steps. Final [theta, omega]: {:?}",
                        traj.len() - 1,
                        traj.last().unwrap().data
                    );
                }
                "sir" => {
                    let sys = SIRModel::new(0.3, 0.1, 990.0, 10.0, 0.0)?;
                    let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
                    println!(
                        "SIR: {} steps. Final [S,I,R]: {:?}",
                        traj.len() - 1,
                        traj.last().unwrap().data
                    );
                }
                _ => {
                    let sys = HarmonicOscillator::new(1.0, 1.0, 0.0)?;
                    let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
                    println!(
                        "Harmonic: {} steps. Final [x,v]: {:?}",
                        traj.len() - 1,
                        traj.last().unwrap().data
                    );
                    println!("Energy (initial): {:.6}, (final): {:.6}", 0.5, {
                        let s = traj.last().unwrap();
                        0.5 * s.data[1] * s.data[1] + 0.5 * s.data[0] * s.data[0]
                    });
                }
            }
        }
        Causal {
            intervene_node,
            intervene_value,
        } => {
            let mut graph = CausalGraph::new();
            let x_id = graph.add_node("x", Some(Expression::Constant(0.0)));
            let y_id = graph.add_node(
                "y",
                Some(Expression::Mul(
                    Box::new(Expression::Constant(2.0)),
                    Box::new(Expression::Variable("x".into())),
                )),
            );
            let z_id = graph.add_node(
                "z",
                Some(Expression::Add(
                    Box::new(Expression::Variable("y".into())),
                    Box::new(Expression::Constant(1.0)),
                )),
            );
            graph.add_edge(x_id, y_id, 1.0)?;
            graph.add_edge(y_id, z_id, 1.0)?;
            graph.nodes[x_id].observed_value = Some(3.0);
            let before = graph.forward_pass()?;
            println!(
                "Before intervention: x={:?}, y={:?}, z={:?}",
                before.get(&x_id),
                before.get(&y_id),
                before.get(&z_id)
            );
            graph.intervene(&intervene_node, intervene_value)?;
            let after = graph.forward_pass()?;
            println!(
                "After do({intervene_node}={intervene_value}): x={:?}, y={:?}, z={:?}",
                after.get(&x_id),
                after.get(&y_id),
                after.get(&z_id)
            );
        }
        CmdField { size, operation } => {
            let data: Vec<f64> = (0..size).map(|i| (i as f64).powi(2)).collect();
            let tensor = Tensor::new(vec![size], data)?;
            let field = Field::new(vec![size], tensor)?;
            match operation.as_str() {
                "laplacian" => {
                    let lap = field.compute_laplacian()?;
                    println!("Laplacian of x²: {:?}", lap.values.data);
                }
                _ => {
                    let grad = field.compute_gradient()?;
                    println!("Gradient of x²: {:?}", grad.values.data);
                }
            }
        }

        Encode {
            input,
            text,
            iterations,
            depth,
        } => {
            let source = if input == "-" {
                text
            } else {
                std::fs::read_to_string(&input)
                    .map_err(|e| lmm::error::LmmError::Perception(e.to_string()))?
                    .trim_end()
                    .to_string()
            };

            println!("━━━ LMM ENCODER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Input text  : {:?}", source);
            println!("Characters  : {}", source.len());
            println!("Running GP symbolic regression ({iterations} iterations, depth {depth})…");
            println!();

            let encoded = encode_text(&source, iterations, depth)?;

            println!("{}", encoded.summary());
            println!();
            println!("━━━ ENCODED DATA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            let data_str = encoded.to_data_string();
            println!("{}", data_str);
            println!();
            println!("━━━ VERIFY ROUND-TRIP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            let decoded = decode_message(&encoded)?;
            println!("Decoded text: {:?}", decoded);
            println!(
                "Round-trip  : {}",
                if decoded == source {
                    "✅ PERFECT"
                } else {
                    "⚠ lossy (residuals correct it)"
                }
            );
            println!();
            println!("To decode later, run:");
            println!(
                "  lmm decode --equation {:?} --length {} --residuals {:?}",
                encoded.equation.to_string(),
                encoded.length,
                encoded
                    .residuals
                    .iter()
                    .map(|r| r.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }

        Decode {
            equation,
            length,
            residuals,
        } => {
            println!("━━━ LMM DECODER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Equation : {}", equation);
            println!("Length   : {}", length);

            let expr = Expression::from_str(&equation)
                .map_err(|e| lmm::error::LmmError::Perception(format!("Bad equation: {e}")))?;

            let res_vec: Vec<i32> = if residuals.is_empty() {
                vec![0; length]
            } else {
                residuals
                    .split(',')
                    .map(|s| s.trim().parse::<i32>().unwrap_or(0))
                    .collect()
            };

            let decoded = lmm::encode::decode_from_parts(&expr, length, &res_vec)?;
            println!();
            println!("━━━ DECODED TEXT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("{}", decoded);
        }
    }
    Ok(())
}
