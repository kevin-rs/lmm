#[cfg(feature = "cli")]
use crate::causal::CausalGraph;
#[cfg(all(feature = "cli", feature = "net"))]
use crate::cli::commands::Commands::Ask;
#[cfg(feature = "cli")]
use crate::cli::commands::{
    Cli,
    Commands::{
        Causal, Consciousness as CmdConsciousness, Decode, Discover, Encode, Essay,
        Field as CmdField, Imagen, Paragraph, Physics, Predict, Sentence, Simulate, Summarize,
    },
};
#[cfg(feature = "cli")]
use crate::consciousness::Consciousness;
#[cfg(feature = "cli")]
use crate::discovery::SymbolicRegression;
#[cfg(feature = "cli")]
use crate::encode::{decode_message, encode_text};
#[cfg(feature = "cli")]
use crate::equation::Expression;
#[cfg(feature = "cli")]
use crate::field::Field;
#[cfg(feature = "cli")]
use crate::lexicon::Lexicon;
#[cfg(feature = "cli")]
use crate::physics::{HarmonicOscillator, LorenzSystem, Pendulum, SIRModel};
#[cfg(feature = "cli")]
use crate::predict::TextPredictor;
#[cfg(feature = "cli")]
use crate::simulation::Simulator;
#[cfg(feature = "cli")]
use crate::tensor::Tensor;
#[cfg(feature = "cli")]
use crate::text::{EssayGenerator, ParagraphGenerator, SentenceGenerator, TextSummarizer};
#[cfg(feature = "cli")]
use crate::traits::{Causal as CausalTrait, Simulatable};
#[cfg(feature = "cli")]
use clap::Parser;
#[cfg(feature = "cli")]
use std::str::FromStr;
#[cfg(feature = "cli")]
use tracing::{Event, Subscriber, error, info};
#[cfg(feature = "cli")]
use tracing_appender::rolling;
#[cfg(feature = "cli")]
use tracing_subscriber::{
    Layer, Registry, filter,
    fmt::{self, FmtContext, FormatEvent, FormatFields},
    prelude::__tracing_subscriber_SubscriberExt,
    registry::LookupSpan,
};

#[cfg(feature = "cli")]
struct NoLevelFormatter;

#[cfg(feature = "cli")]
impl<S, N> FormatEvent<S, N> for NoLevelFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: fmt::format::Writer<'_>,
        event: &Event<'_>,
    ) -> std::fmt::Result {
        ctx.format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

#[cfg(feature = "cli")]
fn setup_logging(verbose: bool) -> anyhow::Result<()> {
    let console_level = if verbose {
        filter::LevelFilter::INFO
    } else {
        filter::LevelFilter::WARN
    };
    let file_appender = rolling::daily("logs", "lmm_log");

    let console_layer = fmt::Layer::new()
        .compact()
        .without_time()
        .with_file(false)
        .with_line_number(false)
        .with_thread_ids(false)
        .with_target(false)
        .with_writer(std::io::stdout)
        .event_format(NoLevelFormatter)
        .with_filter(console_level);

    let file_layer = fmt::Layer::new()
        .compact()
        .with_file(true)
        .with_line_number(true)
        .with_thread_ids(true)
        .with_target(true)
        .with_writer(file_appender)
        .with_filter(filter::LevelFilter::DEBUG);

    let subscriber = Registry::default().with(console_layer).with(file_layer);
    tracing::subscriber::set_global_default(subscriber)?;
    Ok(())
}

#[cfg(feature = "cli")]
fn load_source(input: &str, text: String) -> crate::error::Result<String> {
    if input == "-" {
        Ok(text)
    } else {
        std::fs::read_to_string(input)
            .map_err(|e| crate::error::LmmError::Perception(e.to_string()))
            .map(|s| s.trim_end().to_string())
    }
}

#[cfg(feature = "cli")]
fn char_display_width(cp: u32) -> usize {
    match cp {
        0xFE00..=0xFE0F | 0xE0100..=0xE01EF => 0,
        0x200B | 0x200C | 0x200D | 0xFEFF => 0,
        0x1100..=0x115F
        | 0x2E80..=0x303E
        | 0x3041..=0x33FF
        | 0x3400..=0x9FFF
        | 0xA960..=0xA97F
        | 0xAC00..=0xD7FF
        | 0xF900..=0xFAFF
        | 0xFE10..=0xFE19
        | 0xFE30..=0xFE6F
        | 0xFF01..=0xFF60
        | 0xFFE0..=0xFFE6
        | 0x1B000..=0x1B0FF
        | 0x1F004
        | 0x1F0CF
        | 0x1F300..=0x1FFFF
        | 0x20000..=0x2FFFD
        | 0x30000..=0x3FFFD => 2,
        _ => 1,
    }
}

#[cfg(feature = "cli")]
fn display_width(s: &str) -> usize {
    s.chars().map(|c| char_display_width(c as u32)).sum()
}

#[cfg(feature = "cli")]
fn banner(title: &str, icon: &str) {
    let width = 54usize;
    let label = format!("  {}  {}", icon, title);
    let pad = width.saturating_sub(display_width(&label));
    info!("");
    info!("╔{}╗", "═".repeat(width));
    info!("║{}{}║", label, " ".repeat(pad));
    info!("╚{}╝", "═".repeat(width));
    info!("");
}

#[cfg(feature = "cli")]
fn divider(label: &str) {
    let dashes = "-".repeat(46usize.saturating_sub(label.chars().count() + 4));
    info!("");
    info!("-- {} {}", label, dashes);
}

#[cfg(feature = "cli")]
fn maybe_enhance(text: &str, stochastic: bool, probability: f64) -> String {
    if stochastic {
        crate::stochastic::StochasticEnhancer::new(probability).enhance(text)
    } else {
        text.to_string()
    }
}

#[cfg(feature = "cli")]
pub async fn run_cli_entry(args: Vec<String>) -> anyhow::Result<()> {
    let cli = Cli::parse_from(args);
    setup_logging(cli.verbose)?;

    match cli.command {
        Simulate { step, steps } => {
            banner("Simulate · Harmonic Oscillator", "🌊");
            let osc = HarmonicOscillator::new(1.0, 1.0, 0.0)?;
            let sim = Simulator { step_size: step };
            let trajectory = sim.simulate_trajectory(&osc, osc.state(), steps)?;
            info!(
                "✅ Simulation complete - {} steps, step_size={}",
                trajectory.len() - 1,
                step
            );
            let final_data = &trajectory.last().unwrap().data;
            if cli.verbose {
                divider("Final State");
                info!("  {:?}", final_data);
            } else {
                println!("{:?}", final_data);
            }
        }

        Discover {
            data_path: _,
            iterations,
        } => {
            banner("Discover · Symbolic Regression", "🔭");
            let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
            let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
            let inputs: Vec<Vec<f64>> = xs.iter().map(|&x| vec![x]).collect();
            let sr = SymbolicRegression::new(3, iterations).with_variables(vec!["x".into()]);
            let expr = sr.fit(&inputs, &ys)?;
            if cli.verbose {
                divider("Discovered Equation");
                info!("  f(x) = {}", expr);
            } else {
                println!("{}", expr);
            }
        }

        CmdConsciousness { lookahead } => {
            banner("Consciousness · Perception Loop", "🧠");
            let mut consc = Consciousness::new(Tensor::zeros(vec![4]), lookahead, 0.01);
            let input = vec![128u8, 64, 32, 255];
            let state = consc.tick(&input)?;
            if cli.verbose {
                divider("State");
                info!("  {:?}", state.data);
                divider("Mean Prediction Error");
                info!("  {:.6}", consc.mean_prediction_error());
            } else {
                println!("{:?}\n{:.6}", state.data, consc.mean_prediction_error());
            }
        }

        Physics {
            model,
            steps,
            step_size,
        } => {
            banner("Physics · Dynamical Systems", "⚛️");
            let sim = Simulator { step_size };
            match model.as_str() {
                "lorenz" => {
                    let sys = LorenzSystem::canonical()?;
                    let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
                    info!("✅ Lorenz: {} steps", traj.len() - 1);
                    let final_data = &traj.last().unwrap().data;
                    if cli.verbose {
                        divider("Final xyz");
                        info!("  {:?}", final_data);
                    } else {
                        println!("{:?}", final_data);
                    }
                }
                "pendulum" => {
                    let sys = Pendulum::new(9.81, 1.0, 0.3, 0.0)?;
                    let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
                    info!("✅ Pendulum: {} steps", traj.len() - 1);
                    let final_data = &traj.last().unwrap().data;
                    if cli.verbose {
                        divider("Final [θ, ω]");
                        info!("  {:?}", final_data);
                    } else {
                        println!("{:?}", final_data);
                    }
                }
                "sir" => {
                    let sys = SIRModel::new(0.3, 0.1, 990.0, 10.0, 0.0)?;
                    let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
                    info!("✅ SIR: {} steps", traj.len() - 1);
                    let final_data = &traj.last().unwrap().data;
                    if cli.verbose {
                        divider("Final [S, I, R]");
                        info!("  {:?}", final_data);
                    } else {
                        println!("{:?}", final_data);
                    }
                }
                _ => {
                    let sys = HarmonicOscillator::new(1.0, 1.0, 0.0)?;
                    let traj = sim.simulate_trajectory(&sys, sys.state(), steps)?;
                    info!("✅ Harmonic: {} steps", traj.len() - 1);
                    let s = traj.last().unwrap();
                    let energy = 0.5 * s.data[1] * s.data[1] + 0.5 * s.data[0] * s.data[0];
                    if cli.verbose {
                        divider("Final [x, v]");
                        info!("  {:?}", s.data);
                        divider("Final Energy");
                        info!("  {:.6}", energy);
                    } else {
                        println!("{:?}\n{:.6}", s.data, energy);
                    }
                }
            }
        }

        Causal {
            intervene_node,
            intervene_value,
        } => {
            banner("Causal · Intervention Engine", "🔀");
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
            divider("Before Intervention");
            info!(
                "  x={:?}  y={:?}  z={:?}",
                before.get(&x_id),
                before.get(&y_id),
                before.get(&z_id)
            );
            graph.intervene(&intervene_node, intervene_value)?;
            let after = graph.forward_pass()?;
            if cli.verbose {
                divider(&format!("After do({}={})", intervene_node, intervene_value));
                info!(
                    "  x={:?}  y={:?}  z={:?}",
                    after.get(&x_id),
                    after.get(&y_id),
                    after.get(&z_id)
                );
            } else {
                println!(
                    "x={:?} y={:?} z={:?}",
                    after.get(&x_id),
                    after.get(&y_id),
                    after.get(&z_id)
                );
            }
        }

        CmdField { size, operation } => {
            banner("Field · Differential Operators", "🌐");
            let data: Vec<f64> = (0..size).map(|i| (i as f64).powi(2)).collect();
            let tensor = Tensor::new(vec![size], data)?;
            let field = Field::new(vec![size], tensor)?;
            match operation.as_str() {
                "laplacian" => {
                    let lap = field.compute_laplacian()?;
                    if cli.verbose {
                        divider("∇² f(x)");
                        info!("  {:?}", lap.values.data);
                    } else {
                        println!("{:?}", lap.values.data);
                    }
                }
                _ => {
                    let grad = field.compute_gradient()?;
                    if cli.verbose {
                        divider("∇ f(x)");
                        info!("  {:?}", grad.values.data);
                    } else {
                        println!("{:?}", grad.values.data);
                    }
                }
            }
        }

        Encode {
            input,
            text,
            iterations,
            depth,
        } => {
            banner("Encode · GP Symbolic Compression", "📐");
            let source = load_source(&input, text)?;
            info!("  📝 Input   : {:?}", source);
            info!("  📏 Length  : {} chars", source.len());
            let encoded = encode_text(&source, iterations, depth)?;
            let decoded = decode_message(&encoded)?;
            if cli.verbose {
                divider("Equation");
                info!("  {}", encoded.summary());
                divider("Encoded Data");
                info!("  {}", encoded.to_data_string());
                divider("Round-trip Verify");
                let status = if decoded == source {
                    "✅ PERFECT"
                } else {
                    "⚠️  lossy (residuals correct)"
                };
                info!("  {}", status);
                info!("  Decoded : {:?}", decoded);
                info!("");
                info!("  💾 To decode later:");
                info!(
                    "     lmm decode --equation {:?} --length {} --residuals {:?}",
                    encoded.equation.to_string(),
                    encoded.length,
                    encoded
                        .residuals
                        .iter()
                        .map(|r| r.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                );
            } else {
                println!(
                    "{} {} {}",
                    encoded.equation,
                    encoded.length,
                    encoded
                        .residuals
                        .iter()
                        .map(|r| r.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
        }

        Decode {
            equation,
            length,
            residuals,
        } => {
            banner("Decode · Equation to Text", "🔓");
            info!("  📐 Equation : {}", equation);
            info!("  📏 Length   : {}", length);
            let expr = Expression::from_str(&equation)
                .map_err(|e| crate::error::LmmError::Perception(format!("Bad equation: {e}")))?;
            let res_vec: Vec<i32> = if residuals.is_empty() {
                vec![0; length]
            } else {
                residuals
                    .split(',')
                    .map(|s| s.trim().parse::<i32>().unwrap_or(0))
                    .collect()
            };
            let decoded = crate::encode::decode_from_parts(&expr, length, &res_vec)?;
            if cli.verbose {
                divider("Decoded Text");
                info!("  {}", decoded);
            } else {
                println!("{}", decoded);
            }
        }

        Predict {
            input,
            text,
            window,
            predict_length,
            iterations,
            depth,
            dictionary,
            stochastic,
            probability,
        } => {
            banner("Predict · Symbolic Continuation", "🔮");
            let source = load_source(&input, text)?;
            let lexicon = match dictionary {
                Some(path) => Lexicon::load_from(std::path::Path::new(&path)).ok(),
                None => Lexicon::load_system().ok(),
            };
            let mut predictor = TextPredictor::new(window, iterations, depth);
            if let Some(lex) = lexicon {
                info!("  📚 Dictionary: {} words loaded", lex.word_count());
                predictor = predictor.with_lexicon(lex);
            }
            let result = predictor.predict_continuation(&source, predict_length)?;
            info!("  📝 Input     : {:?}", source);
            info!("  🪟 Window    : {} words", result.window_used);
            info!("  📈 Trajectory: {}", result.trajectory_equation);
            info!("  🎵 Rhythm    : {}", result.rhythm_equation);
            let cont = maybe_enhance(&result.continuation, stochastic, probability);
            if cli.verbose {
                divider("Continuation");
                info!("  {}{}", source, cont);
            } else {
                println!("{}{}", source, cont);
            }
        }

        Summarize {
            input,
            text,
            sentences,
            iterations,
            depth,
            stochastic,
            probability,
        } => {
            banner("Summarize · Key Sentence Extraction", "✂️");
            let source = load_source(&input, text)?;
            info!("  📝 Input : {} chars", source.len());
            info!(
                "  📊 Extracting {} key sentences via GP scoring...",
                sentences
            );
            let summarizer = TextSummarizer::new(sentences, iterations, depth);
            match summarizer.summarize(&source) {
                Ok(summary) => {
                    if cli.verbose {
                        divider("Summary");
                    }
                    for (i, sentence) in summary.iter().enumerate() {
                        let enhanced = maybe_enhance(sentence, stochastic, probability);
                        if cli.verbose {
                            info!("  {}. {}", i + 1, enhanced);
                        } else {
                            println!("{}", enhanced);
                        }
                    }
                }
                Err(e) => {
                    error!("  ❌ Summarization failed: {}", e);
                }
            }
        }

        Sentence {
            input,
            text,
            iterations,
            depth,
            stochastic,
            probability,
        } => {
            banner("Sentence · Single Sentence Generation", "✍️");
            let source = load_source(&input, text)?;
            info!("  🌱 Seed : {:?}", source);
            let sentence_gen = SentenceGenerator::new(iterations, depth);
            match sentence_gen.generate(&source) {
                Ok(sentence) => {
                    let enhanced = maybe_enhance(&sentence, stochastic, probability);
                    if cli.verbose {
                        divider("Generated Sentence");
                        info!("  {}", enhanced);
                    } else {
                        println!("{}", enhanced);
                    }
                }
                Err(e) => {
                    error!("  ❌ Generation failed: {}", e);
                }
            }
        }

        Paragraph {
            input,
            text,
            sentences,
            iterations,
            depth,
            stochastic,
            probability,
        } => {
            banner("Paragraph · Generate a Paragraph", "📄");
            let source = load_source(&input, text)?;
            info!("  🌱 Seed      : {:?}", source);
            info!("  📊 Sentences : {}", sentences);
            let para_gen = ParagraphGenerator::new(sentences, iterations, depth);
            match para_gen.generate(&source) {
                Ok(paragraph) => {
                    let enhanced = maybe_enhance(&paragraph, stochastic, probability);
                    if cli.verbose {
                        divider("Generated Paragraph");
                        info!("");
                        info!("  {}", enhanced);
                    } else {
                        println!("{}", enhanced);
                    }
                }
                Err(e) => {
                    error!("  ❌ Generation failed: {}", e);
                }
            }
        }

        Essay {
            input,
            text,
            paragraphs,
            sentences,
            iterations,
            depth,
            stochastic,
            probability,
        } => {
            banner("Essay · Generate a Full Essay", "📖");
            let source = load_source(&input, text)?;
            info!("  🌱 Topic      : {:?}", source);
            info!(
                "  📊 Paragraphs : {} ({} sentences each)",
                paragraphs, sentences
            );
            let essay_gen = EssayGenerator::new(paragraphs, sentences, iterations, depth);
            match essay_gen.generate(&source) {
                Ok(essay) => {
                    if cli.verbose {
                        divider("Essay");
                        info!("");
                        info!("  ══════════════════════════════════════");
                        info!("  📖  {}", essay.title);
                        info!("  ══════════════════════════════════════");
                    } else {
                        println!(
                            "{}
",
                            essay.title
                        );
                    }
                    let count = essay.paragraphs.len();
                    for (i, para) in essay.paragraphs.iter().enumerate() {
                        let enhanced = maybe_enhance(para, stochastic, probability);
                        if cli.verbose {
                            let label = if i == 0 {
                                "Introduction".to_string()
                            } else if i == count - 1 {
                                "Conclusion".to_string()
                            } else {
                                format!("Body · §{}", i)
                            };
                            divider(&label);
                            info!("");
                            info!("  {}", enhanced);
                            info!("");
                        } else {
                            println!(
                                "{}
",
                                enhanced
                            );
                        }
                    }
                }
                Err(e) => {
                    error!("  ❌ Essay generation failed: {}", e);
                }
            }
        }

        #[cfg(feature = "net")]
        Ask {
            prompt,
            limit,
            sentences,
            region,
            iterations,
            depth,
            stochastic,
            probability,
        } => {
            banner("Ask · Internet-Aware Knowledge Synthesis", "🌐");
            info!("  ❓ Prompt : {:?}", prompt);
            let aggregator = crate::net::SearchAggregator::new().with_region(&region);
            divider("DuckDuckGo Results");

            if let Err(e) = aggregator.search_and_display(&prompt, limit).await {
                error!("  ⚠️  Search display failed: {}", e);
            }

            let corpus = match aggregator.get_response(&prompt).await {
                Ok(resp) => crate::net::corpus_from_response(&resp),
                Err(_) => String::new(),
            };

            let final_corpus = if corpus.trim().is_empty() {
                let lite_results = aggregator.fetch(&prompt, limit).await.unwrap_or_default();
                let quality = crate::net::corpus_from_results(&lite_results);
                if quality.trim().is_empty() {
                    crate::net::corpus_from_results_raw(&lite_results)
                } else {
                    quality
                }
            } else {
                corpus
            };

            info!("");
            divider("LMM Response");

            if final_corpus.trim().is_empty() {
                error!("  ❌ No extractable content from search results.");
            } else {
                let summarizer = TextSummarizer::new(sentences, iterations, depth);
                match summarizer.summarize_with_query(&final_corpus, &prompt) {
                    Ok(summary) => {
                        for sentence in &summary {
                            let enhanced = maybe_enhance(sentence, stochastic, probability);
                            if cli.verbose {
                                info!("  {}", enhanced);
                            } else {
                                println!("{}", enhanced);
                            }
                        }
                    }
                    Err(e) => error!("  ❌ Summarization failed: {}", e),
                }
            }
        }

        Imagen {
            prompt,
            width,
            height,
            components,
            style,
            palette,
            output,
        } => {
            banner("Imagen · Spectral Field Synthesis", "🎨");
            info!("  🖼  Prompt    : {:?}", prompt);
            info!("  📐 Dimensions : {}x{}", width, height);
            info!("  🎭 Style      : {}", style);
            info!("  🎨 Palette    : {}", palette);
            info!("  🌊 Components : {}", components);
            let parsed_style = style
                .parse::<crate::imagen::StyleMode>()
                .unwrap_or(crate::imagen::StyleMode::Plasma);
            let params = crate::imagen::ImagenParams {
                prompt,
                width,
                height,
                components,
                style: parsed_style,
                palette_name: palette,
                output: output.clone(),
            };
            info!("");
            divider("Rendering");
            match crate::imagen::render(&params) {
                Ok(path) => {
                    if cli.verbose {
                        info!("  ✅ Saved to: {}", path);
                    } else {
                        println!("{}", path);
                    }
                }
                Err(e) => error!("  ❌ Render failed: {}", e),
            }
        }
    }

    Ok(())
}
