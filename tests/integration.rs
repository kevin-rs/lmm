use lmm::causal::CausalGraph;
use lmm::compression::compute_mse;
use lmm::consciousness::Consciousness;
use lmm::discovery::SymbolicRegression;
use lmm::encode::{decode_from_parts, decode_message, encode_text};
use lmm::equation::Expression;
use lmm::field::Field;
use lmm::operator::NeuralOperator;
use lmm::physics::{HarmonicOscillator, LorenzSystem, SIRModel};
use lmm::predict::TextPredictor;
use lmm::simulation::Simulator;
use lmm::tensor::Tensor;
use lmm::text::{EssayGenerator, ParagraphGenerator, SentenceGenerator, TextSummarizer};
use lmm::traits::{Causal, Discoverable, Simulatable};
use lmm::world::WorldModel;
use std::collections::HashMap;

#[test]
fn test_tensor_math() {
    let t1 = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
    let t2 = Tensor::new(vec![2], vec![3.0, 4.0]).unwrap();
    let sum = (&t1 + &t2).unwrap();
    assert_eq!(sum.data, vec![4.0, 6.0]);
    let diff = (&t2 - &t1).unwrap();
    assert_eq!(diff.data, vec![2.0, 2.0]);
}

#[test]
fn test_simulation_rk4() {
    let osc = HarmonicOscillator::new(1.0, 1.0, 0.0).unwrap();
    let sim = Simulator { step_size: 0.01 };
    let s1 = sim.rk4_step(&osc, osc.state()).unwrap();
    assert!((s1.data[0] - 0.99995).abs() < 1e-4);
}

#[test]
fn test_simulate_trajectory() {
    let osc = HarmonicOscillator::new(1.0, 1.0, 0.0).unwrap();
    let sim = Simulator { step_size: 0.01 };
    let traj = sim.simulate_trajectory(&osc, osc.state(), 100).unwrap();
    assert_eq!(traj.len(), 101);
}

#[test]
fn test_equation_evaluation() {
    let eq = Expression::Add(
        Box::new(Expression::Variable("x".into())),
        Box::new(Expression::Constant(2.0)),
    );
    let mut vars = HashMap::new();
    vars.insert("x".into(), 5.0);
    assert_eq!(eq.evaluate(&vars).unwrap(), 7.0);
}

#[test]
fn test_equation_symbolic_diff_and_simplify() {
    let eq = Expression::Mul(
        Box::new(Expression::Constant(3.0)),
        Box::new(Expression::Variable("x".into())),
    );
    let d = eq.symbolic_diff("x").simplify();
    let mut vars = HashMap::new();
    vars.insert("x".into(), 99.0);
    assert!((d.evaluate(&vars).unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn test_symbolic_regression_pipeline() {
    let inputs: Vec<Vec<f64>> = (0..12).map(|i| vec![i as f64]).collect();
    let targets: Vec<f64> = (0..12).map(|i| 3.0 * i as f64).collect();
    let sr = SymbolicRegression::new(3, 30).with_variables(vec!["x".into()]);
    let expr = sr.fit(&inputs, &targets).unwrap();
    let mse = compute_mse(&expr, &inputs, &targets);
    let mean_y = targets.iter().sum::<f64>() / targets.len() as f64;
    let baseline =
        targets.iter().map(|&y| (y - mean_y).powi(2)).sum::<f64>() / targets.len() as f64;
    assert!(
        mse < baseline,
        "GP must beat constant baseline: mse={mse:.2}, baseline={baseline:.2}"
    );
}

#[test]
fn test_discover_trait() {
    let data: Vec<Tensor> = (0..6).map(|i| Tensor::from_vec(vec![i as f64])).collect();
    let targets: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let expr = SymbolicRegression::discover(&data, &targets).unwrap();
    assert!(expr.complexity() > 0);
}

#[test]
fn test_neural_operator_identity() {
    let mut kw = vec![0.0; 3];
    kw[1] = 1.0;
    let op = NeuralOperator { kernel_weights: kw };
    let field = Field::new(vec![3], Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap()).unwrap();
    let out = op.transform(&field).unwrap();
    for (a, b) in out.values.data.iter().zip(field.values.data.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn test_world_model_step() {
    let mut wm = WorldModel::new(Tensor::new(vec![2], vec![0.0, 0.0]).unwrap());
    let action = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
    let ns = wm.step(&action).unwrap();
    assert_eq!(ns.data, vec![1.0, 2.0]);
}

#[test]
fn test_world_model_prediction_error() {
    let mut wm = WorldModel::new(Tensor::new(vec![2], vec![0.0, 0.0]).unwrap());
    let predicted = Tensor::new(vec![2], vec![1.0, 1.0]).unwrap();
    let actual = Tensor::new(vec![2], vec![1.5, 0.5]).unwrap();
    let err = wm.record_error(&predicted, &actual).unwrap();
    assert!((err - 0.25).abs() < 1e-10);
    assert!((wm.mean_prediction_error() - 0.25).abs() < 1e-10);
}

#[test]
fn test_consciousness_tick() {
    let init = Tensor::zeros(vec![4]);
    let mut consc = Consciousness::new(init, 3, 0.01);
    let input = vec![255u8, 127, 0, 64];
    let ns = consc.tick(&input).unwrap();
    assert_eq!(ns.shape, vec![4]);
}

#[test]
fn test_causal_end_to_end() {
    let mut g = CausalGraph::new();
    let x = g.add_node("x", Some(Expression::Constant(4.0)));
    let y = g.add_node(
        "y",
        Some(Expression::Mul(
            Box::new(Expression::Constant(3.0)),
            Box::new(Expression::Variable("x".into())),
        )),
    );
    g.add_edge(x, y, 1.0).unwrap();
    let vals = g.forward_pass().unwrap();
    assert!((vals[&y] - 12.0).abs() < 1e-10);
    g.intervene("x", 10.0).unwrap();
    let after = g.forward_pass().unwrap();
    assert!((after[&y] - 30.0).abs() < 1e-10);
}

#[test]
fn test_lorenz_produces_3d_trajectory() {
    let sys = LorenzSystem::canonical().unwrap();
    let sim = Simulator { step_size: 0.01 };
    let traj = sim.simulate_trajectory(&sys, sys.state(), 100).unwrap();
    assert_eq!(traj.first().unwrap().shape, vec![3]);
    assert_eq!(traj.last().unwrap().shape, vec![3]);
}

#[test]
fn test_sir_full_pipeline() {
    let sir = SIRModel::new(0.3, 0.1, 990.0, 10.0, 0.0).unwrap();
    let n0 = sir.total_population();
    let sim = Simulator { step_size: 0.5 };
    let traj = sim.simulate_trajectory(&sir, sir.state(), 200).unwrap();
    let nf: f64 = traj.last().unwrap().data.iter().sum();
    assert!(
        (nf - n0).abs() < 20.0,
        "SIR population drift too large: {}",
        (nf - n0).abs()
    );
    let peak_i = traj.iter().map(|s| s.data[1]).fold(0.0f64, f64::max);
    assert!(peak_i > 10.0, "Infection should peak above initial");
}

#[test]
fn test_encode_decode_roundtrip() {
    let text = "Hi LMM";
    let encoded = encode_text(text, 30, 3).unwrap();
    let decoded = decode_message(&encoded).unwrap();
    assert_eq!(
        decoded, text,
        "Round-trip must recover original text exactly"
    );
}

#[test]
fn test_decode_from_parts_known() {
    let eq = Expression::Constant(65.0);
    let decoded = decode_from_parts(&eq, 3, &[0, 1, 2]).unwrap();
    assert_eq!(decoded.as_bytes(), &[65u8, 66, 67]);
}

#[test]
fn test_field_gradient_1d() {
    let data: Vec<f64> = (0..8).map(|i| (i as f64).powi(2)).collect();
    let tensor = Tensor::new(vec![8], data).unwrap();
    let field = Field::new(vec![8], tensor).unwrap();
    let grad = field.compute_gradient().unwrap();
    assert!((grad.values.data[4] - 8.0).abs() < 1.0);
}

#[test]
fn test_field_laplacian_1d() {
    let data: Vec<f64> = (0..8).map(|i| (i as f64).powi(2)).collect();
    let tensor = Tensor::new(vec![8], data).unwrap();
    let field = Field::new(vec![8], tensor).unwrap();
    let lap = field.compute_laplacian().unwrap();
    for v in &lap.values.data[1..7] {
        assert!((v - 2.0).abs() < 1e-6, "Expected ≈2 got {v}");
    }
}

#[test]
fn test_causal_intervention_pipeline() {
    let mut graph = CausalGraph::new();
    let x_id = graph.add_node("x", Some(Expression::Constant(3.0)));
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
    graph.add_edge(x_id, y_id, 1.0).unwrap();
    graph.add_edge(y_id, z_id, 1.0).unwrap();
    graph.nodes[x_id].observed_value = Some(3.0);
    let before = graph.forward_pass().unwrap();
    assert!((before[&y_id] - 6.0).abs() < 1e-10);
    assert!((before[&z_id] - 7.0).abs() < 1e-10);
    graph.intervene("x", 10.0).unwrap();
    let after = graph.forward_pass().unwrap();
    assert!((after[&y_id] - 20.0).abs() < 1e-10);
    assert!((after[&z_id] - 21.0).abs() < 1e-10);
}

#[test]
fn test_predict_produces_real_words() {
    let input = "Wise AI built the first LMM framework";
    let predictor = TextPredictor::new(10, 20, 3);
    let result = predictor.predict_continuation(input, 30).unwrap();
    let words: Vec<&str> = result.continuation.split_whitespace().collect();
    assert!(!words.is_empty(), "Continuation should not be empty");
    assert!(
        result.continuation.len() >= 20,
        "Continuation should be reasonably long"
    );
}

#[test]
fn test_predict_has_multiple_words() {
    let input = "Large Mathematical Models compress the world into equations";
    let predictor = TextPredictor::new(10, 20, 3);
    let result = predictor.predict_continuation(input, 40).unwrap();
    let word_count = result.continuation.split_whitespace().count();
    assert!(word_count >= 3, "Expected >=3 words, got {word_count}");
}

#[test]
fn test_predict_trajectory_equation_has_variable() {
    let input = "The Pharaohs encoded reality in mathematics";
    let predictor = TextPredictor::new(8, 20, 3);
    let result = predictor.predict_continuation(input, 20).unwrap();
    assert!(result.trajectory_equation.complexity() > 0);
    assert!(result.rhythm_equation.complexity() > 0);
}

#[test]
fn test_text_summarizer_outputs_subset() {
    let input = "Mathematical equations reveal the hidden structure of reality. \
                 Physical laws govern every observable phenomenon in the universe. \
                 Symbolic symmetry connects abstract algebra to concrete geometry. \
                 Entropy describes the irreversible flow of information over time. \
                 Simulation compresses the dynamics of complex systems into equations.";
    let summarizer = TextSummarizer::new(2, 20, 3);
    let summary = summarizer.summarize(input).unwrap();
    assert_eq!(summary.len(), 2);
    for s in &summary {
        assert!(
            input.contains(s.trim_end_matches('.')),
            "'{}' not in input",
            s
        );
    }
}

#[test]
fn test_sentence_generator_produces_punctuation() {
    let input = "Mathematical equations reveal the structure of reality";
    let sentence_gen = SentenceGenerator::new(20, 3);
    let sentence = sentence_gen.generate(input).unwrap();
    assert!(!sentence.is_empty());
    assert!(
        sentence.ends_with('.') || sentence.ends_with('!') || sentence.ends_with('?'),
        "Expected punctuation, got: {:?}",
        sentence
    );
    assert!(sentence.split_whitespace().count() >= 4);
}

#[test]
fn test_paragraph_generator_produces_multiple_sentences() {
    let input = "LMM encodes reality";
    let para_gen = ParagraphGenerator::new(3, 20, 3);
    let paragraph = para_gen.generate(input).unwrap();
    let sentence_count = paragraph
        .split(['.', '!', '?'])
        .filter(|s| !s.trim().is_empty())
        .count();
    assert!(sentence_count >= 2);
}

#[test]
fn test_essay_generator_has_intro_and_conclusion() {
    let input = "Mathematical models and the structure of reality";
    let essay_gen = EssayGenerator::new(2, 3, 20, 3);
    let essay = essay_gen.generate(input).unwrap();
    assert!(!essay.title.is_empty());
    assert_eq!(
        essay.paragraphs.len(),
        4,
        "Intro + 2 Body + Conclusion = 4 paragraphs"
    );
}
