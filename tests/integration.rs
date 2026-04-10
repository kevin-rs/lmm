use lmm::consciousness::Consciousness;
use lmm::discovery::SymbolicRegression;
use lmm::equation::Expression::{Add, Constant, Variable};
use lmm::field::Field;
use lmm::operator::NeuralOperator;
use lmm::simulation::Simulator;
use lmm::tensor::Tensor;
use lmm::traits::{Discoverable, Simulatable};
use lmm::world::WorldModel;
use std::collections::HashMap;

#[test]
fn test_tensor_math() {
    let t1 = Tensor::new(vec![2], vec![1.0, 2.0]).unwrap();
    let t2 = Tensor::new(vec![2], vec![3.0, 4.0]).unwrap();
    let sum = (&t1 + &t2).unwrap();
    assert_eq!(sum.data, vec![4.0, 6.0]);
}

struct HarmonicOscillator;

impl Simulatable for HarmonicOscillator {
    fn state(&self) -> &Tensor {
        unimplemented!()
    }

    fn evaluate_derivatives(&self, state: &Tensor) -> lmm::error::Result<Tensor> {
        let x = state.data[0];
        let v = state.data[1];
        Tensor::new(vec![2], vec![v, -x])
    }
}

#[test]
fn test_simulation() {
    let sim = Simulator { step_size: 0.01 };
    let model = HarmonicOscillator;
    let initial = Tensor::new(vec![2], vec![1.0, 0.0]).unwrap();
    let s1 = sim.rk4_step(&model, &initial).unwrap();
    assert!((s1.data[0] - 0.99995).abs() < 1e-4);
}

#[test]
fn test_equation_evaluation() {
    let eq = Add(Box::new(Variable("x".into())), Box::new(Constant(2.0)));
    let mut vars = HashMap::new();
    vars.insert("x".into(), 5.0);
    let res = eq.evaluate(&vars).unwrap();
    assert_eq!(res, 7.0);
}

#[test]
fn test_symbolic_regression() {
    let eq = SymbolicRegression::discover(&[]).unwrap();
    assert!(eq.complexity() > 0);
}

#[test]
fn test_neural_operator() {
    let field = Field::new(vec![3], Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap()).unwrap();
    let op = NeuralOperator {
        kernel_weights: vec![0.5],
    };
    let out = op.transform(&field).unwrap();
    assert_eq!(out.values.data, vec![0.5, 1.0, 1.5]);
}

#[test]
fn test_world_model() {
    let mut wm = WorldModel {
        current_state: Tensor::new(vec![1], vec![0.0]).unwrap(),
    };
    let a = Tensor::new(vec![1], vec![1.0]).unwrap();
    let ns = wm.step(&a).unwrap();
    assert_eq!(ns.data[0], 1.0);
}

#[test]
fn test_perception_consciousness() {
    let mut consc = Consciousness {
        world_model: WorldModel {
            current_state: Tensor::new(vec![4], vec![0.0, 0.0, 0.0, 0.0]).unwrap(),
        },
    };
    let input = vec![255, 127, 0, 64];
    let ns = consc.tick(&input).unwrap();
    assert_eq!(ns.shape, vec![4]);
}
