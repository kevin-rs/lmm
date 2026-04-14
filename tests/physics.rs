use lmm::physics::{HarmonicOscillator, LorenzSystem, Pendulum, SIRModel};
use lmm::simulation::Simulator;
use lmm::traits::Simulatable;

#[test]
fn test_harmonic_energy_conservation() {
    let osc = HarmonicOscillator::new(1.0, 1.0, 0.0).unwrap();
    let initial_energy = osc.energy();
    let sim = Simulator { step_size: 0.01 };
    let traj = sim.simulate_trajectory(&osc, osc.state(), 1000).unwrap();
    let final_state = traj.last().unwrap();
    let x = final_state.data[0];
    let v = final_state.data[1];
    let final_energy = 0.5 * v * v + 0.5 * x * x;
    assert!(
        (final_energy - initial_energy).abs() < 1e-3,
        "Energy drift={}",
        (final_energy - initial_energy).abs()
    );
}

#[test]
fn test_harmonic_derivatives() {
    let osc = HarmonicOscillator::new(2.0, 3.0, 1.0).unwrap();
    let s = osc.state().clone();
    let d = osc.evaluate_derivatives(&s).unwrap();
    assert!((d.data[0] - 1.0).abs() < 1e-10);
    assert!((d.data[1] - (-4.0 * 3.0)).abs() < 1e-10);
}

#[test]
fn test_lorenz_diverges_from_nearby() {
    let sys1 = LorenzSystem::new(10.0, 28.0, 8.0 / 3.0, 0.0, 1.0, 1.0).unwrap();
    let sys2 = LorenzSystem::new(10.0, 28.0, 8.0 / 3.0, 0.5, 1.0, 1.0).unwrap();
    let sim = Simulator { step_size: 0.01 };
    let t1 = sim.simulate_trajectory(&sys1, sys1.state(), 800).unwrap();
    let t2 = sim.simulate_trajectory(&sys2, sys2.state(), 800).unwrap();
    let max_dist = t1
        .iter()
        .zip(t2.iter())
        .map(|(s1, s2)| {
            ((s1.data[0] - s2.data[0]).powi(2)
                + (s1.data[1] - s2.data[1]).powi(2)
                + (s1.data[2] - s2.data[2]).powi(2))
            .sqrt()
        })
        .fold(0.0f64, f64::max);
    assert!(
        max_dist > 1.0,
        "Lorenz butterfly effect: max_dist={max_dist}"
    );
}

#[test]
fn test_pendulum_energy_approx_conservation() {
    let pend = Pendulum::new(9.81, 1.0, 0.1, 0.0).unwrap();
    let e0 = pend.energy();
    let sim = Simulator { step_size: 0.001 };
    let traj = sim.simulate_trajectory(&pend, pend.state(), 2000).unwrap();
    let s = traj.last().unwrap();
    let theta = s.data[0];
    let omega = s.data[1];
    let ef = 0.5 * 1.0 * omega * omega + 9.81 * 1.0 * (1.0 - theta.cos());
    assert!(
        (ef - e0).abs() < 0.05,
        "Pendulum energy drift={}",
        (ef - e0).abs()
    );
}

#[test]
fn test_sir_population_conserved() {
    let sir = SIRModel::new(0.3, 0.1, 990.0, 10.0, 0.0).unwrap();
    let n0 = sir.total_population();
    let sim = Simulator { step_size: 0.1 };
    let traj = sim.simulate_trajectory(&sir, sir.state(), 500).unwrap();
    let s = traj.last().unwrap();
    let nf = s.data.iter().sum::<f64>();
    assert!(
        (nf - n0).abs() < 5.0,
        "SIR population drift: |{} - {}| = {}",
        nf,
        n0,
        (nf - n0).abs()
    );
}

#[test]
fn test_sir_infection_peaks_and_declines() {
    let sir = SIRModel::new(0.3, 0.05, 990.0, 10.0, 0.0).unwrap();
    let sim = Simulator { step_size: 0.1 };
    let traj = sim.simulate_trajectory(&sir, sir.state(), 2000).unwrap();
    let peak = traj.iter().map(|s| s.data[1]).fold(0.0f64, f64::max);
    let final_i = traj.last().unwrap().data[1];
    assert!(peak > 10.0, "Infection should grow above initial");
    assert!(final_i < peak, "Infection should decline after peak");
}

#[test]
fn test_simulator_rk4_harmonic() {
    let osc = HarmonicOscillator::new(1.0, 1.0, 0.0).unwrap();
    let sim = Simulator { step_size: 0.01 };
    let next = sim.rk4_step(&osc, osc.state()).unwrap();
    assert!((next.data[0] - 0.99995).abs() < 1e-4);
}

#[test]
fn test_simulate_trajectory_length() {
    let osc = HarmonicOscillator::new(1.0, 1.0, 0.0).unwrap();
    let sim = Simulator { step_size: 0.01 };
    let traj = sim.simulate_trajectory(&osc, osc.state(), 50).unwrap();
    assert_eq!(traj.len(), 51);
}
