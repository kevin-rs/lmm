#![cfg_attr(target_arch = "wasm32", allow(dead_code))]
use crate::error::{LmmError, Result};

use std::f64::consts::TAU;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{BufWriter, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

const LCG_MULTIPLIER: u64 = 6364136223846793005;
const LCG_INCREMENT: u64 = 1442695040888963407;
const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;
const PROMPT_INDEX_MULTIPLIER: u64 = 31;

const WAVE_BAND_STRIDE: u64 = 997;
const WAVE_COMPONENT_STRIDE: u64 = 31337;
const MIN_WAVE_COMPONENTS: usize = 3;
const MAX_WAVE_COMPONENTS: usize = 32;
const FRACTAL_MAX_ITERATIONS: u32 = 32;
const FRACTAL_ESCAPE_RADIUS_SQ: f64 = 4.0;
const STYLE_SIGMA_SEED_OFFSET: u64 = 99;
const FRACTAL_C_REAL_SEED_OFFSET: u64 = 77;
const FRACTAL_C_IMAG_SEED_OFFSET: u64 = 78;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StyleMode {
    Wave,
    Radial,
    Orbital,
    Fractal,
    Flow,
    Plasma,
}

impl std::str::FromStr for StyleMode {
    type Err = LmmError;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "wave" => Ok(StyleMode::Wave),
            "radial" => Ok(StyleMode::Radial),
            "orbital" => Ok(StyleMode::Orbital),
            "fractal" => Ok(StyleMode::Fractal),
            "flow" => Ok(StyleMode::Flow),
            "plasma" => Ok(StyleMode::Plasma),
            _ => Err(LmmError::Perception(format!("unknown style: {}", s))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Palette {
    pub r_bias: f64,
    pub g_bias: f64,
    pub b_bias: f64,
    pub r_amp: f64,
    pub g_amp: f64,
    pub b_amp: f64,
}

impl Palette {
    fn from_seed(seed: u64) -> Self {
        let r_bias = lcg_unit(seed) * 0.5 + 0.25;
        let g_bias = lcg_unit(seed.wrapping_add(1)) * 0.5 + 0.25;
        let b_bias = lcg_unit(seed.wrapping_add(2)) * 0.5 + 0.25;
        let r_amp = lcg_unit(seed.wrapping_add(3)) * 0.4 + 0.3;
        let g_amp = lcg_unit(seed.wrapping_add(4)) * 0.4 + 0.3;
        let b_amp = lcg_unit(seed.wrapping_add(5)) * 0.4 + 0.3;
        Self {
            r_bias,
            g_bias,
            b_bias,
            r_amp,
            g_amp,
            b_amp,
        }
    }

    pub fn warm() -> Self {
        Self {
            r_bias: 0.7,
            g_bias: 0.3,
            b_bias: 0.1,
            r_amp: 0.3,
            g_amp: 0.2,
            b_amp: 0.15,
        }
    }

    pub fn cool() -> Self {
        Self {
            r_bias: 0.1,
            g_bias: 0.3,
            b_bias: 0.7,
            r_amp: 0.15,
            g_amp: 0.2,
            b_amp: 0.3,
        }
    }

    pub fn neon() -> Self {
        Self {
            r_bias: 0.1,
            g_bias: 0.9,
            b_bias: 0.5,
            r_amp: 0.5,
            g_amp: 0.5,
            b_amp: 0.5,
        }
    }

    pub fn monochrome() -> Self {
        Self {
            r_bias: 0.5,
            g_bias: 0.5,
            b_bias: 0.5,
            r_amp: 0.45,
            g_amp: 0.45,
            b_amp: 0.45,
        }
    }
}

fn lcg_unit(seed: u64) -> f64 {
    let x = seed
        .wrapping_mul(LCG_MULTIPLIER)
        .wrapping_add(LCG_INCREMENT);
    (x >> 33) as f64 / (u32::MAX as f64)
}

fn prompt_seed(text: &str) -> u64 {
    text.bytes()
        .enumerate()
        .fold(FNV_OFFSET_BASIS, |acc, (i, b)| {
            acc.wrapping_mul(FNV_PRIME)
                .wrapping_add(b as u64)
                .wrapping_add(i as u64 * PROMPT_INDEX_MULTIPLIER)
        })
}

#[derive(Debug, Clone)]
struct WaveComponent {
    amplitude: f64,
    freq_x: f64,
    freq_y: f64,
    phase: f64,
}

impl WaveComponent {
    fn from_seed(seed: u64, band: u64, component: u64) -> Self {
        let s = seed
            .wrapping_add(band * WAVE_BAND_STRIDE)
            .wrapping_add(component * WAVE_COMPONENT_STRIDE);
        let amplitude = lcg_unit(s) * 0.6 + 0.1;
        let freq_x = lcg_unit(s.wrapping_add(1)) * 5.0 + 0.5;
        let freq_y = lcg_unit(s.wrapping_add(2)) * 5.0 + 0.5;
        let phase = lcg_unit(s.wrapping_add(3)) * TAU;
        Self {
            amplitude,
            freq_x,
            freq_y,
            phase,
        }
    }

    fn evaluate(&self, nx: f64, ny: f64) -> f64 {
        self.amplitude * (TAU * self.freq_x * nx + TAU * self.freq_y * ny + self.phase).cos()
    }
}

fn spectral_field(components: &[WaveComponent], nx: f64, ny: f64) -> f64 {
    let raw: f64 = components.iter().map(|c| c.evaluate(nx, ny)).sum();
    let norm = raw / components.len() as f64;
    (norm + 1.0) * 0.5
}

fn apply_style(nx: f64, ny: f64, base: f64, style: StyleMode, seed: u64) -> f64 {
    let cx = nx - 0.5;
    let cy = ny - 0.5;
    let r = (cx * cx + cy * cy).sqrt();
    let theta = cy.atan2(cx);
    let sigma = lcg_unit(seed.wrapping_add(STYLE_SIGMA_SEED_OFFSET)) * 0.3 + 0.2;

    match style {
        StyleMode::Wave => base,
        StyleMode::Radial => {
            let radial_mod = (TAU * r * 4.0 + base * TAU).cos() * 0.5 + 0.5;
            (base + radial_mod) * 0.5
        }
        StyleMode::Orbital => {
            let envelope = (-r * r / (2.0 * sigma * sigma)).exp();
            let angular = (theta * 3.0 + base * TAU).cos() * 0.5 + 0.5;
            envelope * angular + (1.0 - envelope) * base
        }
        StyleMode::Fractal => {
            let mut z_r = cx * 3.0 + base;
            let mut z_i = cy * 3.0;
            let c_r = lcg_unit(seed.wrapping_add(FRACTAL_C_REAL_SEED_OFFSET)) * 2.0 - 1.0;
            let c_i = lcg_unit(seed.wrapping_add(FRACTAL_C_IMAG_SEED_OFFSET)) * 2.0 - 1.0;
            let mut iter = 0u32;
            while iter < FRACTAL_MAX_ITERATIONS && z_r * z_r + z_i * z_i < FRACTAL_ESCAPE_RADIUS_SQ
            {
                let tmp = z_r * z_r - z_i * z_i + c_r;
                z_i = 2.0 * z_r * z_i + c_i;
                z_r = tmp;
                iter += 1;
            }
            iter as f64 / FRACTAL_MAX_ITERATIONS as f64
        }
        StyleMode::Flow => {
            let stream_x = (TAU * ny * 2.0 + base).sin();
            let stream_y = (TAU * nx * 2.0 + base).cos();
            let dot = cx * stream_x + cy * stream_y;
            (dot * 0.5 + 0.5).clamp(0.0, 1.0)
        }
        StyleMode::Plasma => {
            let v1 = (TAU * (nx + base)).sin();
            let v2 = (TAU * (ny + base * 0.7)).sin();
            let v3 = (TAU * (nx + ny + base * 0.3) * 0.5).sin();
            let v4 = {
                let dx = nx - 0.5 + (base * TAU).cos() * 0.25;
                let dy = ny - 0.5 + (base * TAU).sin() * 0.25;
                (TAU * (dx * dx + dy * dy).sqrt() * 4.0).sin()
            };
            ((v1 + v2 + v3 + v4) * 0.25 + 1.0) * 0.5
        }
    }
}

fn field_to_rgb(r_field: f64, g_field: f64, b_field: f64, palette: &Palette) -> [u8; 3] {
    let map = |field: f64, bias: f64, amp: f64| -> u8 {
        let v = (bias + amp * (field * TAU).sin()).clamp(0.0, 1.0);
        (v * 255.0).round() as u8
    };
    [
        map(r_field, palette.r_bias, palette.r_amp),
        map(g_field, palette.g_bias, palette.g_amp),
        map(b_field, palette.b_bias, palette.b_amp),
    ]
}

pub struct ImagenParams {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub components: usize,
    pub style: StyleMode,
    pub palette_name: String,
    pub output: String,
}

#[cfg(not(target_arch = "wasm32"))]
pub fn render(params: &ImagenParams) -> Result<String> {
    let seed = prompt_seed(&params.prompt);

    let palette = match params.palette_name.to_lowercase().as_str() {
        "warm" => Palette::warm(),
        "cool" => Palette::cool(),
        "neon" => Palette::neon(),
        "monochrome" | "mono" => Palette::monochrome(),
        _ => Palette::from_seed(seed),
    };

    let n = params
        .components
        .clamp(MIN_WAVE_COMPONENTS, MAX_WAVE_COMPONENTS);
    let red_waves: Vec<WaveComponent> = (0..n)
        .map(|k| WaveComponent::from_seed(seed, 0, k as u64))
        .collect();
    let green_waves: Vec<WaveComponent> = (0..n)
        .map(|k| WaveComponent::from_seed(seed, 1, k as u64))
        .collect();
    let blue_waves: Vec<WaveComponent> = (0..n)
        .map(|k| WaveComponent::from_seed(seed, 2, k as u64))
        .collect();

    let w = params.width as usize;
    let h = params.height as usize;
    let mut pixels: Vec<u8> = Vec::with_capacity(w * h * 3);

    for py in 0..h {
        let ny = py as f64 / (h - 1).max(1) as f64;
        for px in 0..w {
            let nx = px as f64 / (w - 1).max(1) as f64;
            let r_raw = spectral_field(&red_waves, nx, ny);
            let g_raw = spectral_field(&green_waves, nx, ny);
            let b_raw = spectral_field(&blue_waves, nx, ny);
            let r_val = apply_style(nx, ny, r_raw, params.style, seed.wrapping_add(0));
            let g_val = apply_style(nx, ny, g_raw, params.style, seed.wrapping_add(1));
            let b_val = apply_style(nx, ny, b_raw, params.style, seed.wrapping_add(2));
            let rgb = field_to_rgb(r_val, g_val, b_val, &palette);
            pixels.extend_from_slice(&rgb);
        }
    }

    let mut out_path = std::path::PathBuf::from(&params.output);
    if params.output.ends_with('/') || params.output.ends_with('\\') || out_path.is_dir() {
        if !out_path.exists() {
            std::fs::create_dir_all(&out_path)
                .map_err(|e| LmmError::Perception(format!("cannot create directory: {}", e)))?;
        }
        let seed_hex = format!("{:08x}", seed as u32);
        let style_str = format!("{:?}", params.style).to_lowercase();
        out_path.push(format!(
            "{}_{}_{}.ppm",
            style_str, params.palette_name, seed_hex
        ));
    } else if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| LmmError::Perception(format!("cannot create directory: {}", e)))?;
    }

    write_ppm(&out_path, w, h, &pixels)?;
    Ok(out_path.to_string_lossy().into_owned())
}

#[cfg(not(target_arch = "wasm32"))]
fn write_ppm(path: &Path, width: usize, height: usize, pixels: &[u8]) -> Result<()> {
    let file = File::create(path)
        .map_err(|e| LmmError::Perception(format!("cannot create output file: {}", e)))?;
    let mut writer = BufWriter::new(file);
    write!(writer, "P6\n{} {}\n255\n", width, height)
        .map_err(|e| LmmError::Perception(format!("write error: {}", e)))?;
    writer
        .write_all(pixels)
        .map_err(|e| LmmError::Perception(format!("write error: {}", e)))?;
    Ok(())
}
