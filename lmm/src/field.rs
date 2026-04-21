// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Tensor Field Operations
//!
//! This module provides [`Field`], an n-dimensional discretised scalar or vector field
//! backed by a [`Tensor`]. It implements standard differential operators used in physics
//! and field theory:
//!
//! | Operator | Dimensions | Method |
//! |---|---|---|
//! | Gradient | 1-D, 2-D, 3-D | [`Field::compute_gradient`] |
//! | Laplacian | 1-D, 2-D, 3-D | [`Field::compute_laplacian`] |
//! | Divergence | N-D (components) | [`Field::compute_divergence`] |
//! | Curl | 3-D (3 components) | [`Field::compute_curl`] |
//!
//! All operators use **central-difference** finite-difference stencils at interior points
//! and **forward/backward** differences at boundary points.

use crate::error::LmmError::Simulation;
use crate::error::Result;
use crate::tensor::Tensor;

/// A discretised field defined over an n-dimensional rectangular grid.
///
/// `dimensions` specifies the size along each axis; `values` is the flat [`Tensor`]
/// storing scalar field values in row-major (C) order.
///
/// # Examples
///
/// ```
/// use lmm::traits::Simulatable;
/// use lmm::field::Field;
/// use lmm::tensor::Tensor;
///
/// let t = Tensor::new(vec![5], vec![0.0, 1.0, 4.0, 9.0, 16.0]).unwrap();
/// let f = Field::new(vec![5], t).unwrap();
/// let grad = f.compute_gradient().unwrap();
/// // Central differences: grad[2] ≈ (9-1)/2 = 4
/// assert!((grad.values.data[2] - 4.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    /// Sizes of each dimension of the grid.
    pub dimensions: Vec<usize>,
    /// Scalar values at each grid point, in row-major order.
    pub values: Tensor,
}

impl Field {
    /// Creates a new [`Field`], validating that `dimensions` matches the tensor shape.
    ///
    /// # Arguments
    ///
    /// * `dimensions` - Expected sizes of each dimension.
    /// * `values` - The underlying data tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Simulation`] when `dimensions != values.shape`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::field::Field;
    /// use lmm::tensor::Tensor;
    ///
    /// let t = Tensor::zeros(vec![3, 3]);
    /// assert!(Field::new(vec![3, 3], t).is_ok());
    /// ```
    pub fn new(dimensions: Vec<usize>, values: Tensor) -> Result<Self> {
        if dimensions != values.shape {
            return Err(Simulation(format!(
                "Field bounds mismatch: dimensions {:?} vs tensor shape {:?}",
                dimensions, values.shape
            )));
        }
        Ok(Self { dimensions, values })
    }

    /// Computes the **gradient** of the scalar field using finite differences.
    ///
    /// For 1-D fields, returns `∂f/∂x`. For 2-D fields, returns the gradient magnitude
    /// `√((∂f/∂x)² + (∂f/∂y)²)`. For 3-D fields, returns the gradient magnitude
    /// `√((∂f/∂x)² + (∂f/∂y)² + (∂f/∂z)²)`.
    ///
    /// # Returns
    ///
    /// (`Result<Field>`): The gradient field.
    ///
    /// # Errors
    ///
    /// Returns [`Simulation`] for fields with more than 3 dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::field::Field;
    /// use lmm::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![4], vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    /// let f = Field::new(vec![4], t).unwrap();
    /// let g = f.compute_gradient().unwrap();
    /// // Interior: (4-0)/2 = 2; (9-1)/2 = 4
    /// assert!((g.values.data[1] - 2.0).abs() < 1e-9);
    /// ```
    pub fn compute_gradient(&self) -> Result<Field> {
        match self.dimensions.len() {
            1 => self.gradient_1d(),
            2 => self.gradient_2d(),
            3 => self.gradient_3d(),
            n => Err(Simulation(format!(
                "Gradient unsupported for {n}-D fields (max 3-D)"
            ))),
        }
    }

    fn gradient_1d(&self) -> Result<Field> {
        let n = self.dimensions[0];
        let mut grad = vec![0.0; n];
        for (i, slot) in grad
            .iter_mut()
            .enumerate()
            .take(n.saturating_sub(1))
            .skip(1)
        {
            *slot = (self.values.data[i + 1] - self.values.data[i - 1]) / 2.0;
        }
        if n >= 2 {
            grad[0] = self.values.data[1] - self.values.data[0];
            grad[n - 1] = self.values.data[n - 1] - self.values.data[n - 2];
        }
        let tensor = Tensor::new(self.dimensions.clone(), grad)?;
        Ok(Self {
            dimensions: self.dimensions.clone(),
            values: tensor,
        })
    }

    fn gradient_2d(&self) -> Result<Field> {
        let rows = self.dimensions[0];
        let cols = self.dimensions[1];
        let mut grad_x = vec![0.0; rows * cols];
        let mut grad_y = vec![0.0; rows * cols];

        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                grad_x[idx] = if c + 1 < cols && c > 0 {
                    (self.values.data[r * cols + c + 1] - self.values.data[r * cols + c - 1]) / 2.0
                } else if c + 1 < cols {
                    self.values.data[r * cols + c + 1] - self.values.data[idx]
                } else {
                    self.values.data[idx] - self.values.data[r * cols + c - 1]
                };
                grad_y[idx] = if r + 1 < rows && r > 0 {
                    (self.values.data[(r + 1) * cols + c] - self.values.data[(r - 1) * cols + c])
                        / 2.0
                } else if r + 1 < rows {
                    self.values.data[(r + 1) * cols + c] - self.values.data[idx]
                } else {
                    self.values.data[idx] - self.values.data[(r - 1) * cols + c]
                };
            }
        }

        let total: Vec<f64> = grad_x
            .iter()
            .zip(grad_y.iter())
            .map(|(x, y)| (x * x + y * y).sqrt())
            .collect();
        let tensor = Tensor::new(self.dimensions.clone(), total)?;
        Ok(Self {
            dimensions: self.dimensions.clone(),
            values: tensor,
        })
    }

    fn gradient_3d(&self) -> Result<Field> {
        let d0 = self.dimensions[0];
        let d1 = self.dimensions[1];
        let d2 = self.dimensions[2];
        let total_len = d0 * d1 * d2;
        let mut mag = vec![0.0; total_len];

        for i in 0..d0 {
            for j in 0..d1 {
                for k in 0..d2 {
                    let idx = i * d1 * d2 + j * d2 + k;
                    let gx = if k + 1 < d2 && k > 0 {
                        (self.values.data[i * d1 * d2 + j * d2 + k + 1]
                            - self.values.data[i * d1 * d2 + j * d2 + k - 1])
                            / 2.0
                    } else {
                        0.0
                    };
                    let gy = if j + 1 < d1 && j > 0 {
                        (self.values.data[i * d1 * d2 + (j + 1) * d2 + k]
                            - self.values.data[i * d1 * d2 + (j - 1) * d2 + k])
                            / 2.0
                    } else {
                        0.0
                    };
                    let gz = if i + 1 < d0 && i > 0 {
                        (self.values.data[(i + 1) * d1 * d2 + j * d2 + k]
                            - self.values.data[(i - 1) * d1 * d2 + j * d2 + k])
                            / 2.0
                    } else {
                        0.0
                    };
                    mag[idx] = (gx * gx + gy * gy + gz * gz).sqrt();
                }
            }
        }
        let tensor = Tensor::new(self.dimensions.clone(), mag)?;
        Ok(Self {
            dimensions: self.dimensions.clone(),
            values: tensor,
        })
    }

    /// Computes the **Laplacian** (∇²f) of the scalar field using finite differences.
    ///
    /// Supported for 1-D, 2-D, and 3-D fields. The Laplacian is the divergence of the
    /// gradient and measures local curvature.
    ///
    /// # Returns
    ///
    /// (`Result<Field>`): The Laplacian field.
    ///
    /// # Errors
    ///
    /// Returns [`Simulation`] for fields with more than 3 dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::field::Field;
    /// use lmm::tensor::Tensor;
    ///
    /// // f(x) = x², ∇²f = 2 everywhere (up to boundary effects)
    /// let t = Tensor::new(vec![5], vec![0.0, 1.0, 4.0, 9.0, 16.0]).unwrap();
    /// let f = Field::new(vec![5], t).unwrap();
    /// let lap = f.compute_laplacian().unwrap();
    /// // Interior: f[i+1] - 2f[i] + f[i-1] = (i+1)² - 2i² + (i-1)² = 2
    /// assert!((lap.values.data[2] - 2.0).abs() < 1e-9);
    /// ```
    pub fn compute_laplacian(&self) -> Result<Field> {
        match self.dimensions.len() {
            1 => self.laplacian_1d(),
            2 => self.laplacian_2d(),
            3 => self.laplacian_3d(),
            n => Err(Simulation(format!(
                "Laplacian unsupported for {n}-D fields (max 3-D)"
            ))),
        }
    }

    fn laplacian_1d(&self) -> Result<Field> {
        let n = self.dimensions[0];
        let mut lap = vec![0.0; n];
        for (i, slot) in lap.iter_mut().enumerate().take(n.saturating_sub(1)).skip(1) {
            *slot = self.values.data[i + 1] - 2.0 * self.values.data[i] + self.values.data[i - 1];
        }
        let tensor = Tensor::new(self.dimensions.clone(), lap)?;
        Ok(Self {
            dimensions: self.dimensions.clone(),
            values: tensor,
        })
    }

    fn laplacian_2d(&self) -> Result<Field> {
        let rows = self.dimensions[0];
        let cols = self.dimensions[1];
        let mut lap = vec![0.0; rows * cols];
        for r in 1..rows.saturating_sub(1) {
            for c in 1..cols.saturating_sub(1) {
                let idx = r * cols + c;
                let d2x = self.values.data[r * cols + c + 1] - 2.0 * self.values.data[idx]
                    + self.values.data[r * cols + c - 1];
                let d2y = self.values.data[(r + 1) * cols + c] - 2.0 * self.values.data[idx]
                    + self.values.data[(r - 1) * cols + c];
                lap[idx] = d2x + d2y;
            }
        }
        let tensor = Tensor::new(self.dimensions.clone(), lap)?;
        Ok(Self {
            dimensions: self.dimensions.clone(),
            values: tensor,
        })
    }

    /// 3-D Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
    fn laplacian_3d(&self) -> Result<Field> {
        let d0 = self.dimensions[0];
        let d1 = self.dimensions[1];
        let d2 = self.dimensions[2];
        let mut lap = vec![0.0; d0 * d1 * d2];

        for i in 1..d0.saturating_sub(1) {
            for j in 1..d1.saturating_sub(1) {
                for k in 1..d2.saturating_sub(1) {
                    let idx = i * d1 * d2 + j * d2 + k;
                    let d2z = self.values.data[i * d1 * d2 + j * d2 + k + 1]
                        - 2.0 * self.values.data[idx]
                        + self.values.data[i * d1 * d2 + j * d2 + k - 1];
                    let d2y = self.values.data[i * d1 * d2 + (j + 1) * d2 + k]
                        - 2.0 * self.values.data[idx]
                        + self.values.data[i * d1 * d2 + (j - 1) * d2 + k];
                    let d2x = self.values.data[(i + 1) * d1 * d2 + j * d2 + k]
                        - 2.0 * self.values.data[idx]
                        + self.values.data[(i - 1) * d1 * d2 + j * d2 + k];
                    lap[idx] = d2x + d2y + d2z;
                }
            }
        }
        let tensor = Tensor::new(self.dimensions.clone(), lap)?;
        Ok(Self {
            dimensions: self.dimensions.clone(),
            values: tensor,
        })
    }

    /// Computes the **divergence** of a vector field given its component fields.
    ///
    /// For a vector field `F = (F₀, F₁, ...)`, the divergence is
    /// `∇·F = ∂F₀/∂x₀ + ∂F₁/∂x₁ + ...`.
    ///
    /// # Arguments
    ///
    /// * `fields` - Slice of component fields; one per spatial dimension.
    ///
    /// # Returns
    ///
    /// (`Result<Field>`): The scalar divergence field.
    ///
    /// # Errors
    ///
    /// Returns [`Simulation`] when `fields` is empty or component shapes differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::field::Field;
    /// use lmm::tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![3], vec![0.0, 1.0, 2.0]).unwrap();
    /// let f = Field::new(vec![3], t).unwrap();
    /// let div = Field::compute_divergence(&[f]).unwrap();
    /// assert_eq!(div.dimensions, vec![3]);
    /// ```
    pub fn compute_divergence(fields: &[Field]) -> Result<Field> {
        let n_fields = fields.len();
        if n_fields == 0 {
            return Err(Simulation(
                "Divergence requires at least one field component".into(),
            ));
        }
        let dims = &fields[0].dimensions;
        for f in fields.iter().skip(1) {
            if &f.dimensions != dims {
                return Err(Simulation(
                    "Divergence: all field components must share dimensions".into(),
                ));
            }
        }

        if n_fields == 1 && dims.len() == 1 {
            return fields[0].gradient_1d();
        }

        let n = dims.iter().product();
        let mut div = vec![0.0; n];

        if !dims.is_empty() {
            let cols = *dims.last().unwrap_or(&1);
            for (i, slot) in div.iter_mut().enumerate() {
                let c = i % cols;
                if c > 0 && c + 1 < cols {
                    *slot += (fields[0].values.data[i + 1] - fields[0].values.data[i - 1]) / 2.0;
                }
            }
        }

        if n_fields >= 2 && dims.len() >= 2 {
            let cols = dims[dims.len() - 1];
            let rows = dims[dims.len() - 2];
            let stride = cols;
            for (i, slot) in div.iter_mut().enumerate() {
                let r = (i / stride) % rows;
                if r > 0 && r + 1 < rows {
                    *slot += (fields[1].values.data[i + stride]
                        - fields[1].values.data[i - stride])
                        / 2.0;
                }
            }
        }

        let tensor = Tensor::new(dims.clone(), div)?;
        Ok(Field {
            dimensions: dims.clone(),
            values: tensor,
        })
    }

    /// Computes the **curl** of a 3-D vector field `(Fx, Fy, Fz)`.
    ///
    /// Returns three component fields: `[curl_x, curl_y, curl_z]` where:
    /// - `curl_x = ∂Fz/∂y - ∂Fy/∂z`
    /// - `curl_y = ∂Fx/∂z - ∂Fz/∂x`
    /// - `curl_z = ∂Fy/∂x - ∂Fx/∂y`
    ///
    /// Boundary points are set to zero (no periodic extension).
    ///
    /// # Arguments
    ///
    /// * `fx`, `fy`, `fz` - The three component fields of the vector field.
    ///
    /// # Returns
    ///
    /// (`Result<[Field; 3]>`): `[curl_x, curl_y, curl_z]`.
    ///
    /// # Errors
    ///
    /// Returns [`Simulation`] when any field is not 3-D or shapes differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmm::field::Field;
    /// use lmm::tensor::Tensor;
    ///
    /// let t = Tensor::zeros(vec![3, 3, 3]);
    /// let f = Field::new(vec![3, 3, 3], t).unwrap();
    /// let [cx, cy, cz] = Field::compute_curl(&f, &f.clone(), &f.clone()).unwrap();
    /// assert_eq!(cx.dimensions, vec![3, 3, 3]);
    /// ```
    pub fn compute_curl(fx: &Field, fy: &Field, fz: &Field) -> Result<[Field; 3]> {
        if fx.dimensions.len() != 3 || fy.dimensions.len() != 3 || fz.dimensions.len() != 3 {
            return Err(Simulation("Curl requires 3-D fields".into()));
        }
        let dims = &fx.dimensions;
        if &fy.dimensions != dims || &fz.dimensions != dims {
            return Err(Simulation(
                "Curl: all field components must share dimensions".into(),
            ));
        }

        let d0 = dims[0];
        let d1 = dims[1];
        let d2 = dims[2];
        let n = d0 * d1 * d2;
        let mut curl_x = vec![0.0; n];
        let mut curl_y = vec![0.0; n];
        let mut curl_z = vec![0.0; n];

        for i in 1..d0.saturating_sub(1) {
            for j in 1..d1.saturating_sub(1) {
                for k in 1..d2.saturating_sub(1) {
                    let idx = i * d1 * d2 + j * d2 + k;
                    let dfz_dy = (fz.values.data[i * d1 * d2 + (j + 1) * d2 + k]
                        - fz.values.data[i * d1 * d2 + (j - 1) * d2 + k])
                        / 2.0;
                    let dfy_dz = (fy.values.data[i * d1 * d2 + j * d2 + k + 1]
                        - fy.values.data[i * d1 * d2 + j * d2 + k - 1])
                        / 2.0;
                    let dfx_dz = (fx.values.data[i * d1 * d2 + j * d2 + k + 1]
                        - fx.values.data[i * d1 * d2 + j * d2 + k - 1])
                        / 2.0;
                    let dfz_dx = (fz.values.data[(i + 1) * d1 * d2 + j * d2 + k]
                        - fz.values.data[(i - 1) * d1 * d2 + j * d2 + k])
                        / 2.0;
                    let dfy_dx = (fy.values.data[(i + 1) * d1 * d2 + j * d2 + k]
                        - fy.values.data[(i - 1) * d1 * d2 + j * d2 + k])
                        / 2.0;
                    let dfx_dy = (fx.values.data[i * d1 * d2 + (j + 1) * d2 + k]
                        - fx.values.data[i * d1 * d2 + (j - 1) * d2 + k])
                        / 2.0;
                    curl_x[idx] = dfz_dy - dfy_dz;
                    curl_y[idx] = dfx_dz - dfz_dx;
                    curl_z[idx] = dfy_dx - dfx_dy;
                }
            }
        }

        Ok([
            Field::new(dims.clone(), Tensor::new(dims.clone(), curl_x)?)?,
            Field::new(dims.clone(), Tensor::new(dims.clone(), curl_y)?)?,
            Field::new(dims.clone(), Tensor::new(dims.clone(), curl_z)?)?,
        ])
    }
}

// Copyright 2026 Mahmoud Harmouch.
//
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
