use crate::types::{ModalEquationSolver, PlateGeometry};
use std::f32::consts::{FRAC_PI_4, PI};

/// Modal parameters `(m, n, a, b)` for Chladni standing-wave evaluation.
#[derive(Debug, Clone, Copy)]
pub struct ModalParams {
    pub m: f32,
    pub n: f32,
    pub a: f32,
    pub b: f32,
}

/// Evaluates Chladni plate vibration displacement `w(x, y)` in normalized domain `[-1.0, 1.0]`.
pub fn chladni_displacement(
    x: f32,
    y: f32,
    geometry: PlateGeometry,
    solver: ModalEquationSolver,
    params: ModalParams,
) -> f32 {
    let ModalParams { m, n, a, b } = params;
    match geometry {
        PlateGeometry::Square => match solver {
            ModalEquationSolver::ClassicSymmetric => {
                let term1 = (n * PI * x).cos() * (m * PI * y).cos();
                let term2 = (m * PI * x).cos() * (n * PI * y).cos();
                a * term1 - b * term2
            }
            ModalEquationSolver::AsymmetricMixed => {
                let term1 = (n * PI * x).sin() * (m * PI * y).cos();
                let term2 = (m * PI * x).cos() * (n * PI * y).sin();
                a * term1 + b * term2
            }
            ModalEquationSolver::BesselCircular => {
                let r = (x * x + y * y).sqrt().min(1.0);
                let theta = y.atan2(x);
                let j = bessel_jn(n.round() as i32, (m + 1.0) * PI * r);
                j * (n * theta).cos()
            }
        },
        PlateGeometry::Circular => {
            let r = (x * x + y * y).sqrt();
            if r > 1.0 {
                return 0.0;
            }
            let theta = y.atan2(x);
            match solver {
                ModalEquationSolver::BesselCircular => {
                    let order = (n.round() as i32).abs().clamp(0, 12);
                    let k = m.round().max(1.0) * PI * 1.2;
                    let j = bessel_jn(order, k * r);
                    j * (order as f32 * theta).cos()
                }
                ModalEquationSolver::ClassicSymmetric => {
                    let wave_r = ((m + 1.0) * PI * r).cos();
                    let wave_theta = (n * theta).cos();
                    a * wave_r * wave_theta
                }
                ModalEquationSolver::AsymmetricMixed => {
                    let wave_r = ((m + 1.0) * PI * r).sin();
                    let wave_theta = (n * theta).sin();
                    b * wave_r * wave_theta
                }
            }
        }
        PlateGeometry::Hexagonal => {
            let mut sum = 0.0;
            for i in 0..3 {
                let angle = i as f32 * (2.0 * PI / 3.0);
                let u = x * angle.cos() + y * angle.sin();
                let v = -x * angle.sin() + y * angle.cos();
                let term = (n * PI * u).cos() * (m * PI * v).cos();
                sum += term;
            }
            a * (sum / 3.0)
        }
    }
}

/// Evaluates spatial gradient of vibration energy `-grad(w^2) = -2 * w * grad(w)`.
pub fn chladni_force_gradient(
    x: f32,
    y: f32,
    geometry: PlateGeometry,
    solver: ModalEquationSolver,
    params: ModalParams,
) -> (f32, f32, f32) {
    let eps = 0.002;
    let w = chladni_displacement(x, y, geometry, solver, params);
    let wx_plus = chladni_displacement(x + eps, y, geometry, solver, params);
    let wx_minus = chladni_displacement(x - eps, y, geometry, solver, params);
    let wy_plus = chladni_displacement(x, y + eps, geometry, solver, params);
    let wy_minus = chladni_displacement(x, y - eps, geometry, solver, params);

    let dw_dx = (wx_plus - wx_minus) / (2.0 * eps);
    let dw_dy = (wy_plus - wy_minus) / (2.0 * eps);

    (-2.0 * w * dw_dx, -2.0 * w * dw_dy, w)
}

/// Computes Bessel function `J_n(x)` of the first kind using standard polynomial approximations.
pub fn bessel_jn(n: i32, x: f32) -> f32 {
    let n = n.abs();
    if n == 0 {
        return bessel_j0(x);
    } else if n == 1 {
        return bessel_j1(x);
    }

    let mut j_prev = bessel_j0(x);
    let mut j_curr = bessel_j1(x);

    if x.abs() < 1e-6 {
        return 0.0;
    }

    for k in 1..n {
        let j_next = (2.0 * k as f32 / x) * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = j_next;
    }

    j_curr.clamp(-1.0, 1.0)
}

fn bessel_j0(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        1.0 + y * (-2.249_999_8 + y * (1.265_620_8 + y * (-0.316_386_6 + y * (0.044_447_9 - y * 0.003_944_4))))
    } else {
        let y = 3.75 / ax;
        let f0 = 0.797_884_6 + y * (-0.000_000_8 + y * (-0.005_527_4 + y * (0.000_095_1 + y * (0.001_372_4 - y * 0.000_728_1))));
        let theta0 = ax - FRAC_PI_4 + y * (-0.041_664_0 + y * (-0.000_039_5 + y * (0.002_625_7 + y * (-0.000_541_3 - y * 0.000_293_3))));
        (1.0 / ax.sqrt()) * f0 * theta0.cos()
    }
}

fn bessel_j1(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        let ans = ax * (0.5 + y * (-0.562_499_8 + y * (0.210_935_7 + y * (-0.039_542_9 + y * (0.004_433_2 - y * 0.000_317_6)))));
        if x < 0.0 {
            -ans
        } else {
            ans
        }
    } else {
        let y = 3.75 / ax;
        let f1 = 0.797_884_6 + y * (0.000_001_6 + y * (0.016_596_7 + y * (0.000_171_1 + y * (-0.002_495_1 + y * 0.001_136_5))));
        let theta1 = ax - 3.0 * FRAC_PI_4 + y * (0.041_664_0 + y * (-0.000_039_5 + y * (-0.002_625_7 + y * (-0.000_541_3 + y * 0.000_293_3))));
        let ans = (1.0 / ax.sqrt()) * f1 * theta1.cos();
        if x < 0.0 {
            -ans
        } else {
            ans
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_symmetric_nodal_diagonal_vanishes() {
        let params = ModalParams {
            m: 3.0,
            n: 5.0,
            a: 1.0,
            b: 1.0,
        };
        let w = chladni_displacement(
            0.37,
            0.37,
            PlateGeometry::Square,
            ModalEquationSolver::ClassicSymmetric,
            params,
        );
        assert!(w.abs() < 1e-6, "Expected w=0 on x=y diagonal, got {}", w);

        let (fx, fy, _) = chladni_force_gradient(
            0.25,
            0.1,
            PlateGeometry::Square,
            ModalEquationSolver::ClassicSymmetric,
            params,
        );
        assert!(fx.is_finite() && fy.is_finite());

        let bessel_w = chladni_displacement(
            0.2,
            0.15,
            PlateGeometry::Circular,
            ModalEquationSolver::BesselCircular,
            params,
        );
        assert!(bessel_w.is_finite());
    }
}
