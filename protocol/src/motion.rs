//! Rigid motions over one parameter `τ` from 0 to 1, each the identity at `τ = 0`: what
//! [`Kernel::sweep_solid`](crate::Kernel::sweep_solid) carries a solid by.

use serde_json::{Value, json};

use crate::model::{Rigid, Vec3, add, scale};
use crate::path::PathCurve;

/// How a solid moves as `τ` runs from 0 to 1, from where it is.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Motion {
    /// Carried straight by `τ·vector`.
    Slide { vector: Vec3 },
    /// Turned right-handed by `τ·angle` about the line through `origin` along `axis`.
    Turn {
        origin: Vec3,
        axis: Vec3,
        angle: f64,
    },
    /// Turned as by [`Motion::Turn`] while carried `rise·τ·angle` along `axis`: `rise` per radian.
    Screw {
        origin: Vec3,
        axis: Vec3,
        rise: f64,
        angle: f64,
    },
    /// Carried by a frame riding `path` from its start to its end: the frame's origin on the path,
    /// its x along the path's tangent, its z the part of `up` square to the tangent. The solid
    /// keeps its place in that frame.
    Along { path: PathCurve, up: Vec3 },
    /// Two motions over the one parameter: at `τ` the solid is where the first puts it, moved by
    /// the second.
    Then(Box<Motion>, Box<Motion>),
}

fn unit(a: Vec3) -> Vec3 {
    scale(a, 1.0 / (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt())
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `second` after `first`.
fn composed(first: &Rigid, second: &Rigid) -> Rigid {
    let column = |j: usize| second.rotate([first.r[0][j], first.r[1][j], first.r[2][j]]);
    let c = [column(0), column(1), column(2)];
    Rigid {
        r: [0, 1, 2].map(|i| [c[0][i], c[1][i], c[2][i]]),
        t: second.point(first.t),
    }
}

/// The frame riding `path` at `t`: its origin and its axes x, y, z.
fn riding(path: &PathCurve, up: Vec3, t: f64) -> (Vec3, [Vec3; 3]) {
    let [at, tangent, _] = path.eval(t);
    let x = unit(tangent);
    let along = up[0] * x[0] + up[1] * x[1] + up[2] * x[2];
    let z = unit(add(up, scale(x, -along)));
    (at, [x, cross(z, x), z])
}

impl Motion {
    /// Where the motion has carried a solid's points to at `τ`.
    pub fn at(&self, tau: f64) -> Rigid {
        match self {
            Motion::Slide { vector } => Rigid {
                t: scale(*vector, tau),
                ..Rigid::IDENTITY
            },
            Motion::Turn {
                origin,
                axis,
                angle,
            } => Rigid::about(*origin, unit(*axis), tau * angle),
            Motion::Screw {
                origin,
                axis,
                rise,
                angle,
            } => {
                let k = unit(*axis);
                let turn = Rigid::about(*origin, k, tau * angle);
                Rigid {
                    t: add(turn.t, scale(k, rise * tau * angle)),
                    ..turn
                }
            }
            Motion::Along { path, up } => {
                let (t0, t1) = path.domain();
                let (o0, a0) = riding(path, *up, t0);
                let (o1, a1) = riding(path, *up, t0 + (t1 - t0) * tau);
                // The rotation taking the first frame's axes to this one's: Σ a1ₖ a0ₖᵀ.
                let r =
                    [0, 1, 2].map(|i| [0, 1, 2].map(|j| (0..3).map(|k| a1[k][i] * a0[k][j]).sum()));
                let turned = Rigid { r, t: [0.0; 3] }.rotate(o0);
                Rigid {
                    r,
                    t: add(o1, scale(turned, -1.0)),
                }
            }
            Motion::Then(first, second) => composed(&first.at(tau), &second.at(tau)),
        }
    }

    /// The motion seen once the world is moved by `m`: it carries `m`'s image of a solid as this
    /// one carries the solid.
    pub fn moved(&self, m: &Rigid) -> Motion {
        match self {
            Motion::Slide { vector } => Motion::Slide {
                vector: m.rotate(*vector),
            },
            Motion::Turn {
                origin,
                axis,
                angle,
            } => Motion::Turn {
                origin: m.point(*origin),
                axis: m.rotate(*axis),
                angle: *angle,
            },
            Motion::Screw {
                origin,
                axis,
                rise,
                angle,
            } => Motion::Screw {
                origin: m.point(*origin),
                axis: m.rotate(*axis),
                rise: *rise,
                angle: *angle,
            },
            Motion::Along { path, up } => Motion::Along {
                path: PathCurve::new(
                    path.degree,
                    path.knots.clone(),
                    path.ctrl.iter().map(|&c| m.point(c)).collect(),
                ),
                up: m.rotate(*up),
            },
            Motion::Then(first, second) => {
                Motion::Then(Box::new(first.moved(m)), Box::new(second.moved(m)))
            }
        }
    }

    fn numbers(&self, out: &mut Vec<f64>) {
        match self {
            Motion::Slide { vector } => {
                out.push(0.0);
                out.extend(vector);
            }
            Motion::Turn {
                origin,
                axis,
                angle,
            } => {
                out.push(1.0);
                out.extend(origin.iter().chain(axis).chain([angle]));
            }
            Motion::Screw {
                origin,
                axis,
                rise,
                angle,
            } => {
                out.push(2.0);
                out.extend(origin.iter().chain(axis).chain([rise, angle]));
            }
            Motion::Along { path, up } => {
                out.extend([3.0, path.degree as f64]);
                out.extend(&path.knots);
                out.push(-1.0);
                out.extend(path.ctrl.iter().flatten().chain(up));
            }
            Motion::Then(first, second) => {
                out.push(4.0);
                first.numbers(out);
                second.numbers(out);
            }
        }
    }

    /// The parameters as one canonical string, for case ids.
    pub fn key(&self) -> String {
        let mut numbers = Vec::new();
        self.numbers(&mut numbers);
        let nums: Vec<String> = numbers.iter().map(|v| format!("{v:.12e}")).collect();
        format!("motion:{}", nums.join(","))
    }

    pub fn to_json(&self) -> Value {
        match self {
            Motion::Slide { vector } => json!({"slide": vector}),
            Motion::Turn {
                origin,
                axis,
                angle,
            } => json!({"turn": {"origin": origin, "axis": axis, "angle": angle}}),
            Motion::Screw {
                origin,
                axis,
                rise,
                angle,
            } => json!({"screw": {"origin": origin, "axis": axis, "rise": rise, "angle": angle}}),
            Motion::Along { path, up } => json!({"along": {
                "path": {"degree": path.degree, "knots": path.knots, "ctrl": path.ctrl},
                "up": up,
            }}),
            Motion::Then(first, second) => json!({"then": [first.to_json(), second.to_json()]}),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: Vec3, b: Vec3) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-12)
    }

    fn all() -> Vec<Motion> {
        let turn = Motion::Turn {
            origin: [1.0, 0.0, 0.0],
            axis: [0.0, 0.0, 2.0],
            angle: 1.5,
        };
        let slide = Motion::Slide {
            vector: [0.0, 0.0, 3.0],
        };
        vec![
            slide.clone(),
            turn.clone(),
            Motion::Screw {
                origin: [1.0, 0.0, 0.0],
                axis: [0.0, 0.0, 1.0],
                rise: 2.0,
                angle: 1.5,
            },
            Motion::Along {
                path: PathCurve::new(
                    3,
                    vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                    vec![[0.0; 3], [2.0, 0.0, 0.0], [4.0, 2.0, 0.0], [4.0, 5.0, 1.0]],
                ),
                up: [0.0, 0.0, 1.0],
            },
            Motion::Then(Box::new(turn), Box::new(slide)),
        ]
    }

    #[test]
    fn every_motion_starts_where_the_solid_is() {
        for m in all() {
            let p = [0.3, -1.2, 2.5];
            assert!(near(m.at(0.0).point(p), p), "{m:?}");
        }
    }

    #[test]
    fn a_turn_then_a_slide_along_its_axis_is_the_screw() {
        let [_, _, screw, _, composed] = <[Motion; 5]>::try_from(all()).unwrap();
        let p = [0.3, -1.2, 2.5];
        for tau in [0.25, 0.5, 1.0] {
            assert!(near(composed.at(tau).point(p), screw.at(tau).point(p)));
        }
        assert!(near(
            screw.at(1.0).point([2.0, 0.0, 0.0]),
            [1.0 + 1.5_f64.cos(), 1.5_f64.sin(), 3.0]
        ));
    }

    #[test]
    fn a_path_carries_the_frame_riding_it() {
        let along = &all()[3];
        let Motion::Along { path, .. } = along else {
            unreachable!()
        };
        // The path starts at the origin heading along x, so the solid's frame there is the world's.
        let end = along.at(1.0);
        assert!(near(end.point([0.0; 3]), path.eval(1.0)[0]));
        let tangent = unit(path.eval(1.0)[1]);
        assert!(near(end.rotate([1.0, 0.0, 0.0]), tangent));
        assert!(end.rotate([0.0, 0.0, 1.0])[2] > 0.0);
    }

    #[test]
    fn a_moved_motion_carries_the_moved_solid_alike() {
        let m = Rigid::about([1.0, 2.0, 3.0], unit([1.0, 1.0, 0.5]), 0.8);
        let p = [0.3, -1.2, 2.5];
        for motion in all() {
            let there = m.point(motion.at(0.7).point(p));
            assert!(
                near(motion.moved(&m).at(0.7).point(m.point(p)), there),
                "{motion:?}"
            );
        }
    }

    #[test]
    fn every_motion_reads_back_as_written() {
        for m in all() {
            assert_eq!(Motion::from_json(&m.to_json()), Ok(m));
        }
        assert!(Motion::from_json(&json!({"spin": 1})).is_err());
    }
}
