//! The layout of a cluster summary $u_a$.
//!
//! A cluster summary holds the three [`Blocks`] of one cluster's rows, unweighted
//! (the cluster's weight is constant, so it is applied later, per stratum):
//!
//! $$u_a = \big(\text{vec}\,S^Y_a,\ \text{vech}\,S_a,\ \text{vech}\,S^T_a\big).$$
//!
//! The **distinct layout** stores $S^Y$ column-major ($d \cdot o$ entries) and only
//! the upper triangles of the symmetric $S$ and $S^T$ ($d(d+1)/2$ each), so
//! $q = do + d(d+1)$. Entry $(j, m)$ with $j \le m$ of a triangle sits at
//! $m(m+1)/2 + j$.

use faer::Mat;

use crate::blocks::Blocks;

/// Which block an entry of $u_a$ belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Block {
    /// $S^Y$, entry (row $j$ of $R$, outcome $l$).
    Sy,
    /// $S$, entry $(j, m)$ with $j \le m$.
    S,
    /// $S^T$, entry $(j, m)$ with $j \le m$.
    St,
}

/// The distinct layout of a cluster summary, for $d$ columns of $R$ and $o$ outcomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// Columns of $R$.
    pub d: usize,
    /// Outcome columns.
    pub o: usize,
}

impl Layout {
    /// A layout for $d$ columns of $R$ and $o$ outcome columns.
    pub fn new(d: usize, o: usize) -> Self {
        Self { d, o }
    }

    /// Entries of one triangle, $d(d+1)/2$.
    pub fn triangle(&self) -> usize {
        self.d * (self.d + 1) / 2
    }

    /// Length of $u_a$, $q = do + d(d+1)$.
    pub fn q(&self) -> usize {
        self.d * self.o + 2 * self.triangle()
    }

    /// Position of $S^Y_{jl}$.
    pub fn sy(&self, j: usize, l: usize) -> usize {
        l * self.d + j
    }

    fn tri(j: usize, m: usize) -> usize {
        let (lo, hi) = if j <= m { (j, m) } else { (m, j) };
        hi * (hi + 1) / 2 + lo
    }

    /// Position of $S_{jm}$ (either order).
    pub fn s(&self, j: usize, m: usize) -> usize {
        self.d * self.o + Self::tri(j, m)
    }

    /// Position of $S^T_{jm}$ (either order).
    pub fn st(&self, j: usize, m: usize) -> usize {
        self.d * self.o + self.triangle() + Self::tri(j, m)
    }

    /// The block and indices of every entry, in order: `(block, j, l)` for $S^Y$,
    /// `(block, j, m)` with $j \le m$ for the triangles.
    pub fn entries(&self) -> Vec<(Block, usize, usize)> {
        let mut out = Vec::with_capacity(self.q());
        for l in 0..self.o {
            for j in 0..self.d {
                out.push((Block::Sy, j, l));
            }
        }
        for block in [Block::S, Block::St] {
            for m in 0..self.d {
                for j in 0..=m {
                    out.push((block, j, m));
                }
            }
        }
        out
    }

    /// The [`Blocks`] stored in a summary (or a sum of summaries) `u`, scaled by `w`.
    pub fn blocks(&self, u: &[f64], w: f64) -> Blocks {
        let (d, o) = (self.d, self.o);
        Blocks {
            s_y: Mat::from_fn(d, o, |j, l| w * u[self.sy(j, l)]),
            s: Mat::from_fn(d, d, |j, m| w * u[self.s(j, m)]),
            s_t: Mat::from_fn(d, d, |j, m| w * u[self.st(j, m)]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_enumerate_every_entry_once() {
        let layout = Layout::new(3, 2);
        let entries = layout.entries();
        assert_eq!(entries.len(), layout.q());
        assert_eq!(layout.q(), 3 * 2 + 3 * 4);
        for (k, &(block, j, m)) in entries.iter().enumerate() {
            let position = match block {
                Block::Sy => layout.sy(j, m),
                Block::S => layout.s(j, m),
                Block::St => layout.st(j, m),
            };
            assert_eq!(position, k);
        }
        assert_eq!(layout.s(2, 0), layout.s(0, 2));
    }
}
