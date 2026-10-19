//! Philox4x64-10, the counter-based generator behind every bootstrap weight.
//!
//! A counter-based generator is a keyed bijection of a counter: the output for
//! any (key, counter) is computed directly, with no state carried from one
//! number to the next (Salmon et al., "Parallel random numbers: as easy as
//! 1, 2, 3", SC 2011). This makes a bootstrap weight a pure function of
//! (seed, cluster, draw), the same on every machine and in every order
//! (decision 0018). This implementation matches numpy's `np.random.Philox`.

const M0: u64 = 0xD2E7_470E_E14C_6C93;
const M1: u64 = 0xCA5A_8263_9512_1157;
const W0: u64 = 0x9E37_79B9_7F4A_7C15;
const W1: u64 = 0xBB67_AE85_84CA_A73B;
const ROUNDS: usize = 10;

#[inline]
fn mulhilo(a: u64, b: u64) -> (u64, u64) {
    let p = (a as u128) * (b as u128);
    ((p >> 64) as u64, p as u64)
}

/// One Philox4x64-10 block: four 64-bit outputs for a counter and a key.
#[inline]
pub fn philox4x64(counter: [u64; 4], key: [u64; 2]) -> [u64; 4] {
    let (mut c, mut k) = (counter, key);
    for round in 0..ROUNDS {
        if round > 0 {
            k = [k[0].wrapping_add(W0), k[1].wrapping_add(W1)];
        }
        let (hi0, lo0) = mulhilo(M0, c[0]);
        let (hi1, lo1) = mulhilo(M1, c[2]);
        c = [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0];
    }
    c
}

/// A 64-bit output to a uniform number in the open interval $(0, 1)$:
/// $(\lfloor x / 2^{12} \rfloor + \tfrac12)\, 2^{-52}$.
///
/// 52 bits, not 53: with 53, $(2^{53} - 1) + \tfrac12$ rounds to $2^{53}$ and the
/// largest output would map to exactly 1.
#[inline]
pub fn open_unit(x: u64) -> f64 {
    ((x >> 12) as f64 + 0.5) * (1.0 / 4_503_599_627_370_496.0)
}

/// Uniform draws keyed by (seed, unit) for one stream, indexed by draw `b`.
#[derive(Clone, Copy, Debug)]
pub struct Stream {
    key: [u64; 2],
    tag: u64,
}

impl Stream {
    /// The stream of `unit` (a cluster key or stratum index) under `seed`, for the
    /// scheme `tag`.
    pub fn new(seed: u64, unit: u64, tag: u64) -> Self {
        Self { key: [seed, unit], tag }
    }

    /// Writes the uniforms of draws `0..out.len()` into `out`, four per block.
    pub fn uniforms(&self, out: &mut [f64]) {
        for (block, chunk) in out.chunks_mut(4).enumerate() {
            let words = philox4x64([block as u64, self.tag, 0, 0], self.key);
            for (slot, word) in chunk.iter_mut().zip(words) {
                *slot = open_unit(word);
            }
        }
    }

    /// The uniform of draw `b`.
    pub fn uniform(&self, b: usize) -> f64 {
        open_unit(philox4x64([(b / 4) as u64, self.tag, 0, 0], self.key)[b % 4])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_random123_known_answer() {
        // Known-answer test of Random123 (kat_vectors), philox4x64 with 10 rounds.
        let out = philox4x64([0, 0, 0, 0], [0, 0]);
        assert_eq!(
            out,
            [
                0x16554d9eca36314c,
                0xdb20fe9d672d0fdc,
                0xd7e772cee186176b,
                0x7e68b68aec7ba23b
            ]
        );
        let out = philox4x64([u64::MAX; 4], [u64::MAX; 2]);
        assert_eq!(
            out,
            [
                0x87b092c3013fe90b,
                0x438c3c67be8d0224,
                0x9cc7d7c69cd777b6,
                0xa09caebf594f0ba0
            ]
        );
    }

    #[test]
    fn uniforms_are_in_the_open_unit_interval() {
        assert!(open_unit(0) > 0.0 && open_unit(u64::MAX) < 1.0);
        let mut out = vec![0.0; 9];
        Stream::new(1, 2, 0).uniforms(&mut out);
        assert_eq!(out[5], Stream::new(1, 2, 0).uniform(5));
    }
}
