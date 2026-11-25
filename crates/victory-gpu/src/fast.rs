//! Hashing for maps on the recording path. The standard maps hash with
//! SipHash, which is built to resist chosen keys; ours are pointers, slot
//! numbers and table addresses the process makes itself, looked up for
//! every draw (a list's texture states, its dynamic table slots, the
//! heap's buffer views). A multiply-and-fold step per word does for them.

use std::hash::{BuildHasherDefault, Hasher};

/// The 64-bit odd constant of the golden ratio.
const K: u64 = 0x9e37_79b9_7f4a_7c15;

/// Folds the 128-bit product of `a` and `b` into 64 bits: every input bit
/// reaches every output bit.
#[inline]
fn fold(a: u64, b: u64) -> u64 {
    let product = u128::from(a).wrapping_mul(u128::from(b));
    (product as u64) ^ (product >> 64) as u64
}

/// A hasher for keys the process makes itself (see the module text).
#[derive(Default, Clone, Copy)]
pub(crate) struct Fast(u64);

impl Hasher for Fast {
    #[inline]
    fn finish(&self) -> u64 {
        fold(self.0, K)
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(word));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            word[..rest.len()].copy_from_slice(rest);
            // The length keeps "ab" and "ab\0" apart.
            self.write_u64(u64::from_le_bytes(word) ^ (rest.len() as u64) << 59);
        }
    }

    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.write_u64(u64::from(n));
    }

    #[inline]
    fn write_u16(&mut self, n: u16) {
        self.write_u64(u64::from(n));
    }

    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.0 = fold(self.0 ^ n, K);
    }

    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

/// A map hashed with [`Fast`].
pub(crate) type FastMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<Fast>>;
/// A set hashed with [`Fast`].
pub(crate) type FastSet<K> = std::collections::HashSet<K, BuildHasherDefault<Fast>>;

/// A 64-bit digest of descriptor records (`[tag, index, generation]`
/// each), for telling tables with the same contents: two words per record,
/// folded in turn, seeded with the count.
pub(crate) fn records_digest(records: impl IntoIterator<Item = [u32; 3]>) -> u64 {
    let mut h = K;
    let mut n = 0u64;
    for [tag, index, generation] in records {
        let low = u64::from(tag) | u64::from(index) << 32;
        h = fold(h ^ low, K ^ u64::from(generation).rotate_left(17) ^ n);
        n += 1;
    }
    fold(h ^ n, K)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, Hash};

    fn hash_of(value: impl Hash) -> u64 {
        BuildHasherDefault::<Fast>::default().hash_one(value)
    }

    #[test]
    fn maps_work_with_the_fast_hasher() {
        let mut map: FastMap<(usize, u64), u32> = FastMap::default();
        for i in 0..10_000u32 {
            map.insert((i as usize * 64, u64::from(i) << 32), i);
        }
        assert_eq!(map.len(), 10_000);
        for i in 0..10_000u32 {
            assert_eq!(map.get(&(i as usize * 64, u64::from(i) << 32)), Some(&i));
        }
        let set: FastSet<usize> = (0..1000).map(|i| i * 4096).collect();
        assert!(set.contains(&(999 * 4096)) && !set.contains(&1));
    }

    #[test]
    fn nearby_keys_spread_over_the_low_bits() {
        // Pointers 64 bytes apart must not all land in a few buckets.
        let buckets: FastSet<u64> = (0..4096u64).map(|i| hash_of(0x7ff0_0000_0000usize + i as usize * 64) & 0xfff).collect();
        assert!(buckets.len() > 2048, "{} buckets of 4096", buckets.len());
    }

    #[test]
    fn byte_strings_of_other_lengths_differ() {
        assert_ne!(hash_of(&b"ab"[..]), hash_of(&b"ab\0"[..]));
        assert_ne!(hash_of("victory"), hash_of("victorz"));
    }

    #[test]
    fn records_digest_tells_tables_apart() {
        let a = [[1, 2, 3], [1, 5, 6]];
        let b = [[1, 5, 6], [1, 2, 3]];
        assert_eq!(records_digest(a), records_digest(a));
        // Order, a changed generation and a longer table all count.
        assert_ne!(records_digest(a), records_digest(b));
        assert_ne!(records_digest(a), records_digest([[1, 2, 3], [1, 5, 7]]));
        assert_ne!(records_digest(a), records_digest([[1, 2, 3], [1, 5, 6], [0, 0, 0]]));
        // No collisions over many tables that differ in one slot.
        let digests: FastSet<u64> = (0..100_000u32).map(|i| records_digest([[7, i, 1], [7, 0, 1]])).collect();
        assert_eq!(digests.len(), 100_000);
    }
}
