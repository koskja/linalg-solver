//! Compact bit-list backed by a smallvec of bytes.
#![allow(dead_code)]
use smallvec::SmallVec;

pub const INLINE_BIT_BYTES: usize = 16;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BitList {
    len: usize,
    data: SmallVec<[u8; INLINE_BIT_BYTES]>,
}

impl BitList {
    pub fn zeros(len: usize) -> Self {
        let byte_len = (len + 7) / 8;
        let mut data: SmallVec<[u8; INLINE_BIT_BYTES]> = SmallVec::new();
        data.resize(byte_len.max(1), 0);
        Self { len, data }
    }

    pub fn from_bools(bits: impl IntoIterator<Item = bool>) -> Self {
        let mut data: SmallVec<[u8; INLINE_BIT_BYTES]> = SmallVec::new();
        let mut len = 0;

        for (idx, value) in bits.into_iter().enumerate() {
            let byte_idx = idx / 8;
            let bit_idx = idx % 8;
            if byte_idx >= data.len() {
                data.push(0);
            }
            if value {
                data[byte_idx] |= 1 << bit_idx;
            }
            len += 1;
        }

        Self { len, data }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get(&self, idx: usize) -> bool {
        if idx >= self.len {
            return false;
        }
        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        (self.data[byte_idx] >> bit_idx) & 1 == 1
    }

    pub fn set(&mut self, idx: usize, value: bool) {
        if idx >= self.len {
            return;
        }
        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        if value {
            self.data[byte_idx] |= 1 << bit_idx;
        } else {
            self.data[byte_idx] &= !(1 << bit_idx);
        }
    }

    pub fn count_ones(&self) -> usize {
        self.data.iter().map(|b| b.count_ones() as usize).sum()
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}
