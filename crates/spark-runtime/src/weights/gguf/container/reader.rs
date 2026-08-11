// SPDX-License-Identifier: AGPL-3.0-only
//! Little-endian bounds-checked cursor + metadata value decoder for the GGUF
//! container parser. Split out of `container.rs` to keep each file ≤500 LoC.

use anyhow::{Context, Result, anyhow, bail};

use super::MetaValue;

/// Round `pos` up to the next multiple of `align`.
pub(super) fn align_up(pos: usize, align: usize) -> usize {
    let align = align.max(1);
    pos.div_ceil(align) * align
}

/// Decode one metadata value of the given wire type (recurses for arrays).
pub(super) fn read_value(r: &mut Reader, vtype: u32) -> Result<MetaValue> {
    Ok(match vtype {
        0 => MetaValue::U8(r.u8()?),
        1 => MetaValue::I8(r.i8()?),
        2 => MetaValue::U16(r.u16()?),
        3 => MetaValue::I16(r.i16()?),
        4 => MetaValue::U32(r.u32()?),
        5 => MetaValue::I32(r.i32()?),
        6 => MetaValue::F32(r.f32()?),
        7 => MetaValue::Bool(r.u8()? != 0),
        8 => MetaValue::Str(r.string()?),
        9 => {
            let elem_type = r.u32().context("array element type")?;
            let len = r.u64().context("array length")?;
            let mut items = Vec::with_capacity(len.min(1 << 20) as usize);
            for _ in 0..len {
                items.push(read_value(r, elem_type)?);
            }
            MetaValue::Array(items)
        }
        10 => MetaValue::U64(r.u64()?),
        11 => MetaValue::I64(r.i64()?),
        12 => MetaValue::F64(r.f64()?),
        other => bail!("unknown gguf metadata value type {other}"),
    })
}

/// Little-endian cursor over the mmapped slice. `pos` is the read head.
pub(super) struct Reader<'a> {
    buf: &'a [u8],
    pub(super) pos: usize,
}

impl<'a> Reader<'a> {
    pub(super) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| anyhow!("length overflow reading {n} bytes at {}", self.pos))?;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| anyhow!("unexpected EOF: need {n} bytes at offset {}", self.pos))?;
        self.pos = end;
        Ok(slice)
    }

    pub(super) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub(super) fn i8(&mut self) -> Result<i8> {
        Ok(self.take(1)?[0] as i8)
    }
    pub(super) fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub(super) fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub(super) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(super) fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(super) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub(super) fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub(super) fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(super) fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// GGUF string: u64 length + UTF-8 bytes.
    pub(super) fn string(&mut self) -> Result<String> {
        let len = self.u64().context("string length")? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).context("invalid UTF-8 in GGUF string")
    }
}
