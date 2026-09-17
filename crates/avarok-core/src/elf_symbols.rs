// SPDX-License-Identifier: AGPL-3.0-only

//! The kernel names a code object actually defines, read straight from its
//! ELF symbol table.
//!
//! **Why this exists.** Avarok has optional kernel modules: a `.cu` whose whole
//! body sits inside a capability guard (`nvfp4_mmq.cu` is entirely within
//! `#if defined(BLACKWELL_MMA_AVAILABLE)`), so on a target without that
//! capability it compiles to an object with no kernels in it at all. On NVIDIA
//! that is harmless: `cuModuleGetFunction` answers "not found",
//! `spark_model::layers::kernel_probe::try_kernel` maps the error to
//! `KernelHandle(0)`, and every use site is already guarded.
//!
//! Observed on SCALE 1.7.1 / gfx1201 (AMD Radeon AI PRO R9700), the driver does
//! NOT answer "not found": `cuModuleGetFunction` on the empty module returns
//! success with a handle that is not backed by any code, and the first launch
//! through it dies with `CUDA_ERROR_INVALID_IMAGE (200)` naming
//! `nvfp4_mmq::avarok_nvfp4_repack`. The registry therefore cannot trust the
//! driver's answer for a name the object does not define, and this module is
//! how it gets a second opinion: the set of symbols the blob itself declares.
//!
//! **Format subset.** ELF64, little-endian, generic section-header +
//! symbol-table walk. `EM_AMDGPU` is not special-cased and neither is any other
//! machine; nothing here knows what a GPU is. Anything else returns `None`,
//! meaning "could not parse, do not guard": ELF32, big endian, a clang offload
//! bundle (`__CLANG_OFFLOAD_BUNDLE__`), a PTX text blob, a truncated or
//! otherwise malformed image. It runs at boot on bytes whose shape is not
//! guaranteed, so every read is bounds-checked and nothing here can panic.

use std::collections::HashSet;

const SHT_SYMTAB: u32 = 2;
const SHT_DYNSYM: u32 = 11;
const SHN_UNDEF: u16 = 0;
const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;
const SHDR_SIZE: usize = 64;
const SYM_SIZE: usize = 24;

/// AMDGPU code objects carry each kernel twice: an `STT_FUNC` for the code and
/// an `STT_OBJECT` kernel descriptor named `<kernel>.kd`. Either form counts.
const KD_SUFFIX: &str = ".kd";

fn u16_at(blob: &[u8], off: usize) -> Option<u16> {
    Some(u16::from_le_bytes(blob.get(off..off + 2)?.try_into().ok()?))
}

fn u32_at(blob: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_le_bytes(blob.get(off..off + 4)?.try_into().ok()?))
}

fn u64_at(blob: &[u8], off: usize) -> Option<u64> {
    Some(u64::from_le_bytes(blob.get(off..off + 8)?.try_into().ok()?))
}

/// One section header, reduced to the four fields this walk uses.
struct Section {
    sh_type: u32,
    sh_offset: usize,
    sh_size: usize,
    sh_link: usize,
}

/// Every section header in `blob`, or `None` if the image is not an ELF64 LE
/// file or any header runs past the end of the blob.
fn sections(blob: &[u8]) -> Option<Vec<Section>> {
    // e_ident: magic, then EI_CLASS (2 = ELF64) and EI_DATA (1 = little endian).
    if blob.len() < SHDR_SIZE || !blob.starts_with(b"\x7fELF") {
        return None;
    }
    if blob.get(4)? != &2 || blob.get(5)? != &1 {
        return None;
    }
    let e_shoff = u64_at(blob, 0x28)? as usize;
    let e_shentsize = u16_at(blob, 0x3a)? as usize;
    let e_shnum = u16_at(blob, 0x3c)? as usize;
    // `e_shnum == 0` means either "no sections" or the extended-numbering form
    // where the real count lives in section 0. This walk does not implement the
    // extended form, and cannot tell the two apart, so it declines both.
    if e_shoff == 0 || e_shnum == 0 || e_shentsize < SHDR_SIZE {
        return None;
    }
    let mut out = Vec::with_capacity(e_shnum);
    for i in 0..e_shnum {
        let base = e_shoff.checked_add(i.checked_mul(e_shentsize)?)?;
        // Reject rather than truncate: a header table that does not fit is a
        // malformed image, not a short one.
        if base.checked_add(SHDR_SIZE)? > blob.len() {
            return None;
        }
        out.push(Section {
            sh_type: u32_at(blob, base + 4)?,
            sh_offset: u64_at(blob, base + 24)? as usize,
            sh_size: u64_at(blob, base + 32)? as usize,
            sh_link: u32_at(blob, base + 40)? as usize,
        });
    }
    Some(out)
}

/// The NUL-terminated string at `off` in the string-table bytes `strtab`.
fn string_at(strtab: &[u8], off: usize) -> Option<String> {
    let rest = strtab.get(off..)?;
    let end = rest.iter().position(|&b| b == 0)?;
    std::str::from_utf8(&rest[..end]).ok().map(str::to_owned)
}

/// The names of the function symbols `blob` **defines** (`st_shndx !=
/// SHN_UNDEF`), for an ELF64 little-endian image.
///
/// A name is included if the object carries it as an `STT_FUNC`, or as an
/// `STT_OBJECT` called `<name>.kd` (the AMDGPU kernel descriptor), or both.
/// Undefined symbols are never included: the external helpers a relocatable
/// object still needs linked are exactly what must not be mistaken for a
/// launchable kernel.
///
/// `Some(set)` is an answer to trust; `Some(empty)` genuinely means "this
/// object defines no kernels". `None` means the bytes could not be parsed, and
/// callers must fall back to asking the driver rather than guarding on a set
/// they do not have.
pub fn defined_function_symbols(blob: &[u8]) -> Option<HashSet<String>> {
    let sections = sections(blob)?;
    // `.symtab` if there is one, `.dynsym` otherwise. A stripped object keeps
    // only the dynamic table, and its kernels are exported either way.
    let symtab = sections
        .iter()
        .find(|s| s.sh_type == SHT_SYMTAB)
        .or_else(|| sections.iter().find(|s| s.sh_type == SHT_DYNSYM));
    let Some(symtab) = symtab else {
        // No symbol table at all: a valid ELF that defines nothing we can name.
        return Some(HashSet::new());
    };
    let strtab = sections.get(symtab.sh_link)?;
    let strtab_bytes = blob.get(strtab.sh_offset..strtab.sh_offset.checked_add(strtab.sh_size)?)?;
    let symtab_bytes = blob.get(symtab.sh_offset..symtab.sh_offset.checked_add(symtab.sh_size)?)?;

    let mut funcs = HashSet::new();
    let mut descriptors = HashSet::new();
    for entry in symtab_bytes.chunks_exact(SYM_SIZE) {
        let st_name = u32_at(entry, 0)? as usize;
        let st_type = entry.get(4)? & 0x0f;
        let st_shndx = u16_at(entry, 6)?;
        if st_shndx == SHN_UNDEF || st_name == 0 {
            continue;
        }
        let Some(name) = string_at(strtab_bytes, st_name) else {
            // A name pointing outside its own string table is malformed; the
            // whole answer is untrustworthy, so no guard is better than a
            // partial one.
            return None;
        };
        match st_type {
            STT_FUNC => {
                funcs.insert(name);
            }
            STT_OBJECT => {
                if let Some(kernel) = name.strip_suffix(KD_SUFFIX) {
                    descriptors.insert(kernel.to_owned());
                }
            }
            _ => {}
        }
    }
    funcs.extend(descriptors);
    Some(funcs)
}

#[cfg(test)]
#[path = "elf_symbols_tests.rs"]
mod elf_symbols_tests;
