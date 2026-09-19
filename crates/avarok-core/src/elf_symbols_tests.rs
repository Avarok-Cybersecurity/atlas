// SPDX-License-Identifier: AGPL-3.0-only

//! The images here are assembled byte by byte rather than checked in as
//! fixtures: the point of the parser is what it does with a *shape*, and a
//! shape written in code can be truncated, stripped of its symbol table or
//! given a bogus name offset in one line.

use super::defined_function_symbols;

const SHDR_SIZE: usize = 64;
const SYM_SIZE: usize = 24;

/// `(name, st_type, st_shndx)`. `st_shndx = 0` is `SHN_UNDEF`, i.e. a symbol
/// this object references but does not define.
type Symbol<'a> = (&'a str, u8, u16);

const STT_OBJECT: u8 = 1;
const STT_FUNC: u8 = 2;

/// A minimal but valid ELF64 LE relocatable: `.strtab`, an optional `.symtab`
/// linked to it, and a `.text` for defined symbols to point at.
fn elf64(symbols: &[Symbol], with_symtab: bool) -> Vec<u8> {
    let mut strtab = vec![0u8];
    let mut name_offsets = Vec::new();
    for (name, _, _) in symbols {
        name_offsets.push(strtab.len() as u32);
        strtab.extend_from_slice(name.as_bytes());
        strtab.push(0);
    }

    // The null symbol is mandatory and index 0 of every symbol table.
    let mut symtab = vec![0u8; SYM_SIZE];
    for ((_, st_type, st_shndx), st_name) in symbols.iter().zip(&name_offsets) {
        let mut sym = Vec::with_capacity(SYM_SIZE);
        sym.extend_from_slice(&st_name.to_le_bytes());
        sym.push(*st_type); // st_info: binding 0 (LOCAL) << 4 | type
        sym.push(0); // st_other
        sym.extend_from_slice(&st_shndx.to_le_bytes());
        sym.extend_from_slice(&0u64.to_le_bytes()); // st_value
        sym.extend_from_slice(&0u64.to_le_bytes()); // st_size
        symtab.extend_from_slice(&sym);
    }
    if !with_symtab {
        symtab.clear();
    }

    let strtab_off = SHDR_SIZE;
    let symtab_off = strtab_off + strtab.len();
    let shoff = symtab_off + symtab.len();
    let shnum: u16 = if with_symtab { 4 } else { 3 };

    let mut out = Vec::new();
    out.extend_from_slice(b"\x7fELF");
    out.push(2); // EI_CLASS = ELFCLASS64
    out.push(1); // EI_DATA  = ELFDATA2LSB
    out.push(1); // EI_VERSION
    out.extend_from_slice(&[0u8; 9]); // EI_OSABI .. EI_PAD
    out.extend_from_slice(&1u16.to_le_bytes()); // e_type = ET_REL
    out.extend_from_slice(&224u16.to_le_bytes()); // e_machine = EM_AMDGPU
    out.extend_from_slice(&1u32.to_le_bytes()); // e_version
    out.extend_from_slice(&0u64.to_le_bytes()); // e_entry
    out.extend_from_slice(&0u64.to_le_bytes()); // e_phoff
    out.extend_from_slice(&(shoff as u64).to_le_bytes()); // e_shoff
    out.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    out.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
    out.extend_from_slice(&0u16.to_le_bytes()); // e_phentsize
    out.extend_from_slice(&0u16.to_le_bytes()); // e_phnum
    out.extend_from_slice(&(SHDR_SIZE as u16).to_le_bytes()); // e_shentsize
    out.extend_from_slice(&shnum.to_le_bytes()); // e_shnum
    out.extend_from_slice(&1u16.to_le_bytes()); // e_shstrndx
    assert_eq!(out.len(), SHDR_SIZE);

    out.extend_from_slice(&strtab);
    out.extend_from_slice(&symtab);
    assert_eq!(out.len(), shoff);

    let mut shdr = |sh_type: u32, sh_offset: usize, sh_size: usize, sh_link: u32| {
        out.extend_from_slice(&0u32.to_le_bytes()); // sh_name
        out.extend_from_slice(&sh_type.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // sh_flags
        out.extend_from_slice(&0u64.to_le_bytes()); // sh_addr
        out.extend_from_slice(&(sh_offset as u64).to_le_bytes());
        out.extend_from_slice(&(sh_size as u64).to_le_bytes());
        out.extend_from_slice(&sh_link.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // sh_info
        out.extend_from_slice(&1u64.to_le_bytes()); // sh_addralign
        out.extend_from_slice(&0u64.to_le_bytes()); // sh_entsize
    };
    shdr(0, 0, 0, 0); // 0: SHT_NULL
    shdr(3, strtab_off, strtab.len(), 0); // 1: .strtab (SHT_STRTAB)
    if with_symtab {
        shdr(2, symtab_off, symtab.len(), 1); // 2: .symtab -> .strtab
    }
    shdr(1, 0, 0, 0); // last: .text (SHT_PROGBITS), what st_shndx points at
    out
}

fn names(blob: &[u8]) -> Option<Vec<String>> {
    let mut v: Vec<String> = defined_function_symbols(blob)?.into_iter().collect();
    v.sort();
    Some(v)
}

#[test]
fn reports_only_the_defined_function_and_never_the_undefined_one() {
    let blob = elf64(
        &[
            ("vecadd", STT_FUNC, 3),
            ("extern_helper", STT_FUNC, 0),
            ("vecadd.kd", STT_OBJECT, 3),
        ],
        true,
    );
    assert_eq!(names(&blob).unwrap(), vec!["vecadd".to_string()]);
}

/// AMDGPU code objects name the kernel descriptor `<kernel>.kd` as an
/// `STT_OBJECT`. On its own that is still a kernel this object defines.
#[test]
fn a_kernel_descriptor_alone_counts_as_a_defined_kernel() {
    let blob = elf64(&[("avarok_nvfp4_repack.kd", STT_OBJECT, 3)], true);
    assert_eq!(
        names(&blob).unwrap(),
        vec!["avarok_nvfp4_repack".to_string()]
    );
}

/// The case this whole module exists for: the optional module compiled out.
/// A valid object, a symbol table, and nothing launchable in it.
#[test]
fn an_object_with_no_kernels_parses_to_an_empty_set() {
    let blob = elf64(&[("extern_helper", STT_FUNC, 0)], true);
    assert!(defined_function_symbols(&blob).unwrap().is_empty());
}

#[test]
fn an_image_with_no_symbol_table_parses_to_an_empty_set() {
    let blob = elf64(&[], false);
    assert!(defined_function_symbols(&blob).unwrap().is_empty());
}

/// Truncation must not be read as "defines nothing", which would guard against
/// kernels the object really has. Every prefix of a real image declines.
#[test]
fn a_truncated_image_declines_to_answer() {
    let blob = elf64(&[("vecadd", STT_FUNC, 3)], true);
    for cut in [0, 4, 16, 63, blob.len() / 2, blob.len() - 1] {
        assert!(
            defined_function_symbols(&blob[..cut]).is_none(),
            "a {cut}-byte prefix must return None"
        );
    }
    // Sanity: the untruncated image does answer.
    assert_eq!(names(&blob).unwrap(), vec!["vecadd".to_string()]);
}

#[test]
fn non_elf_blobs_decline_to_answer() {
    assert!(defined_function_symbols(b"").is_none());
    assert!(defined_function_symbols(b".version 8.7\n.target sm_121a\n").is_none());
    assert!(defined_function_symbols(b"__CLANG_OFFLOAD_BUNDLE__").is_none());
    // Right magic, wrong class/endianness: ELF32, and ELF64 big-endian.
    let mut elf32 = elf64(&[("vecadd", STT_FUNC, 3)], true);
    elf32[4] = 1;
    assert!(defined_function_symbols(&elf32).is_none());
    let mut be = elf64(&[("vecadd", STT_FUNC, 3)], true);
    be[5] = 2;
    assert!(defined_function_symbols(&be).is_none());
}

/// A name offset past the end of its string table is malformed. Answering
/// with the symbols that did parse would guard on a half-read object.
#[test]
fn a_name_offset_outside_the_string_table_declines_to_answer() {
    let mut blob = elf64(&[("vecadd", STT_FUNC, 3)], true);
    let symtab_off = SHDR_SIZE + 1 + "vecadd".len() + 1;
    let first_sym = symtab_off + SYM_SIZE;
    blob[first_sym..first_sym + 4].copy_from_slice(&0xffff_u32.to_le_bytes());
    assert!(defined_function_symbols(&blob).is_none());
}
