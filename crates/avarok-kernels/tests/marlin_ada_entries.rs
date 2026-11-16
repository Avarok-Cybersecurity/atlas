// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: pin the export table of `kernels/ada/common/marlin_ada.cu` (port sprint
//! lane L18a), the API the Marlin host (L18b, `spark-model/src/port/marlin/`) binds by name.
//!
//! Every Marlin entry point is one `MARLIN_ADA_ENTRY(name, weight, threads, m_blocks,
//! n_blocks, k_blocks, m_block_size_8)` line, and the NAME is the only thing the host sees.
//! A name that says `k128_n64_t128` over template arguments for another tile would launch the
//! wrong geometry with no runtime error (the grid and shared memory the host derives from the
//! name would not match the kernel). These tests decode each name and check it against its
//! own template arguments, check the whole set against vLLM's instantiation rule
//! (`generate_kernels.py:53-161`, the thread-config filter of `marlin.cu:75-89`), and, on a
//! build with compiled PTX, that every Ada target carries exactly those 30 entries with the
//! 17-parameter ABI and that no other target carries the module at all.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The Ada kernel directory.
fn ada_common() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/ada/common")
}

fn read(name: &str) -> String {
    let path = ada_common().join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The files lane L18a adds under kernels/ada/common.
const MARLIN_FILES: &[&str] = &[
    "marlin_ada.cu",
    "marlin_common_ada.cuh",
    "marlin_dequant_ada.cuh",
    "marlin_dtypes_ada.cuh",
    "marlin_mma_ada.cuh",
    "marlin_scalar_type_ada.cuh",
    "marlin_template_ada.cuh",
];

/// Parameters of every exported entry (kernel.h's MARLIN_KERNEL_PARAMS: 8 pointers, 4 ints,
/// `locks`, 3 flags, `max_shared_mem`).
const ENTRY_PARAMS: usize = 17;

/// One `MARLIN_ADA_ENTRY(...)` line.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Entry {
    name: String,
    weight_id: String,
    threads: u32,
    m_blocks: u32,
    n_blocks: u32,
    k_blocks: u32,
    m8: bool,
}

fn parse_entries(src: &str) -> Vec<Entry> {
    src.lines()
        .map(str::trim)
        .filter(|l| l.starts_with("MARLIN_ADA_ENTRY("))
        .map(|l| {
            let inner = l
                .strip_prefix("MARLIN_ADA_ENTRY(")
                .and_then(|r| r.strip_suffix(')'))
                .unwrap_or_else(|| panic!("entry line is not one macro call: {l}"));
            let f: Vec<&str> = inner.split(',').map(str::trim).collect();
            assert_eq!(f.len(), 7, "entry needs 7 arguments: {l}");
            let num = |s: &str| -> u32 {
                s.parse()
                    .unwrap_or_else(|_| panic!("not a number {s}: {l}"))
            };
            Entry {
                name: f[0].to_string(),
                weight_id: f[1].to_string(),
                threads: num(f[2]),
                m_blocks: num(f[3]),
                n_blocks: num(f[4]),
                k_blocks: num(f[5]),
                m8: match f[6] {
                    "true" => true,
                    "false" => false,
                    other => panic!("m_block_size_8 must be true or false, got {other}: {l}"),
                },
            }
        })
        .collect()
}

/// What a name promises: (weight label, m label, thread_k, thread_n, threads).
fn decode_name(name: &str) -> (String, String, u32, u32, u32) {
    let rest = name
        .strip_prefix("marlin_ada_bf16_")
        .unwrap_or_else(|| panic!("{name}: must start with marlin_ada_bf16_"));
    let parts: Vec<&str> = rest.split('_').collect();
    assert_eq!(parts.len(), 6, "{name}: <w>_g128_m<M>_k<K>_n<N>_t<T>");
    assert_eq!(parts[1], "g128", "{name}: group 128 only");
    let field = |p: &str, prefix: char| -> u32 {
        p.strip_prefix(prefix)
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("{name}: bad field {p}"))
    };
    let _ = field(parts[2], 'm');
    (
        parts[0].to_string(),
        parts[2].to_string(),
        field(parts[3], 'k'),
        field(parts[4], 'n'),
        field(parts[5], 't'),
    )
}

/// vLLM's rule, written out: weight types x thread_m_blocks x thread configs, with the
/// 256-thread configs filtered by batch size (generate_kernels.py:153-161).
fn expected_set() -> BTreeSet<Entry> {
    // (label, ScalarType id constant)
    let weights = [("u4b8", "vllm::kU4B8Id"), ("u4", "vllm::kU4Id")];
    // (label, thread_m_blocks, m_block_size_8); vLLM's 0.5 is m_blocks 1 with m8 set.
    let ms = [
        ("m8", 1, true),
        ("m16", 1, false),
        ("m32", 2, false),
        ("m48", 3, false),
        ("m64", 4, false),
    ];
    // (thread_k, thread_n, threads)
    let all_configs = [
        (128, 128, 256),
        (64, 256, 256),
        (64, 128, 128),
        (128, 64, 128),
    ];
    let mut out = BTreeSet::new();
    for (w, wid) in weights {
        for (m, m_blocks, m8) in ms {
            // vLLM's `m_blocks <= 1`; m8 (its 0.5) has m_blocks 1 here, so it is small too.
            let small = m_blocks <= 1;
            for (tk, tn, threads) in all_configs {
                if threads == 256 {
                    if small && (tk, tn) != (128, 128) {
                        continue;
                    }
                    if !small && (tk, tn) != (64, 256) {
                        continue;
                    }
                }
                out.insert(Entry {
                    name: format!("marlin_ada_bf16_{w}_g128_{m}_k{tk}_n{tn}_t{threads}"),
                    weight_id: wid.to_string(),
                    threads,
                    m_blocks,
                    n_blocks: tn / 16,
                    k_blocks: tk / 16,
                    m8,
                });
            }
        }
    }
    out
}

#[test]
fn every_entry_name_matches_its_template_arguments() {
    let entries = parse_entries(&read("marlin_ada.cu"));
    assert!(
        !entries.is_empty(),
        "no MARLIN_ADA_ENTRY lines in marlin_ada.cu"
    );
    for e in &entries {
        let (w, m, tk, tn, threads) = decode_name(&e.name);
        let want_id = match w.as_str() {
            "u4b8" => "vllm::kU4B8Id",
            "u4" => "vllm::kU4Id",
            other => panic!("{}: unknown weight label {other}", e.name),
        };
        assert_eq!(e.weight_id, want_id, "{}: weight type", e.name);
        assert_eq!(e.threads, threads, "{}: threads", e.name);
        assert_eq!(e.k_blocks * 16, tk, "{}: thread_k", e.name);
        assert_eq!(e.n_blocks * 16, tn, "{}: thread_n", e.name);
        let (want_blocks, want_m8) = match m.as_str() {
            "m8" => (1, true),
            "m16" => (1, false),
            "m32" => (2, false),
            "m48" => (3, false),
            "m64" => (4, false),
            other => panic!("{}: unknown m label {other}", e.name),
        };
        assert_eq!(
            (e.m_blocks, e.m8),
            (want_blocks, want_m8),
            "{}: thread_m_blocks",
            e.name
        );
    }
}

#[test]
fn entry_set_is_vllms_instantiation_rule() {
    let entries = parse_entries(&read("marlin_ada.cu"));
    let names: BTreeSet<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names.len(), entries.len(), "duplicate entry names");
    let got: BTreeSet<Entry> = entries.iter().cloned().collect();
    let want = expected_set();
    let missing: Vec<_> = want.difference(&got).map(|e| e.name.clone()).collect();
    let extra: Vec<_> = got.difference(&want).map(|e| e.name.clone()).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "missing {missing:?}, extra {extra:?}"
    );
    assert_eq!(got.len(), 30);
}

#[test]
fn header_lists_every_exported_name() {
    // The file header is the API L18b binds; it must name each entry (once in the list, once
    // in its MARLIN_ADA_ENTRY line), and say how many there are.
    let src = read("marlin_ada.cu");
    assert!(
        src.contains("30 entry points"),
        "header must state the entry count"
    );
    for e in parse_entries(&src) {
        let n = src.matches(e.name.as_str()).count();
        assert!(
            n >= 2,
            "{}: listed {n} time(s); the header list must name it",
            e.name
        );
    }
}

#[test]
fn marlin_files_carry_the_copy_attribution() {
    for f in MARLIN_FILES {
        let src = read(f);
        let first = src.lines().next().unwrap_or_default();
        assert_eq!(
            first, "// SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0",
            "{f}: SPDX line"
        );
        assert!(
            src.contains("vllm-project/vllm, commit 8a36f08"),
            "{f}: vLLM source and commit"
        );
        assert!(
            src.contains("Changes (by Atlas contributors)"),
            "{f}: change notes"
        );
    }
    // The two files whose upstream carries the Marlin copyright keep its notice.
    for f in ["marlin_template_ada.cuh", "marlin_ada.cu"] {
        assert!(
            read(f).contains("Copyright (C) Marlin.2024 Elias Frantar"),
            "{f}: Marlin copyright"
        );
    }
}

#[test]
fn marlin_sources_exist_only_in_the_ada_tree() {
    // No other hardware tree may compile Marlin (GB10, Hopper, B200, B300, Metal and Strix
    // stay byte-identical), and no tree may reach these files through a symlink.
    let kernels = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
    let mut stack = vec![kernels.clone()];
    while let Some(dir) = stack.pop() {
        for ent in
            std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        {
            let ent = ent.expect("dir entry");
            let path = ent.path();
            let meta = std::fs::symlink_metadata(&path).expect("metadata");
            if meta.is_dir() {
                stack.push(path);
                continue;
            }
            let name = ent.file_name().to_string_lossy().into_owned();
            if MARLIN_FILES.contains(&name.as_str()) {
                let rel = path
                    .strip_prefix(&kernels)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                assert_eq!(
                    rel,
                    format!("ada/common/{name}"),
                    "{rel}: Marlin sources live only in kernels/ada/common"
                );
                assert!(
                    !meta.file_type().is_symlink(),
                    "{rel}: must be a real file, not a symlink"
                );
            }
        }
    }
}

/// Count `.param` declarations of a PTX `.entry` by name (same rule as kernel_arity.rs).
fn ptx_param_count(ptx: &str, kernel: &str) -> Option<usize> {
    let needle = format!(".entry {kernel}(");
    let start = ptx.find(&needle)?;
    let body = &ptx[start..];
    let close = body.find(')')?;
    Some(body[..close].matches(".param").count())
}

#[test]
#[ignore = "requires nvcc and AVAROK_SKIP_BUILD unset"]
fn compiled_ptx_carries_the_export_table_on_ada_only() {
    let sets = avarok_kernels::available_targets();
    if sets.iter().all(|s| s.modules.is_empty()) {
        eprintln!("no compiled PTX in this binary (stub build); marlin_ada PTX checks skipped");
        return;
    }
    let want: Vec<String> = expected_set().into_iter().map(|e| e.name).collect();
    let mut ada_targets = 0usize;
    for set in &sets {
        let marlin = set.modules.iter().find(|(m, _)| *m == "marlin_ada");
        let is_ada = set.target.arch == "sm_89";
        if !is_ada {
            assert!(
                marlin.is_none(),
                "target {} ({}) must not carry marlin_ada",
                set.target.model,
                set.target.arch
            );
            continue;
        }
        let Some((_, blob)) = marlin else {
            panic!("Ada target {} has no marlin_ada module", set.target.model);
        };
        let ptx = std::str::from_utf8(blob).expect("Ada PTX is text");
        for name in &want {
            assert_eq!(
                ptx_param_count(ptx, name),
                Some(ENTRY_PARAMS),
                "{}::marlin_ada::{name}: missing or wrong arity",
                set.target.model
            );
        }
        let entries = ptx.matches(".entry marlin_ada_").count();
        assert_eq!(
            entries,
            want.len(),
            "{}: unexpected marlin_ada entries",
            set.target.model
        );
        ada_targets += 1;
    }
    assert!(ada_targets >= 1, "no Ada target in this build");
}
