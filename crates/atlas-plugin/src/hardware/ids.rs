// SPDX-License-Identifier: AGPL-3.0-only

//! Box-class ids — the strings a gate baseline is keyed by.
//!
//! A benchmark's thresholds live under `kernels/<hw>/<model>/BENCH.toml` and
//! are assembled into `GateBaseline.hardware`, keyed by the `<hw>` directory
//! name. Which key a run is scored against comes from two directions:
//!
//! * an operator typing `spark benchmark run <id> --hardware h100`, and
//! * a record's own fingerprint, via [`super::Hardware::gate_key`].
//!
//! Those two agreeing is not automatic. The baseline is only evidence of what
//! has ALREADY been measured, so it cannot answer "is `h100` a box class we
//! recognise?" — before the first Hopper run, an `h100` typed at the command
//! line and an `h800` typed by mistake look identical to it. They are not the
//! same situation: one is fixed by running the gate on the box, the other by
//! correcting the spelling, and a caller told the wrong one hunts for a typo
//! that is not there.
//!
//! [`KNOWN_HARDWARE_IDS`] is that registry — the ids Atlas recognises,
//! independent of what has been measured on them.

/// Every box class Atlas recognises, whether or not any gate has a record on
/// it yet.
///
/// SSOT for `--hardware` validation. An id belongs here once the project has
/// decided it is a class worth scoring separately — which is a threshold
/// question, not a hardware-availability one: two boxes share an id only if a
/// ceiling measured on one is meaningful on the other.
///
/// Kept in step with the `kernels/<hw>/HARDWARE.toml` directories by
/// `every_kernel_hardware_dir_is_registered` below. The list is deliberately
/// wider than that tree: `h100` and `h200` are registered before any Hopper
/// kernels land, so the resolver can tell an operator "no record yet" instead
/// of "unknown hardware" for the whole span of the porting campaign.
///
/// TWO KINDS OF ID LIVE HERE, and the difference is worth stating because it
/// is not visible from the strings:
///
/// * **architecture classes** — `gb10`, `hopper`, `b200`, `metal`, `strix`.
///   These are `kernels/<hw>/` directory names. One kernel set is compiled per
///   architecture, so this is the unit a `-arch=` and a PTX artefact exist for.
/// * **bench SKUs** — `h100`, `h200`, `gh200`, `b200`, `gb200`, `mi300x`.
///   These are what an operator types at `--hardware`, what
///   `GateBaseline.hardware` is keyed by, and what the campaign results
///   template's columns are. This is the unit a THRESHOLD is meaningful for.
///
/// They coincide for `gb10` and `b200` and diverge for `hopper`, which is one
/// architecture (SM 9.0) across two SKUs whose memory bandwidth differs by
/// 43%. Both kinds must be registered: the arch classes because
/// `every_kernel_hardware_dir_is_registered` demands it, the SKUs because
/// `--hardware` validates against this list.
pub const KNOWN_HARDWARE_IDS: [&str; 11] = [
    // NVIDIA GB10 / DGX Spark — the box every committed record was measured on.
    // Architecture class and SKU at once.
    "gb10",
    // NVIDIA Hopper, the architecture class: `kernels/hopper/` (sm_90a).
    "hopper",
    // …and its two SKUs. Two ids, not one: H200 is the same SM with 141 GB of
    // HBM3e at 4.8 TB/s against H100's 80 GB at 3.35 TB/s, and every metric a
    // gate carries (TTFT, TPOT, node tok/s) moves with memory bandwidth. One
    // shared id would score an H100 run against H200 numbers.
    "h100",
    "h200",
    // NVIDIA Grace-Hopper superchip. Hopper silicon, but its own class: the GPU
    // sits behind NVLink-C2C against LPDDR5X Grace memory rather than a PCIe
    // host, so a TTFT ceiling measured on an H100 board says nothing about it.
    "gh200",
    // NVIDIA Blackwell datacentre, SM 10.0 — `kernels/b200/` (sm_100a). Here
    // the architecture class and the discrete SXM SKU are the same id, so this
    // entry wears both hats. GB200 is the Grace-Blackwell superchip: same
    // silicon and same kernel set, its own baseline key, for exactly the
    // reason `gh200` has one. B300 / GB300 are SM 10.3 and get NEITHER — a
    // different arch (`sm_103a`, not forward-compatible from `sm_100a`) that
    // Atlas ships no target for, so registering them would promise a build
    // that does not exist.
    "b200",
    "gb200",
    // Apple Silicon, via the Metal backend.
    "metal",
    // AMD Strix Halo — Vulkan and HIP are separate targets and separate
    // classes; the same silicon through two backends does not produce
    // interchangeable numbers.
    "strix",
    "strix-hip",
    // AMD Instinct. No kernels in the tree yet; registered because it is the id
    // `hardware_id_from_gpu_name` maps the SKU onto, and the resolver fixtures
    // already use it as a second box class.
    "mi300x",
];

/// Every GPU-name token that names a box class, and the class it names.
///
/// Matched as a WHOLE token, never a substring: `gh200` contains `h200`, and a
/// substring match would file every Grace-Hopper run under the H200 baseline.
///
/// Deliberately short. A SKU earns an entry only when the project has decided
/// its numbers are worth scoring separately — everything absent falls through
/// to [`super::Hardware::gate_key`]'s normalisation, which keeps the capacity
/// and generation in the key (`a100sxm480gb`) and so cannot silently merge two
/// parts. Absent is therefore the SAFE default, and the reason this returns
/// `Option` rather than guessing.
const SKU_TOKENS: [(&str, &str); 7] = [
    ("gb10", "gb10"),
    ("h100", "h100"),
    ("h200", "h200"),
    ("gh200", "gh200"),
    ("b200", "b200"),
    ("gb200", "gb200"),
    ("mi300x", "mi300x"),
];

/// Map a GPU's reported name onto the box class its numbers belong to.
///
/// The name is what `nvidia-smi --query-gpu=name` (or `rocm-smi`) answers, and
/// it is free text carrying marketing SKU detail: `"NVIDIA H100 80GB HBM3"`,
/// `"NVIDIA H100 PCIe"`, `"NVIDIA H200 NVL"`. Normalising it by stripping
/// punctuation — which is all [`super::Hardware::gate_key`] could do before
/// this — turns those three into `h10080gbhbm3`, `h100pcie` and `h200nvl`, so
/// three Hopper boxes key against three baselines that do not exist while the
/// `h100` and `h200` entries sit unused.
///
/// `None` means "no opinion", not "unknown box": the caller keeps its existing
/// normalisation, which never merges two parts. That is why `"NVIDIA B300"`
/// and the A100 capacities are absent rather than mapped — a wrong merge is
/// silent, a missing entry costs one line. B300/GB300 are SM 10.3 and Atlas
/// compiles nothing for them; the A100 capacities differ only in memory, so a
/// family-level guess would merge two parts whose numbers are not comparable.
///
/// One SKU family, one id. `H100 PCIe` and `H100 SXM` share `h100` even though
/// their bandwidths differ, because the campaign's unit of comparison is the
/// family; the exact SKU survives verbatim in `Hardware::gpu` and has its own
/// column in the campaign results template, so the distinction is never lost —
/// only the baseline key is coarse.
pub fn hardware_id_from_gpu_name(name: &str) -> Option<&'static str> {
    name.split(|c: char| !c.is_ascii_alphanumeric())
        .map(str::to_ascii_lowercase)
        .find_map(|token| {
            SKU_TOKENS
                .iter()
                .find(|(sku, _)| *sku == token)
                .map(|(_, id)| *id)
        })
}

/// True when `id` names a box class Atlas recognises.
///
/// Case- and shape-sensitive on purpose: the id is a directory name and a
/// baseline key, so `H100` is not `h100` any more than it is on disk.
pub fn is_known_hardware_id(id: &str) -> bool {
    KNOWN_HARDWARE_IDS.contains(&id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Oracle: the registry's own contract — the ids the Hopper campaign types
    /// at `--hardware` must be recognised before the first record exists, or
    /// the resolver cannot distinguish "go measure it" from "you typoed".
    #[test]
    fn the_hopper_slots_are_registered_before_they_are_measured() {
        assert!(is_known_hardware_id("h100"));
        assert!(is_known_hardware_id("h200"));
        assert!(is_known_hardware_id("gb10"));
    }

    /// Oracle: the same contract, negative side. `h800` is a real NVIDIA part
    /// that Atlas has never registered, and `H100` is the right part spelled
    /// the wrong way — a baseline key is a directory name, so case matters.
    #[test]
    fn an_unregistered_or_miscased_id_is_not_known() {
        assert!(!is_known_hardware_id("h800"));
        assert!(!is_known_hardware_id("H100"));
        assert!(!is_known_hardware_id(""));
        // B300 / GB300 are SM 10.3 (`sm_103a`), a different architecture from
        // B200's 10.0 with no forward compatibility between them, and Atlas
        // ships no target for either. They hold the slot `b200` held before
        // `kernels/b200/` landed.
        assert!(!is_known_hardware_id("b300"));
        assert!(!is_known_hardware_id("gb300"));
    }

    /// Oracle: the two things this registry is asked about. `hopper` and
    /// `b200` are `kernels/<hw>/` directory names — architecture classes,
    /// which is the unit a kernel set is compiled for. `h100`, `h200` and
    /// `b200` are what an operator types at `--hardware` and what the campaign
    /// results template columns are — SKUs, which is the unit a THRESHOLD is
    /// meaningful for. `b200` is both, because on Blackwell datacentre the two
    /// units coincide; `hopper` is one arch across two SKUs, and that is why
    /// this list holds all of them rather than one set.
    #[test]
    fn the_kernel_arch_classes_and_the_bench_skus_are_both_registered() {
        for arch_class in ["gb10", "hopper", "b200"] {
            assert!(is_known_hardware_id(arch_class), "{arch_class}");
        }
        for sku in ["h100", "h200", "b200", "gb200"] {
            assert!(is_known_hardware_id(sku), "{sku}");
        }
    }

    /// Oracle: the GPU-name strings `nvidia-smi --query-gpu=name` actually
    /// answers on each part, and `rocm-smi`'s equivalent on AMD.
    #[test]
    fn each_sku_spelling_lands_on_its_box_class() {
        for name in ["NVIDIA GB10", "GB10", "nvidia gb10"] {
            assert_eq!(hardware_id_from_gpu_name(name), Some("gb10"), "{name}");
        }
        for name in [
            "NVIDIA H100 80GB HBM3",
            "NVIDIA H100 PCIe",
            "NVIDIA H100-SXM5-80GB",
        ] {
            assert_eq!(hardware_id_from_gpu_name(name), Some("h100"), "{name}");
        }
        for name in ["NVIDIA H200", "NVIDIA H200 NVL"] {
            assert_eq!(hardware_id_from_gpu_name(name), Some("h200"), "{name}");
        }
        assert_eq!(
            hardware_id_from_gpu_name("AMD Instinct MI300X"),
            Some("mi300x")
        );
    }

    /// Oracle: NVIDIA's own product separation. GH200 is Hopper silicon, and
    /// the tempting answer is `h100` — but the GPU hangs off NVLink-C2C against
    /// Grace's LPDDR5X rather than a PCIe host, so it is its own class. It also
    /// has to survive the substring trap: `gh200` CONTAINS `h200`, and a
    /// substring match would file every Grace-Hopper run under the H200
    /// baseline, which is the exact silent-merge this table exists to prevent.
    #[test]
    fn a_grace_hopper_superchip_is_neither_h100_nor_h200() {
        for name in ["NVIDIA GH200 480GB", "NVIDIA GH200 120GB"] {
            assert_eq!(hardware_id_from_gpu_name(name), Some("gh200"), "{name}");
        }
    }

    /// ★ Oracle: the known-bad. A part the table has no entry for must answer
    /// `None` — never the nearest-looking id. `B200` is Blackwell, not Hopper;
    /// mapping it onto `h100`/`h200` would score a Blackwell run against Hopper
    /// thresholds and report the result as a pass.
    ///
    /// The A100 capacities are the same failure one step subtler: they differ
    /// only in memory, so a family-level guess would merge them. `None` hands
    /// them back to the normalisation that keeps the capacity in the key.
    #[test]
    fn an_unlisted_part_answers_none_rather_than_the_nearest_id() {
        for name in [
            // Blackwell ULTRA. B300/GB300 are SM 10.3 and `sm_103a` PTX is a
            // different, non-interchangeable target from B200's `sm_100a`;
            // Atlas compiles neither. Filing them under `b200` would score a
            // box Atlas cannot even build for against B200 thresholds.
            "NVIDIA B300",
            "NVIDIA GB300",
            "NVIDIA A100-SXM4-40GB",
            "NVIDIA A100-SXM4-80GB",
            "NVIDIA L40S",
            "AMD Radeon 8060S (gfx1151)",
            "",
        ] {
            let got = hardware_id_from_gpu_name(name);
            assert!(got.is_none(), "{name:?} must not map anywhere, got {got:?}");
        }
    }

    /// Oracle: the GPU-name strings `nvidia-smi --query-gpu=name` answers on
    /// Blackwell datacentre parts, and NVIDIA's own product separation.
    ///
    /// B200 is a discrete SXM board; GB200 is the Grace-Blackwell superchip,
    /// where the GPU sits behind NVLink-C2C against Grace LPDDR5X instead of a
    /// PCIe host. Same reasoning that gave GH200 its own id rather than
    /// folding it into `h100`: the interconnect and the host memory are what
    /// TTFT is made of, so a ceiling measured on one says nothing about the
    /// other. They share `kernels/b200/` (both are SM 10.0) and they do NOT
    /// share a baseline key.
    ///
    /// `gb200` also has to survive the substring trap `gh200` had: it CONTAINS
    /// `b200`, and a substring match would file every Grace-Blackwell run
    /// under the B200 baseline.
    #[test]
    fn each_blackwell_datacentre_spelling_lands_on_its_box_class() {
        for name in [
            "NVIDIA B200",
            "NVIDIA B200 180GB HBM3e",
            "NVIDIA B200-SXM-180GB",
        ] {
            assert_eq!(hardware_id_from_gpu_name(name), Some("b200"), "{name}");
        }
        for name in ["NVIDIA GB200", "NVIDIA GB200 NVL72"] {
            assert_eq!(hardware_id_from_gpu_name(name), Some("gb200"), "{name}");
        }
    }

    /// ★ Oracle: the known-bad this table's `None` default was protecting
    /// while B200 had no entry. Now that it HAS one, the wrong answer is no
    /// longer `None` — it is `h100` or `h200`.
    ///
    /// Blackwell datacentre is SM 10.0; Hopper is 9.0. A B200 run scored
    /// against Hopper thresholds would pass a gate it never met, and nothing
    /// in the record would say so. Asserted as its own case rather than left
    /// implied by the equality above, because this is the failure that costs
    /// something.
    #[test]
    fn a_blackwell_datacentre_part_is_never_filed_under_hopper() {
        for name in [
            "NVIDIA B200",
            "NVIDIA B200 180GB HBM3e",
            "NVIDIA GB200",
            "NVIDIA GB200 NVL72",
        ] {
            let got = hardware_id_from_gpu_name(name);
            assert_ne!(got, Some("h100"), "{name}");
            assert_ne!(got, Some("h200"), "{name}");
            assert_ne!(got, Some("gh200"), "{name}");
        }
    }

    /// Oracle: the registry above. An id the SKU table can produce but
    /// `--hardware` refuses as unknown would mean a box could write records the
    /// operator cannot ask for — the two halves have to name the same set.
    #[test]
    fn every_id_the_sku_table_produces_is_registered() {
        for (sku, id) in SKU_TOKENS {
            assert!(
                is_known_hardware_id(id),
                "{sku} maps to {id:?}, which is not in KNOWN_HARDWARE_IDS"
            );
        }
    }

    /// Oracle: the `kernels/<hw>/HARDWARE.toml` tree — the thing that actually
    /// produces baseline keys (`gate::bench::load_all`).
    ///
    /// A backend whose kernels are in the tree but whose id is not registered
    /// would have its `--hardware` refused as "unknown" the moment its last
    /// BENCH.toml entry went `unmeasured`. Registration is one line; noticing
    /// that regression from a benchmark refusal is not.
    #[test]
    fn every_kernel_hardware_dir_is_registered() {
        let kernels = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
        let mut found = Vec::new();
        for entry in std::fs::read_dir(&kernels)
            .expect("kernels/ is in the tree")
            .flatten()
        {
            if !entry.path().join("HARDWARE.toml").exists() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            assert!(
                is_known_hardware_id(&name),
                "kernels/{name}/ declares a HARDWARE.toml but {name:?} is not in \
                 KNOWN_HARDWARE_IDS; add it there or --hardware {name} reads as a typo"
            );
            found.push(name);
        }
        assert!(
            !found.is_empty(),
            "read no hardware dirs at all — the walk is broken, not the registry"
        );
    }
}
