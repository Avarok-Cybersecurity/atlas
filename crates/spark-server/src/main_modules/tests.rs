// SPDX-License-Identifier: AGPL-3.0-only

use clap::Parser;

use crate::cli::{Cli, Command};
use crate::main_modules::build_layer_kv_dtypes;

#[test]
fn test_cli_parse_positional_model() {
    let cli = Cli::try_parse_from([
        "spark",
        "serve",
        "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
        "--port",
        "9999",
        "--max-seq-len",
        "8192",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Serve(args) => {
            assert_eq!(
                args.model.as_deref(),
                Some("nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4"),
            );
            assert!(args.model_from_path.is_none());
            assert_eq!(args.port, 9999);
            assert_eq!(args.max_seq_len, 8192);
            assert_eq!(args.gpu_memory_utilization, 0.90);
            assert_eq!(args.scheduling_policy, "fifo");
            assert_eq!(args.tbt_deadline_ms, 100);
        }
    }
}

#[test]
fn test_cli_parse_model_from_path() {
    let cli = Cli::try_parse_from([
        "spark",
        "serve",
        "--model-from-path",
        "/tmp/model",
        "--port",
        "8888",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Serve(args) => {
            assert!(args.model.is_none());
            assert_eq!(
                args.model_from_path,
                Some(std::path::PathBuf::from("/tmp/model")),
            );
        }
    }
}

#[test]
fn test_cli_parse_slai_policy() {
    let cli = Cli::try_parse_from([
        "spark",
        "serve",
        "nvidia/model",
        "--scheduling-policy",
        "slai",
        "--tbt-deadline-ms",
        "50",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Serve(args) => {
            assert_eq!(args.scheduling_policy, "slai");
            assert_eq!(args.tbt_deadline_ms, 50);
        }
    }
}

#[test]
fn test_build_layer_kv_dtypes_disabled() {
    // high_precision_layers=0 returns empty vec (backward compatible)
    let dtypes = build_layer_kv_dtypes(
        spark_runtime::kv_cache::KvCacheDtype::Nvfp4,
        12,
        0,
        spark_runtime::kv_cache::KvCacheDtype::Bf16,
    );
    assert!(dtypes.is_empty());
}

#[test]
fn test_build_layer_kv_dtypes_bf16_noop() {
    // Already BF16 — no benefit from high-precision overlay
    let dtypes = build_layer_kv_dtypes(
        spark_runtime::kv_cache::KvCacheDtype::Bf16,
        12,
        2,
        spark_runtime::kv_cache::KvCacheDtype::Bf16,
    );
    assert!(dtypes.is_empty());
}

#[test]
fn test_build_layer_kv_dtypes_basic() {
    use spark_runtime::kv_cache::KvCacheDtype;
    let dtypes = build_layer_kv_dtypes(
        KvCacheDtype::Nvfp4,
        12,
        2,
        spark_runtime::kv_cache::KvCacheDtype::Bf16,
    );
    assert_eq!(dtypes.len(), 12);
    // First 2: BF16
    assert_eq!(dtypes[0], KvCacheDtype::Bf16);
    assert_eq!(dtypes[1], KvCacheDtype::Bf16);
    // Middle 8: NVFP4
    for i in 2..10 {
        assert_eq!(dtypes[i], KvCacheDtype::Nvfp4, "layer {i}");
    }
    // Last 2: BF16
    assert_eq!(dtypes[10], KvCacheDtype::Bf16);
    assert_eq!(dtypes[11], KvCacheDtype::Bf16);
}

#[test]
fn test_build_layer_kv_dtypes_overlap() {
    use spark_runtime::kv_cache::KvCacheDtype;
    // 4 layers, hp=3 → all become BF16 (first 3 and last 3 overlap)
    let dtypes = build_layer_kv_dtypes(
        KvCacheDtype::Fp8,
        4,
        3,
        spark_runtime::kv_cache::KvCacheDtype::Bf16,
    );
    assert_eq!(dtypes.len(), 4);
    for d in &dtypes {
        assert_eq!(*d, KvCacheDtype::Bf16);
    }
}

#[test]
fn test_build_layer_kv_dtypes_single_layer() {
    use spark_runtime::kv_cache::KvCacheDtype;
    let dtypes = build_layer_kv_dtypes(
        KvCacheDtype::Nvfp4,
        1,
        1,
        spark_runtime::kv_cache::KvCacheDtype::Bf16,
    );
    assert_eq!(dtypes.len(), 1);
    assert_eq!(dtypes[0], KvCacheDtype::Bf16);
}

#[test]
fn test_auto_high_precision_layers_non_turbo_none() {
    use spark_runtime::kv_cache::KvCacheDtype;
    for d in [KvCacheDtype::Bf16, KvCacheDtype::Fp8, KvCacheDtype::Nvfp4] {
        assert_eq!(crate::main_modules::auto_high_precision_layers(d, 10), None);
    }
}

#[test]
fn test_auto_high_precision_layers_baseline_formula() {
    use spark_runtime::kv_cache::KvCacheDtype;
    // ceil(n/3), floor 2 — flagship (10 attn layers) lands on 4
    assert_eq!(
        crate::main_modules::auto_high_precision_layers(KvCacheDtype::Turbo8, 10),
        Some(4),
    );
    assert_eq!(
        crate::main_modules::auto_high_precision_layers(KvCacheDtype::Turbo4KTurbo8V, 3),
        Some(2),
    );
}

#[test]
fn test_auto_high_precision_layers_weak_dtypes_stronger_default() {
    use spark_runtime::kv_cache::KvCacheDtype;
    // turbo2 + bf16k_turbo3v: 0/5-0/10 agentic at hp=4, 5/5 at hp=8 on the
    // 10-attn-layer flagship; ceil(4n/5) floor 4 reproduces the validated 8.
    for d in [KvCacheDtype::Turbo2, KvCacheDtype::Bf16KTurbo3V] {
        assert_eq!(
            crate::main_modules::auto_high_precision_layers(d, 10),
            Some(8)
        );
        assert_eq!(
            crate::main_modules::auto_high_precision_layers(d, 2),
            Some(4)
        );
    }
}

#[test]
fn test_auto_high_precision_layers_every_turbo_dtype_covered() {
    use spark_runtime::kv_cache::KvCacheDtype;
    // Every turbo dtype must auto-enable boundary layers; only the three
    // non-rotated baseline dtypes opt out. Enum-walk so a future variant
    // cannot silently fall through to hp=0.
    for d in [
        KvCacheDtype::Turbo2,
        KvCacheDtype::Turbo3,
        KvCacheDtype::Turbo4,
        KvCacheDtype::Turbo8,
        KvCacheDtype::Turbo4KTurbo3V,
        KvCacheDtype::Turbo4KTurbo8V,
        KvCacheDtype::Turbo3KTurbo8V,
        KvCacheDtype::Bf16KTurbo4V,
        KvCacheDtype::Bf16KTurbo3V,
        KvCacheDtype::Fp8KTurbo4V,
        KvCacheDtype::Fp8KTurbo3V,
        KvCacheDtype::Bf16KTurbo2V,
        KvCacheDtype::Fp8KTurbo2V,
    ] {
        assert!(
            crate::main_modules::auto_high_precision_layers(d, 10).unwrap_or(0) >= 2,
            "{d:?} must auto-enable high-precision boundary layers",
        );
    }
}

#[test]
fn test_cli_parse_kv_high_precision_layers() {
    let cli = Cli::try_parse_from([
        "spark",
        "serve",
        "nvidia/model",
        "--kv-high-precision-layers",
        "3",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Serve(args) => {
            assert_eq!(args.kv_high_precision_layers, "3");
        }
    }
}

#[test]
fn test_cli_default_kv_high_precision_layers() {
    let cli = Cli::try_parse_from(["spark", "serve", "nvidia/model"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Serve(args) => {
            assert_eq!(args.kv_high_precision_layers, "0");
        }
    }
}
