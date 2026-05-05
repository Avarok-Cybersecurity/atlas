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
    let dtypes = build_layer_kv_dtypes(spark_runtime::kv_cache::KvCacheDtype::Nvfp4, 12, 0);
    assert!(dtypes.is_empty());
}

#[test]
fn test_build_layer_kv_dtypes_bf16_noop() {
    // Already BF16 — no benefit from high-precision overlay
    let dtypes = build_layer_kv_dtypes(spark_runtime::kv_cache::KvCacheDtype::Bf16, 12, 2);
    assert!(dtypes.is_empty());
}

#[test]
fn test_build_layer_kv_dtypes_basic() {
    use spark_runtime::kv_cache::KvCacheDtype;
    let dtypes = build_layer_kv_dtypes(KvCacheDtype::Nvfp4, 12, 2);
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
    let dtypes = build_layer_kv_dtypes(KvCacheDtype::Fp8, 4, 3);
    assert_eq!(dtypes.len(), 4);
    for d in &dtypes {
        assert_eq!(*d, KvCacheDtype::Bf16);
    }
}

#[test]
fn test_build_layer_kv_dtypes_single_layer() {
    use spark_runtime::kv_cache::KvCacheDtype;
    let dtypes = build_layer_kv_dtypes(KvCacheDtype::Nvfp4, 1, 1);
    assert_eq!(dtypes.len(), 1);
    assert_eq!(dtypes[0], KvCacheDtype::Bf16);
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
