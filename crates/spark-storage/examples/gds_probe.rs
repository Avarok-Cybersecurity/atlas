// SPDX-License-Identifier: AGPL-3.0-only
//
// Phase-0 probe binary. Run with:
//
//     cargo run --release -p spark-storage --example gds-probe -- \
//         --dir /workspace/atlas-swap-probe
//
// Reports whether cuFile/GDS is available on the current host and benchmarks
// the candidate backends. Output is human-readable and ends with a single
// JSON line for downstream tooling.

use std::path::PathBuf;

use anyhow::{Result, bail};
use spark_storage::{ProbeConfig, run_probe};

fn parse_args() -> Result<ProbeConfig> {
    let mut dir: Option<PathBuf> = None;
    let mut bytes: u64 = 256 * 1024 * 1024;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => {
                dir = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--dir requires a value"))?,
                ));
            }
            "--test-bytes" => {
                bytes = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--test-bytes requires a value"))?
                    .parse()?;
            }
            "--help" | "-h" => {
                println!("usage: gds-probe --dir <path> [--test-bytes <N>]");
                std::process::exit(0);
            }
            other => bail!("unknown arg: {other}"),
        }
    }
    Ok(ProbeConfig {
        dir: dir.ok_or_else(|| anyhow::anyhow!("--dir is required (PCND: no default)"))?,
        test_file_bytes: bytes,
    })
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();
    let cfg = parse_args()?;
    eprintln!("== Atlas storage probe ==");
    eprintln!("dir            : {}", cfg.dir.display());
    eprintln!(
        "test file size : {} MiB",
        cfg.test_file_bytes / (1024 * 1024)
    );
    let result = run_probe(&cfg)?;
    eprintln!();
    eprintln!("libcufile.so loaded     : {}", result.libcufile_loaded);
    if let Some(e) = &result.libcufile_load_error {
        eprintln!("  libcufile load error  : {e}");
    }
    if let Some(v) = result.cufile_version {
        eprintln!("cuFile version          : {v}");
    }
    eprintln!("nvidia-fs kmod loaded   : {}", result.nvidia_fs_kmod_loaded);
    eprintln!("cuFileDriverOpen ok     : {}", result.cufile_driver_open_ok);
    if let Some(e) = &result.cufile_driver_open_error {
        eprintln!("  cuFileDriverOpen error: {e}");
    }
    eprintln!();
    eprintln!("Bandwidth (MiB/s):");
    let row = |label, v: Option<f64>| match v {
        Some(x) => eprintln!("  {label:<26} {x:>9.1}"),
        None => eprintln!("  {label:<26} {:>9}", "—"),
    };
    row("cuFile seq 4 MiB", result.cufile_seq_4mib);
    row("cuFile rand 64 KiB", result.cufile_rand_64kib);
    row("io_uring seq 4 MiB", result.io_uring_seq_4mib);
    row("io_uring rand 64 KiB", result.io_uring_rand_64kib);
    row("posix seq 4 MiB", result.posix_seq_4mib);
    row("posix rand 64 KiB", result.posix_rand_64kib);
    eprintln!();
    eprintln!("Recommended backend     : {:?}", result.recommended);
    eprintln!("Reason                  : {}", result.recommendation_reason);
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
