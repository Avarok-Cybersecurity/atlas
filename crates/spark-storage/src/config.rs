// SPDX-License-Identifier: AGPL-3.0-only
//
// Configuration shape for `--high-speed-swap`. All fields are required
// (PCND); validation runs at startup and a `HighSpeedSwap` orchestrator
// cannot be constructed with a partial / inconsistent config. The shape
// matches the locked CLI flag set in the plan.

use anyhow::{Result, bail};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Clone, Debug, Deserialize)]
pub struct HighSpeedSwapConfig {
    /// `--high-speed-swap-dir`: directory where per-layer KV files live.
    pub dir: PathBuf,
    /// `--high-speed-swap-bytes`: total disk budget; the layout fails fast
    /// if the budget can't fit `num_layers × bytes_per_layer`.
    pub bytes: u64,
    /// `--high-speed-swap-resident-blocks`: HBM scratch slot count. The
    /// scratch pool allocates exactly this many slots × per-slot bytes.
    pub resident_blocks: u32,
    /// `--high-speed-swap-rank`: predictor low-rank dimension.
    pub rank: u32,
    /// `--high-speed-swap-qd`: io_uring submission queue depth. Phase-3
    /// shows QD=8 reaches 3.4 GB/s on this hardware (random 64 KiB).
    pub qd: u32,
    /// `--high-speed-swap-graph`: capture the per-layer body in a CUDA
    /// graph and replay (Phase 4).
    pub graph: bool,
    /// Predictor seed; when omitted in CLI we'd error rather than default.
    pub projection_seed: u64,
}

impl HighSpeedSwapConfig {
    /// Validate cross-field invariants. Returns `Ok` only if the config is
    /// internally consistent and the directory is plausibly usable.
    pub fn validate(&self) -> Result<()> {
        if self.bytes == 0 {
            bail!("--high-speed-swap-bytes must be > 0");
        }
        if self.resident_blocks == 0 {
            bail!("--high-speed-swap-resident-blocks must be > 0");
        }
        if self.rank == 0 || self.rank > 128 {
            bail!(
                "--high-speed-swap-rank must be in 1..=128, got {}",
                self.rank
            );
        }
        if self.qd == 0 || self.qd > 64 {
            bail!("--high-speed-swap-qd must be in 1..=64, got {}", self.qd);
        }
        // The scratch pool must hold *at least one full tile*; tile_capacity
        // is taken to equal resident_blocks (single-tile fast path). Any
        // smaller would degrade to streaming-only-per-block.
        Ok(())
    }

    /// Validate, then ensure the directory exists or can be created.
    pub fn validate_and_prepare(&self) -> Result<()> {
        self.validate()?;
        std::fs::create_dir_all(&self.dir)
            .map_err(|e| anyhow::anyhow!("create {}: {e}", self.dir.display()))?;
        // Cross-mount check vs --swap-space-gb is performed at the
        // spark-server layer where we know both paths; out of scope here.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> HighSpeedSwapConfig {
        HighSpeedSwapConfig {
            dir: PathBuf::from("/tmp/atlas-hss-cfg"),
            bytes: 64 << 30,
            resident_blocks: 8192,
            rank: 32,
            qd: 8,
            graph: true,
            projection_seed: 0xCAFE_F00D,
        }
    }

    #[test]
    fn happy_path() {
        cfg().validate().unwrap();
    }

    #[test]
    fn rejects_zero_bytes() {
        let mut c = cfg();
        c.bytes = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_zero_resident_blocks() {
        let mut c = cfg();
        c.resident_blocks = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_out_of_range_rank() {
        let mut c = cfg();
        c.rank = 0;
        assert!(c.validate().is_err());
        c.rank = 129;
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejects_out_of_range_qd() {
        let mut c = cfg();
        c.qd = 0;
        assert!(c.validate().is_err());
        c.qd = 65;
        assert!(c.validate().is_err());
    }
}
