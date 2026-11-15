// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the measured-ceiling table: per card,
//! model, weight format, KV dtype and speculation mode, the longest
//! `--max-seq-len` a real serve booted with AND survived a full-length prefill
//! at, and the preflight rule that lets such a measurement override the
//! formula only DOWNWARD.
//!
//! # The method (after the MiaAI planner)
//!
//! A formula predicts whether a configuration fits; on a 16 GB card it is
//! right until a driver or display reservation, a fragmentation effect or an
//! allocation nobody charged makes it wrong by a few hundred MiB, and the serve
//! then dies mid-prefill instead of refusing at boot. The MiaAI kit keeps its
//! formula but grows a real prefill until it fails under a stated budget, and
//! where such a measurement exists it wins, but only to LOWER the answer: a
//! measurement never admits a configuration the formula refuses. Atlas's
//! formula is preflight's reserve check and the KV stage's budget; this table
//! adds the measured refusal in front of them.
//!
//! # Rows
//!
//! A row is a receipt, never an estimate. It names the card (by its total
//! device memory, the number `--gpu-memory-utilization` multiplies), the model
//! (`model_type`, hidden size, layer count), the weight format, the KV dtype,
//! whether the serve speculated, the batch it was measured at, the longest
//! context that survived, and the log it came from. A ceiling measured at
//! batch B also bounds every batch above B (more sequences only take memory),
//! so a serve is checked against every matching row measured at a batch at or
//! below its own, and the tightest wins.
//!
//! The table is EMPTY at write time: no context ceiling has been measured on
//! Ada (`docs/porting/consumer-nvidia-ada.md`, "Largest `--max-seq-len` that
//! serves at batch 1: not measured"). The GPU pass fills it from
//! a script outside this repository (cell `ceiling`), on
//! a clean card each time (a ceiling measured beside other processes
//! is contention, not a ceiling). With no row nothing is refused, so every
//! serve behaves exactly as before.
//!
//! # Scope
//!
//! Only a target that declares `memory_fit_consumer` (ada) consults the table;
//! `AVAROK_MEASURED_CEILING_OFF` (presence) turns a refusal into a warning.

use anyhow::{Result, bail};
use avarok_core::config::ModelConfig;
use spark_runtime::kv_cache::KvCacheDtype;

use super::headroom::PostLoadInputs;
use crate::cli;

/// How the weights reach the device, as far as the reserve cares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WeightsTag {
    /// A GGUF file (dequantized or kept packed by the GGUF loader).
    Gguf,
    /// Safetensors with the load-time BF16 -> FP8 FFN route freeing memory.
    Fp8Ffn,
    /// Safetensors loaded as the checkpoint stores them.
    Checkpoint,
}

impl WeightsTag {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::Fp8Ffn => "fp8-ffn",
            Self::Checkpoint => "checkpoint",
        }
    }
}

/// One measured ceiling.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeasuredCeiling {
    /// The card, for the log (`RTX 4060 Ti 16 GB`).
    pub(crate) card: &'static str,
    /// Inclusive range of `gpu.total_memory()` in MiB the row applies to.
    pub(crate) total_mib: (usize, usize),
    pub(crate) model_type: &'static str,
    pub(crate) hidden_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) weights: WeightsTag,
    pub(crate) kv_dtype: KvCacheDtype,
    pub(crate) speculative: bool,
    /// `--max-batch-size` of the measurement.
    pub(crate) max_batch_size: usize,
    /// The longest `--max-seq-len` that booted and survived a full prefill.
    pub(crate) max_seq_len: usize,
    /// Where the receipt is (pass log path and date).
    pub(crate) receipt: &'static str,
}

/// The receipts. Empty until the GPU pass measures one (see the module docs).
pub(crate) const MEASURED: &[MeasuredCeiling] = &[];

/// What a serve is, in the table's terms.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ServeKey<'a> {
    pub(crate) total_mib: usize,
    pub(crate) model_type: &'a str,
    pub(crate) hidden_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) weights: WeightsTag,
    pub(crate) kv_dtype: KvCacheDtype,
    pub(crate) speculative: bool,
    pub(crate) max_batch_size: usize,
}

/// The tightest row that bounds `key`, if any.
pub(crate) fn tightest<'t>(
    table: &'t [MeasuredCeiling],
    key: &ServeKey<'_>,
) -> Option<&'t MeasuredCeiling> {
    table
        .iter()
        .filter(|r| {
            (r.total_mib.0..=r.total_mib.1).contains(&key.total_mib)
                && r.model_type == key.model_type
                && r.hidden_size == key.hidden_size
                && r.num_hidden_layers == key.num_hidden_layers
                && r.weights == key.weights
                && r.kv_dtype == key.kv_dtype
                && r.speculative == key.speculative
                && r.max_batch_size <= key.max_batch_size
        })
        .min_by_key(|r| r.max_seq_len)
}

/// The rule: `Ok(None)` when no row bounds the serve or the context is within
/// the measured ceiling; `Ok(Some(warning))` when a row is exceeded but the
/// operator opted out; `Err` when a row is exceeded. Never admits anything:
/// the formula's own checks run after this either way.
pub(crate) fn judge(
    table: &[MeasuredCeiling],
    key: &ServeKey<'_>,
    max_seq_len: usize,
    opted_out: bool,
) -> Result<Option<String>> {
    let Some(row) = tightest(table, key) else {
        return Ok(None);
    };
    if max_seq_len <= row.max_seq_len {
        return Ok(None);
    }
    let why = format!(
        "--max-seq-len {max_seq_len} exceeds the measured ceiling of {} tokens for {} ({} weights, \
         {} KV, {}speculative, measured at --max-batch-size {}; receipt: {}). The memory formula \
         admits it, but the measurement says a full-length prefill does not survive on this card",
        row.max_seq_len,
        row.card,
        row.weights.name(),
        row.kv_dtype.name(),
        if row.speculative { "" } else { "not " },
        row.max_batch_size,
        row.receipt,
    );
    if opted_out {
        return Ok(Some(format!(
            "{why}; serving anyway (AVAROK_MEASURED_CEILING_OFF)"
        )));
    }
    bail!(
        "{why}. Lower --max-seq-len to {} or below, or set AVAROK_MEASURED_CEILING_OFF to serve \
         against the formula alone.",
        row.max_seq_len
    )
}

/// The weight format of the checkpoint preflight is about to load.
fn weights_tag(post_load: &PostLoadInputs<'_>) -> WeightsTag {
    let is_gguf = |p: &std::path::Path| p.extension().is_some_and(|x| x == "gguf");
    let dir = post_load.model_dir;
    let gguf = is_gguf(dir)
        || std::fs::read_dir(dir)
            .map(|it| it.flatten().any(|e| is_gguf(&e.path())))
            .unwrap_or(false);
    if gguf {
        WeightsTag::Gguf
    } else if post_load.fp8_ffn_freed > 0 {
        WeightsTag::Fp8Ffn
    } else {
        WeightsTag::Checkpoint
    }
}

/// Preflight's check, before the reserve is priced. Inert off the consumer
/// targets and with an empty table.
pub(super) fn check(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    post_load: &PostLoadInputs<'_>,
) -> Result<()> {
    let consumer = spark_model::layers::ops::target_defaults::resolved()
        .memory_fit_consumer
        .value;
    if !consumer {
        return Ok(());
    }
    let key = ServeKey {
        total_mib: post_load.total_mem / (1024 * 1024),
        model_type: &config.model_type,
        hidden_size: config.hidden_size,
        num_hidden_layers: config.num_hidden_layers,
        weights: weights_tag(post_load),
        kv_dtype: post_load.kv_dtype,
        speculative: args.speculative
            || args.self_speculative
            || args.ngram_speculative
            || args.dflash,
        max_batch_size: args.max_batch_size,
    };
    let opted_out = std::env::var_os("AVAROK_MEASURED_CEILING_OFF").is_some();
    if let Some(warning) = judge(MEASURED, &key, args.max_seq_len, opted_out)? {
        tracing::warn!("Preflight measured ceiling: {warning}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROW: MeasuredCeiling = MeasuredCeiling {
        card: "RTX 4060 Ti 16 GB",
        total_mib: (16_000, 16_400),
        model_type: "qwen3_5",
        hidden_size: 5120,
        num_hidden_layers: 64,
        weights: WeightsTag::Gguf,
        kv_dtype: KvCacheDtype::Fp8,
        speculative: true,
        max_batch_size: 1,
        max_seq_len: 40_960,
        receipt: "test row",
    };

    fn key(batch: usize) -> ServeKey<'static> {
        ServeKey {
            total_mib: 16_380,
            model_type: "qwen3_5",
            hidden_size: 5120,
            num_hidden_layers: 64,
            weights: WeightsTag::Gguf,
            kv_dtype: KvCacheDtype::Fp8,
            speculative: true,
            max_batch_size: batch,
        }
    }

    #[test]
    fn the_shipped_table_refuses_nothing() {
        // A row needs a receipt (see the module docs); until the GPU pass
        // records one, no context is refused.
        for batch in [1, 8, 64] {
            assert!(
                judge(MEASURED, &key(batch), 262_144, false)
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn a_measured_ceiling_only_lowers_the_answer() {
        let t = [ROW];
        assert!(judge(&t, &key(1), 40_960, false).unwrap().is_none());
        assert!(judge(&t, &key(1), 8_192, false).unwrap().is_none());
        let e = judge(&t, &key(1), 65_536, false).unwrap_err();
        let msg = format!("{e:#}");
        assert!(msg.contains("40960") && msg.contains("test row"), "{msg}");
        let w = judge(&t, &key(1), 65_536, true).unwrap().unwrap();
        assert!(w.contains("AVAROK_MEASURED_CEILING_OFF"), "{w}");
    }

    #[test]
    fn a_ceiling_bounds_larger_batches_and_not_smaller_ones() {
        let at4 = MeasuredCeiling {
            max_batch_size: 4,
            max_seq_len: 16_384,
            ..ROW
        };
        let t = [ROW, at4];
        // Batch 8: both rows bound it, the tighter (16K) wins.
        assert_eq!(tightest(&t, &key(8)).unwrap().max_seq_len, 16_384);
        // Batch 1: only the batch-1 row applies.
        assert_eq!(tightest(&t, &key(1)).unwrap().max_seq_len, 40_960);
    }

    #[test]
    fn a_row_for_another_card_format_or_mode_does_not_apply() {
        let t = [ROW];
        let mut k = key(1);
        k.total_mib = 24_564;
        assert!(tightest(&t, &k).is_none());
        let mut k = key(1);
        k.weights = WeightsTag::Fp8Ffn;
        assert!(tightest(&t, &k).is_none());
        let mut k = key(1);
        k.kv_dtype = KvCacheDtype::Bf16;
        assert!(tightest(&t, &k).is_none());
        let mut k = key(1);
        k.speculative = false;
        assert!(tightest(&t, &k).is_none());
        let mut k = key(1);
        k.hidden_size = 2560;
        assert!(tightest(&t, &k).is_none());
    }
}
