// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: crate root of the L04 routing-gate study (route_gate), a pure-Rust CPU port of the NumPy harness.
//!
//! Question: can a cheap signal tell, before generation, that the 4B Qwen3.5 model will FAIL a prompt? The routing
//! extension is released only if it can. Gate (master plan L04): AUROC of "4B fails" for the readout plus a logistic
//! head `>= 0.70` AND `>= length/keyword baseline + 0.05`. Nothing here is in the Atlas serving path; everything runs
//! on the CPU, with no GPU, no NumPy, no torch and no transformers. The `route_gate` binary (`src/bin/route_gate.rs`)
//! drives the subcommands `build`, `collect`, `grade`, `summary`, `rows`, `goldens`, `tokens` and `fit`.
//!
//! Map of the modules (each row replaces one Python file of the old harness):
//!
//! | module | Python | job |
//! |---|---|---|
//! | [`tensorio`] | `safetensors_min.py` | memory-mapped safetensors reader (BF16 widening) and writer |
//! | [`stage_one`] | `stage_one_np.py` | float64 Stage-1 head: p of each candidate, four usage legs |
//! | [`signals`] | `signals_np.py` | PII and topic heads, with a searched grid for undocumented choices |
//! | [`metrics`] | `metrics_np.py` | AUROC, bootstrap, stratified CV, ridge logistic regression, calibration |
//! | [`fit_eval`] | `fit_eval.py` | the study: arms, gate, report |
//! | [`prep`] | `prep.py` | build the prompts, collect replies from an Atlas server, grade, summary |
//! | [`grading`] | `grading.py` | answer extraction and sandboxed code execution |
//! | [`encoder`], [`serializer`], [`goldens`], [`rows`] | `arc_rows.py` | CPU readouts, the goldens oracle, token check |
//! | [`lm_check`], [`lm_texts`] | (new) | full-LM oracle of the encoder on the stock checkpoint (`encoder_check` binary) |
//!
//! Attribution: the Stage-1 head is a translation of Apache-2.0 code (Atlas Futures, Inc.) and the readouts follow the
//! Apache-2.0 ARC-1.0 release documentation; see `RAYLINE-ROUTING-ATTRIBUTION.md` and `NOTICE-rayline.txt`.

pub mod canonical;
pub mod cli;
pub mod encoder;
pub mod error;
pub mod fit_eval;
pub mod goldens;
pub mod grading;
pub mod lm_check;
pub mod lm_texts;
pub mod mat;
pub mod metrics;
pub mod mt;
pub mod prep;
pub mod probe;
pub mod rows;
pub mod serializer;
pub mod signals;
pub mod stage_one;
pub mod tensorio;
pub mod token_report;

#[cfg(test)]
mod tests_encoder;
#[cfg(test)]
mod tests_encoder_rope;
#[cfg(test)]
mod tests_fit_eval;
#[cfg(test)]
mod tests_grading;
#[cfg(test)]
mod tests_lm_check;
#[cfg(test)]
mod tests_metrics;
#[cfg(test)]
mod tests_mt;
#[cfg(test)]
mod tests_prep;
#[cfg(test)]
mod tests_rows;
#[cfg(test)]
mod tests_serializer;
#[cfg(test)]
mod tests_signals;
#[cfg(test)]
mod tests_stage_one;
#[cfg(test)]
mod tests_stage_one_head;
#[cfg(test)]
mod tests_tensorio;
