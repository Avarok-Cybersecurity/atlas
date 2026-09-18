// SPDX-License-Identifier: AGPL-3.0-only

//! Control vectors (activation steering) on the mHC highway.
//!
//! A control vector is a per-layer direction `v` in the residual stream,
//! applied at the END of a layer — after that layer's `hc_post`, before the
//! next layer reads the highway. Two arms:
//!
//! ```text
//! project:  h <- h - s * (h . v) * v     (v unit norm; s = 1 fully ablates)
//! add:      h <- h + s * v
//! ```
//!
//! **No weight is read or written.** The highway is FP32 whatever the weights
//! are quantized to, so NVFP4 never enters this arithmetic and none of the
//! LoRA-under-NVFP4 hazards apply. See
//! `docs/design/qwen4exp-control-vectors.md`.
//!
//! # Boot-time only
//!
//! A control vector changes every hidden state, so the KV it produces differs
//! from the KV produced without it. Making it per-request would require the
//! prefix cache to be keyed by cvec identity; until that exists this is
//! resolved ONCE at model construction and carried, never read per request and
//! never read from the environment on a forward path.
//!
//! # Fails closed, loudly
//!
//! Every validation error here aborts the boot. A silently-skipped projection
//! serves a model the operator believes has been steered, and unlike a
//! performance regression there is no counter that would show it.

use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::{GgmlType, GgufFile};

/// Must match `CVEC_BLOCK` in `control_vector.cu`. The dot product is a block
/// reduction, so its rounding — and therefore, on a speculative path, the
/// accepted text — is a function of this. Changing it needs a re-gate.
const CVEC_BLOCK: u32 = 256;

/// Which intervention the vector applies.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CvecMode {
    /// `h -= s * (h . v) * v`. The vector is unit-normalized at load and its
    /// original norm folded into the per-layer scale, exactly as llama.cpp
    /// does, so a file of unit vectors at user scale 1.0 gives s = 1.0.
    #[default]
    Project,
    /// `h += s * v`. A different vector and a much smaller scale (~0.1).
    Add,
}

/// What the operator asked for, before the file is read.
#[derive(Clone, Debug)]
pub struct ControlVectorSpec {
    pub path: PathBuf,
    pub scale: f32,
    /// Inclusive layer range, in model layer indices.
    pub layer_start: usize,
    pub layer_end: usize,
    pub mode: CvecMode,
    /// The model this is being loaded onto, checked against the file's
    /// `controlvector.model_hint`. `None` skips the check — used by the
    /// CPU-only unit tests, which have no model.
    pub model_type: Option<String>,
}

/// A loaded control vector, owned by the model and borrowed by the forward
/// paths. Dropping the model frees the table.
pub struct ControlVector {
    /// `[n_layer, hidden]` F32, row `il` = the direction for layer `il`.
    /// Rows outside the active range are zero AND their scale is zero, so an
    /// out-of-range layer is skipped at the host without a launch.
    directions: DevicePtr,
    /// Per-layer scale; `0.0` means "this layer takes no intervention".
    per_layer_scale: Vec<f32>,
    hidden: usize,
    mode: CvecMode,
    project_k: KernelHandle,
    add_k: KernelHandle,
    cos_k: KernelHandle,
    /// SHA-256 of the vector file.
    pub sha256: String,
    /// The resolved configuration, retained because the request-visible
    /// identity has to depend on it — see [`Self::config_identity`].
    scale: f32,
    layer_start: usize,
    layer_end: usize,
    /// `AVAROK_CVEC_PROBE=1`: log `mean|cos(h, v)|` either side of every
    /// application. Resolved ONCE here and carried, per the levers rule — the
    /// consumer runs per layer per forward pass, which is exactly where an
    /// environment read is forbidden.
    ///
    /// The probe synchronizes and copies D2H, so it is illegal inside CUDA
    /// graph capture. Run it with `AVAROK_DEBUG_NO_GRAPH=1`; on GB10 that
    /// costs essentially nothing, because decode graphs measured
    /// speed-neutral on this model (16.4 replay vs 16.5 eager).
    probe: bool,
}

/// Reduce a model identifier to its comparable core.
///
/// `controlvector.model_hint` follows llama.cpp's naming and Atlas's
/// `model_type` follows its own, so the SAME model is spelled `qwen4exp` in the
/// published refusal projection and `qwen4_exp` in this engine. Comparing those
/// literally would reject the one artifact this feature shipped for, which is
/// why the check normalises rather than demanding equality: lowercase, and drop
/// everything that is not alphanumeric.
fn normalize_model_id(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Reject a file that is not a control vector, or is one for another model.
///
/// Geometry alone does not establish identity. Two unrelated models can share a
/// hidden size, and a direction derived for one applied to the other is not an
/// error anywhere downstream — it is a quiet quality regression with no counter
/// that would ever attribute it.
fn validate_identity(gguf: &GgufFile, spec: &ControlVectorSpec) -> Result<()> {
    // Both known producers write this: llama.cpp's exporter and
    // `scripts/derive_control_vector.py`. Requiring it costs nothing real and
    // turns "someone passed a model shard by mistake" into an immediate,
    // legible failure rather than a confusing tensor-name error.
    let arch = gguf.get_str("general.architecture").unwrap_or_default();
    ensure!(
        arch == "controlvector",
        "control vector {}: general.architecture is {:?}, expected \"controlvector\". \
         This file is not a control vector.",
        spec.path.display(),
        arch
    );

    // Advisory in the file, load-bearing here. Absent is tolerated — older
    // hand-built vectors predate the convention — but a hint that disagrees
    // with the model is refused.
    if let (Some(hint), Some(model)) = (
        gguf.get_str("controlvector.model_hint"),
        spec.model_type.as_deref(),
    ) && normalize_model_id(hint) != normalize_model_id(model)
    {
        bail!(
            "control vector {}: controlvector.model_hint is {hint:?} but this model is \
             {model:?}. A direction derived on a different model is not an error further \
             down — matching hidden sizes make it load and steer, and the only symptom is \
             output that is quietly worse. Re-derive against this model, or pass the \
             intended file.",
            spec.path.display()
        );
    }
    Ok(())
}

/// The host half of the load: parse the GGUF and build the `[n_layer, hidden]`
/// table plus the per-layer scales.
///
/// Split out from [`ControlVector::load`] with no GPU in the signature so the
/// validation is testable in CI, which is CPU-only.
pub fn build_table(
    bytes: &[u8],
    spec: &ControlVectorSpec,
    hidden: usize,
    n_layer: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    ensure!(
        spec.layer_start <= spec.layer_end,
        "control vector layer range {}..={} is empty",
        spec.layer_start,
        spec.layer_end
    );
    ensure!(
        spec.layer_end < n_layer,
        "control vector layer range {}..={} exceeds the model's {n_layer} layers",
        spec.layer_start,
        spec.layer_end
    );
    ensure!(
        spec.scale.is_finite(),
        "control vector scale {} is not finite",
        spec.scale
    );

    let gguf = GgufFile::parse(bytes).context("parsing the control-vector GGUF")?;
    validate_identity(&gguf, spec)?;
    let mut table = vec![0.0f32; n_layer * hidden];
    let mut scales = vec![0.0f32; n_layer];
    let mut found = 0usize;
    // Row norms of the ACTIVE layers under `add`, to catch a unit-normalised
    // file being used with the operator that needs raw magnitudes.
    let mut add_norms: Vec<f64> = Vec::new();

    for t in &gguf.tensors {
        let Some(suffix) = t.name.strip_prefix("direction.") else {
            bail!(
                "control vector: unexpected tensor {:?}; a control-vector GGUF \
                 holds only `direction.<layer>` tensors",
                t.name
            );
        };
        let il: usize = suffix
            .parse()
            .with_context(|| format!("control vector: tensor {:?} has no layer index", t.name))?;
        ensure!(
            t.ggml_type == GgmlType::F32,
            "control vector: {:?} is {:?}, expected F32",
            t.name,
            t.ggml_type
        );
        ensure!(
            t.dims == [hidden],
            "control vector: {:?} has dims {:?}, expected [{hidden}] \
             (the model's hidden_size)",
            t.name,
            t.dims
        );
        // llama.cpp has no layer-0 direction and neither do we: layer 0's
        // output is the first thing every later layer reads, and the
        // reference generator never emits one.
        ensure!(
            il > 0 && il < n_layer,
            "control vector: {:?} is outside the model's layers 1..{}",
            t.name,
            n_layer - 1
        );
        found += 1;
        if il < spec.layer_start || il > spec.layer_end {
            continue; // outside the active range: leave the row zeroed
        }

        let off = gguf.tensor_abs_offset(t);
        let end = off + hidden * 4;
        ensure!(
            end <= bytes.len(),
            "control vector: {:?} runs past the end of the file",
            t.name
        );
        let row = &mut table[il * hidden..(il + 1) * hidden];
        for (d, chunk) in row.iter_mut().zip(bytes[off..end].chunks_exact(4)) {
            *d = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        ensure!(
            row.iter().all(|x| x.is_finite()),
            "control vector: {:?} contains a non-finite value",
            t.name
        );

        match spec.mode {
            CvecMode::Project => {
                // Store the UNIT direction and fold its norm into the scale,
                // mirroring llama.cpp so a scaled file behaves identically
                // there and here. A unit file at user scale 1.0 gives s = 1.0.
                let norm = (row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>()).sqrt();
                ensure!(
                    norm > 0.0,
                    "control vector: {:?} is the zero vector, which would \
                     silently no-op this layer",
                    t.name
                );
                for x in row.iter_mut() {
                    *x = (*x as f64 / norm) as f32;
                }
                scales[il] = spec.scale * norm as f32;
            }
            CvecMode::Add => {
                let norm = (row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>()).sqrt();
                ensure!(
                    norm > 0.0,
                    "control vector: {:?} is the zero vector, which would \
                     silently no-op this layer",
                    t.name
                );
                add_norms.push(norm);
                scales[il] = spec.scale;
            }
        }
    }

    // `add` applies the row VERBATIM under one global scalar, so a file of
    // unit rows is the wrong artifact for it — and wrong in the quietest
    // possible way. The published projection vector is unit-normalised, and
    // using it here at scale 1.0 displaces the residual stream by ~1 against a
    // norm of order 1e0..1e1, i.e. a nudge that produces output indistinguishable
    // from no steering at all. That reads in a results table as "add mode does
    // nothing" — a dose of zero published as a null result. It cost a full
    // sweep here before anything noticed.
    //
    // Detected by MEASURING rather than by metadata, so it also catches the
    // llama.cpp-era files that predate any magnitude convention. A genuine
    // raw-magnitude file cannot look like this: |mean-diff| tracks the stream
    // norm, which grows an order of magnitude across depth, so every active
    // layer landing within 1% of exactly 1.0 is a normalisation signature and
    // nothing else.
    if spec.mode == CvecMode::Add
        && add_norms.len() > 1
        && add_norms.iter().all(|n| (n - 1.0).abs() < 1e-2)
    {
        bail!(
            "control vector {}: every layer has |v| ~ 1.0, so this is a UNIT-normalised \
             file, and it was loaded in `add` mode.\n`add` applies the row verbatim under \
             one global scale, so unit rows make the dose meaningless — typically far too \
             small to do anything, which looks exactly like the vector having no effect.\n\
             Use the raw-magnitude file for `add` (`derive_control_vector.py --magnitude \
             raw`), or use `project` mode with this one, which folds each row's norm into \
             the per-layer scale and is what unit rows are for.",
            spec.path.display()
        );
    }

    ensure!(
        found > 0,
        "control vector: the file has no `direction.*` tensors"
    );
    let active = scales.iter().filter(|s| **s != 0.0).count();
    ensure!(
        active > 0,
        "control vector: no direction falls inside layers {}..={} — the file \
         covers {found} layers and this would be a silent no-op",
        spec.layer_start,
        spec.layer_end
    );
    Ok((table, scales))
}

impl ControlVector {
    /// Read, validate and upload. Any problem aborts the boot.
    pub fn load(
        gpu: &dyn GpuBackend,
        spec: &ControlVectorSpec,
        hidden: usize,
        n_layer: usize,
    ) -> Result<Self> {
        let bytes = std::fs::read(&spec.path)
            .with_context(|| format!("reading control vector {}", spec.path.display()))?;
        let sha256 = {
            use sha2::{Digest, Sha256};
            format!("{:x}", Sha256::digest(&bytes))
        };
        let (table, per_layer_scale) = build_table(&bytes, spec, hidden, n_layer)?;

        let bytes_len = table.len() * 4;
        let directions = gpu.alloc(bytes_len).context("allocating the cvec table")?;
        let raw = unsafe { std::slice::from_raw_parts(table.as_ptr() as *const u8, bytes_len) };
        gpu.copy_h2d(raw, directions)
            .context("uploading the cvec table")?;

        let active: Vec<usize> = per_layer_scale
            .iter()
            .enumerate()
            .filter(|(_, s)| **s != 0.0)
            .map(|(i, _)| i)
            .collect();
        tracing::info!(
            "control vector {}: mode={:?} layers {}..={} ({} active) scale={} sha256={}",
            spec.path.display(),
            spec.mode,
            spec.layer_start,
            spec.layer_end,
            active.len(),
            spec.scale,
            sha256
        );

        Ok(Self {
            directions,
            per_layer_scale,
            hidden,
            mode: spec.mode,
            project_k: gpu.kernel("control_vector", "cvec_project_highway")?,
            add_k: gpu.kernel("control_vector", "cvec_add_highway")?,
            cos_k: gpu.kernel("control_vector", "cvec_cos_highway")?,
            sha256,
            scale: spec.scale,
            layer_start: spec.layer_start,
            layer_end: spec.layer_end,
            probe: std::env::var("AVAROK_CVEC_PROBE").as_deref() == Ok("1"),
        })
    }

    /// Everything that changes the arithmetic, as one canonical string.
    ///
    /// The per-request id is derived from this and not from the name alone,
    /// and that distinction is the whole point. Two ranks can each register
    /// `"refusal"` while loading different FILES, MODES, SCALES or LAYER
    /// RANGES. A name-derived id matches in every one of those cases, the
    /// worker's "do I have this id?" check passes, and the ranks then steer
    /// differently — silently, because nothing in the protocol disagrees. The
    /// halves of the model simply compute different things.
    ///
    /// Folding the configuration in makes the ids *diverge* whenever the
    /// configuration diverges, so the existing worker-side check stops being
    /// decorative and starts failing closed.
    ///
    /// It also retires a second hazard: a different file loaded under an
    /// existing name now yields a different id, so prefix-cache entries
    /// computed under the old vector can no longer be served as hits.
    ///
    /// `scale` is hashed via `to_bits` because it is a config value being
    /// compared for exact identity, not a number being compared for
    /// closeness: 1.0 and 1.0000001 are different configurations and must get
    /// different ids. (`to_bits` also makes this total — there is no NaN
    /// scale, since the parser rejects non-finite values, but relying on that
    /// from a distance would be fragile.)
    pub fn config_identity(&self) -> String {
        format!(
            "sha256={} mode={:?} scale=0x{:08x} layers={}..={}",
            self.sha256,
            self.mode,
            self.scale.to_bits(),
            self.layer_start,
            self.layer_end,
        )
    }

    /// Free the device table.
    ///
    /// Consuming, because the table pointer is the whole object: anything that
    /// could still call `apply` after this would be reading freed memory, and
    /// taking ownership makes that a compile error rather than a rule.
    pub fn release(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.directions)
            .context("freeing the control-vector table")
    }

    /// Whether the cosine probe is armed. Read per layer, so this must stay a
    /// field access and never become an environment read.
    #[inline]
    pub fn probe_enabled(&self) -> bool {
        self.probe
    }

    /// Whether this layer takes an intervention. Cheap enough to call per
    /// layer — it is a `Vec` index, not an environment read.
    #[inline]
    pub fn applies_to(&self, layer_idx: usize) -> bool {
        self.per_layer_scale
            .get(layer_idx)
            .is_some_and(|s| *s != 0.0)
    }

    /// Row `layer_idx` of the table.
    #[inline]
    fn row(&self, layer_idx: usize) -> DevicePtr {
        DevicePtr(self.directions.0 + (layer_idx * self.hidden * 4) as u64)
    }

    /// Apply the intervention to `[num_tokens, hc_mult, hidden]` FP32 at
    /// `highway`, in place.
    ///
    /// `highway` must already be offset to the first row this call owns — the
    /// K-row verify and mixed steps do not start at highway row 0.
    ///
    /// Graph-capture legal: a pure stream-ordered launch against a boot-time
    /// allocation, with no synchronize, no H2D and no environment read.
    pub fn apply(
        &self,
        gpu: &dyn GpuBackend,
        highway: DevicePtr,
        layer_idx: usize,
        num_tokens: usize,
        hc_mult: usize,
        stream: u64,
    ) -> Result<()> {
        if num_tokens == 0 || !self.applies_to(layer_idx) {
            return Ok(());
        }
        let scale = self.per_layer_scale[layer_idx];
        let v = self.row(layer_idx);
        match self.mode {
            CvecMode::Project => KernelLaunch::new(gpu, self.project_k)
                .grid([num_tokens as u32, hc_mult as u32, 1])
                .block([CVEC_BLOCK, 1, 1])
                .shared_mem((self.hidden as u32 + 32) * 4)
                .arg_ptr(highway)
                .arg_ptr(v)
                .arg_f32(scale)
                .arg_u32(self.hidden as u32)
                .arg_u32(hc_mult as u32)
                .launch(stream),
            CvecMode::Add => {
                let n = (num_tokens * hc_mult * self.hidden) as u32;
                KernelLaunch::new(gpu, self.add_k)
                    .grid([n.div_ceil(CVEC_BLOCK), 1, 1])
                    .block([CVEC_BLOCK, 1, 1])
                    .arg_ptr(highway)
                    .arg_ptr(v)
                    .arg_f32(scale)
                    .arg_u32(self.hidden as u32)
                    .arg_u32(n)
                    .launch(stream)
            }
        }
    }

    /// Mean `|cos(h, v_layer)|` over every (token, stream) of `highway`.
    ///
    /// The validation gate: run it either side of [`Self::apply`] and `post`
    /// must be ~0 while `pre` is clearly non-zero. Synchronizes and copies
    /// D2H, so it is a debug path and must never run inside graph capture.
    pub fn cos_probe(
        &self,
        gpu: &dyn GpuBackend,
        highway: DevicePtr,
        layer_idx: usize,
        num_tokens: usize,
        hc_mult: usize,
        stream: u64,
    ) -> Result<f32> {
        if num_tokens == 0 || !self.applies_to(layer_idx) {
            return Ok(0.0);
        }
        let n = num_tokens * hc_mult;
        let out = gpu.alloc(n * 4).context("cvec probe scratch")?;
        let launched = KernelLaunch::new(gpu, self.cos_k)
            .grid([num_tokens as u32, hc_mult as u32, 1])
            .block([CVEC_BLOCK, 1, 1])
            .shared_mem((self.hidden as u32 + 64) * 4)
            .arg_ptr(highway)
            .arg_ptr(self.row(layer_idx))
            .arg_ptr(out)
            .arg_u32(self.hidden as u32)
            .arg_u32(hc_mult as u32)
            .launch(stream);
        let mean = launched.and_then(|()| {
            let mut host = vec![0u8; n * 4];
            gpu.copy_d2h_on_stream(out, &mut host, stream)?;
            let vals: &[f32] =
                unsafe { std::slice::from_raw_parts(host.as_ptr() as *const f32, n) };
            Ok(vals.iter().map(|v| *v as f64).sum::<f64>() as f32 / n as f32)
        });
        // Free on both arms: a probe that leaks on error turns a diagnostic
        // run into an OOM several layers later.
        let _ = gpu.free(out);
        mean
    }
}

#[cfg(test)]
#[path = "control_vector_tests.rs"]
mod control_vector_tests;
