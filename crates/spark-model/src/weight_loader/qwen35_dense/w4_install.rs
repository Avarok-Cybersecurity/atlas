// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the one place the dense Qwen3.5 loader meets the 4-bit (`W4Linear`)
//! routes: the per-layer installs the loader's hook lines call (affine FFN / GDN /
//! attention by lane L23a's quantizers, the NVFP4 FFN by this lane, keep-packed GGUF FFN
//! tensors), the load-time self-check, the residency tally, the summary lines and the `/metrics`
//! record. The per-load plan and its types live in the child `w4_plan.rs`, re-exported here.
//!
//! **THE TYPES L23a CODES AGAINST** are [`PlanInputs`],
//! [`RoutePlan`] and [`LayerCtx`]. L23a's `plan` functions are silent and look nothing up with
//! their row off; its `install` functions quantize, adopt and free, and do NOT install into the
//! layer, tally residency or self-check: this module does all three, so the boot line, the gauge
//! and the self-check have one owner.
//!
//! **ORDER.** The FFN weights are installed with `DenseFfnLayer::set_w4` BEFORE the layer's
//! `finalize_q4k_load` / `finalize_nvfp4_mmq_load` (a request from lane L21: those then reserve the
//! shared W4 scratch and skip the NVFP4 repacks); the GDN and attention pairs are installed right
//! after their layer is built, followed by the layer's own scratch reservation, so no CUDA graph
//! capture allocates.
//!
//! **OFF BY DEFAULT.** Every 4-bit row is off on every target, so [`W4Load::resolve`] returns an
//! inert plan without a lookup or a log line, every install returns at once, and the loader
//! builds exactly what it built before. The only visible effect of an inert load is that
//! `/metrics` reports zero 4-bit bytes.

use anyhow::{Context, Result};
use avarok_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::affine4_selfcheck::SelfChecks;
use super::fp8_residency::DerivedResidency;
use super::{affine4_from_bf16, attn_affine4_from_bf16, gdn_affine4_from_bf16};
use super::{ct_install, nvfp4_from_bf16, packed_install};
use crate::layers::{DenseFfnLayer, Qwen3AttentionLayer, Qwen3SsmLayer};
use crate::port::marlin::compressed::CtW4Scheme;
use crate::port::w4::{W4Box, W4Format};
use crate::port::w4_stats::{self, RoleRecord, W4Role};
use crate::weight_map::{DenseWeight, Nvfp4Variant};

#[path = "w4_plan.rs"]
mod plan;
#[cfg(test)]
pub use plan::nvfp4_plan;
pub use plan::{
    Fp8Wins, LayerCtx, LoadPlans, PlanInputs, ROW_OFF, Route, RoutePlan, boot_line,
    dense_loader_takes, plan_formats,
};

/// What one layer type installed on this load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    /// The format of the first installed weight (a GGUF FFN may mix Q4_K and Q6_K, and an affine
    /// route may serve some projections through Marlin; the gauge names one).
    pub format: Option<W4Format>,
    pub layers: u32,
    /// Resident device bytes of the installed weights (the packed ones included).
    pub bytes: u64,
    /// BF16 bytes the from-BF16 routes replaced (0 for keep-packed tensors).
    pub bf16_freed: u64,
}

impl Tally {
    pub fn add(&mut self, boxes: &[W4Box], from_bf16: bool) {
        self.format = self.format.or(boxes.first().map(|w| w.format()));
        self.layers += 1;
        self.bytes += boxes.iter().map(|w| w.device_bytes() as u64).sum::<u64>();
        if from_bf16 {
            self.bf16_freed += boxes
                .iter()
                .map(|w| (w.n() * w.k() * 2) as u64)
                .sum::<u64>();
        }
    }
}

fn slot(role: W4Role) -> usize {
    W4Role::ALL.iter().position(|&r| r == role).unwrap_or(0)
}

/// One installed layer type's summary line.
pub fn summary_line(role: W4Role, t: &Tally, per_row: bool) -> String {
    let gb = |b: u64| b as f64 / 1e9;
    let freed = if t.bf16_freed > 0 {
        format!("{:.2} GB of BF16 replaced", gb(t.bf16_freed))
    } else {
        "kept packed from the checkpoint".to_string()
    };
    let format = t.format.map(|f| f.to_string()).unwrap_or_default();
    let tail = if per_row {
        "; self-check failed, batched rows run one row at a time"
    } else {
        ""
    };
    format!(
        "[avarok] 4-bit {}: {format} on {} layers, {:.2} GB resident, {freed}{tail}",
        role.label(),
        t.layers,
        gb(t.bytes),
    )
}

/// The 4-bit side of one dense load.
pub struct W4Load<'a> {
    gpu: &'a dyn GpuBackend,
    store: &'a WeightStore,
    config: &'a ModelConfig,
    stream: u64,
    plans: LoadPlans,
    checks: SelfChecks,
    tally: [Tally; 4],
    /// The compressed-tensors `pack-quantized` W4A16 scheme (`ct_install.rs`): `None` for every
    /// other checkpoint, `Some(Err)` when the checkpoint is one but its scheme is not served.
    ct: Option<std::result::Result<CtW4Scheme, String>>,
}

impl<'a> W4Load<'a> {
    /// Plan this load's routes and print one boot line per route whose row is on. `fp8` is what
    /// the loader's own FP8 routes resolved to. With every 4-bit row off: no lookup, no line.
    pub fn resolve(
        gpu: &'a dyn GpuBackend,
        store: &'a WeightStore,
        config: &'a ModelConfig,
        variant: Nvfp4Variant,
        stream: u64,
        fp8: Fp8Wins,
    ) -> Self {
        let bf16 = matches!(variant, Nvfp4Variant::Bf16Raw);
        let lora = config.adapter_max_rank > 0;
        let plans = LoadPlans::plan(gpu, bf16, config.tp_world_size, lora, fp8);
        for (route, p) in plans.routes() {
            if let Some(line) = boot_line(route, &p) {
                tracing::info!("{line}");
            }
        }
        Self::with_plans(gpu, store, config, stream, plans)
    }

    /// A load with explicit plans (tests, and [`Self::resolve`]).
    pub fn with_plans(
        gpu: &'a dyn GpuBackend,
        store: &'a WeightStore,
        config: &'a ModelConfig,
        stream: u64,
        plans: LoadPlans,
    ) -> Self {
        Self {
            gpu,
            store,
            config,
            stream,
            plans,
            checks: SelfChecks::default(),
            tally: [Tally::default(); 4],
            ct: ct_install::resolve_scheme(store, config),
        }
    }

    /// True when this is a compressed-tensors `pack-quantized` checkpoint.
    fn ct_on(&self) -> bool {
        self.ct.is_some()
    }

    /// The served compressed-tensors scheme, or the load error that says why it is not served.
    fn ct_scheme(&self) -> Result<CtW4Scheme> {
        match &self.ct {
            Some(Ok(s)) => Ok(s.clone()),
            Some(Err(e)) => Err(anyhow::anyhow!(
                "compressed-tensors pack-quantized checkpoint not served: {e}"
            )),
            None => Err(anyhow::anyhow!(
                "not a compressed-tensors pack-quantized checkpoint"
            )),
        }
    }

    fn ctx<'b>(&'b self, lp: &'b str) -> LayerCtx<'b> {
        LayerCtx {
            gpu: self.gpu,
            store: self.store,
            stream: self.stream,
            lp,
            config: self.config,
        }
    }

    /// Self-check the boxes (the first layer of each type), tally them and, for the from-BF16
    /// routes (`residency` given), count their device bytes as kept residency.
    fn accept<const N: usize>(
        &mut self,
        role: W4Role,
        layer: usize,
        boxes: [W4Box; N],
        residency: Option<&mut DerivedResidency>,
    ) -> Result<[W4Box; N]> {
        let from_bf16 = residency.is_some();
        if let Some(r) = residency {
            r.keep(boxes.iter().map(|w| w.device_bytes()).sum());
        }
        let boxes = self
            .checks
            .apply(self.gpu, role, layer, boxes, self.stream)?;
        self.tally[slot(role)].add(&boxes, from_bf16);
        Ok(boxes)
    }

    /// This layer's FFN through its from-BF16 route (affine or NVFP4), checked and tallied;
    /// `None` when no route runs or the layer does not qualify (then nothing was touched). On
    /// `Some` the three BF16 checkpoint tensors are freed.
    pub fn ffn_boxes(
        &mut self,
        lp: &str,
        layer: usize,
        residency: &mut DerivedResidency,
    ) -> Result<Option<[W4Box; 3]>> {
        let plan = self.plans.ffn();
        if !plan.on() {
            return Ok(None);
        }
        let boxes = match self.plans.nvfp4.filter(|_| !self.plans.ffn_affine.on()) {
            Some(nv) => {
                let (h, inter) = (self.config.hidden_size, super::ffn_inter(self.config));
                if !nvfp4_from_bf16::layer_qualifies(self.store, lp, h, inter) {
                    return Ok(None);
                }
                nv.quantize_ffn(self.store, lp, h, inter, self.gpu, self.stream)?
            }
            None => match affine4_from_bf16::install(&plan, &self.ctx(lp))? {
                Some(b) => b,
                None => return Ok(None),
            },
        };
        self.accept(W4Role::Ffn, layer, boxes, Some(residency))
            .map(Some)
    }

    /// Hook of the loader's BF16 FFN arm (a layer built with the BF16 overlay aliasing the store
    /// tensors and a NULL NVFP4 fallback), called before the layer's finalizers.
    pub fn install_ffn(
        &mut self,
        dffn: &mut DenseFfnLayer,
        lp: &str,
        layer: usize,
        residency: &mut DerivedResidency,
    ) -> Result<()> {
        if let Some([g, u, d]) = self.ffn_boxes(lp, layer, residency)? {
            // The detached overlay aliased the store tensors the route just freed.
            let _detached = dffn.set_w4(g, u, d);
        }
        Ok(())
    }

    /// The store holds a keep-packed 4-bit tensor for this layer's FFN.
    pub fn ffn_packed(&self, lp: &str) -> bool {
        packed_install::ffn_any_packed(self.store, lp)
            || (self.ct_on() && ct_install::ffn_any(self.store, lp))
    }

    /// This layer's keep-packed FFN as `W4Linear`s borrowing the store's buffers (GGUF) or built
    /// from the checkpoint's int4 codes (compressed-tensors, Marlin), checked and tallied.
    pub fn packed_ffn_boxes(&mut self, lp: &str, layer: usize) -> Result<[W4Box; 3]> {
        if self.ct_on() && ct_install::ffn_any(self.store, lp) {
            let scheme = self.ct_scheme()?;
            let boxes =
                ct_install::ffn_boxes(self.gpu, self.store, self.config, lp, &scheme, self.stream)?;
            return self.accept(W4Role::Ffn, layer, boxes, None);
        }
        let gpu = self.gpu;
        let boxes = packed_install::ffn_boxes(
            self.store,
            lp,
            self.config.hidden_size,
            super::ffn_inter(self.config),
            self.config.tp_world_size,
            self.config.adapter_max_rank > 0,
            |t| crate::port::gguf_q4::from_weight_tensor(gpu, t),
        )?;
        self.accept(W4Role::Ffn, layer, boxes, None)
    }

    /// Hook of the loader's FFN arm when [`Self::ffn_packed`] is true (the layer was built with a
    /// NULL NVFP4 fallback and no BF16 overlay), called before the layer's finalizers.
    pub fn install_packed_ffn(
        &mut self,
        dffn: &mut DenseFfnLayer,
        lp: &str,
        layer: usize,
    ) -> Result<()> {
        let [g, u, d] = self.packed_ffn_boxes(lp, layer)?;
        let _detached = dffn.set_w4(g, u, d);
        Ok(())
    }

    /// The store holds a keep-packed 4-bit tensor for this attention layer's q/k/v/o (`p` is
    /// `<layer>.self_attn`).
    pub fn attn_packed(&self, p: &str) -> bool {
        packed_install::attn_any_packed(self.store, p)
            || (self.ct_on() && ct_install::attn_any(self.store, p))
    }

    /// Hook of the loader's attention arm when [`Self::attn_packed`] is true: the layer was built
    /// with NULL q/k/v/o slots (as the keep-packed Q2 arm builds it), and the four keep-packed
    /// weights, borrowing the store's buffers, are installed, checked, tallied and their scratch
    /// reserved before the KV cache is sized and before any graph capture. Any refusal fails the
    /// load: nothing else can serve these tensors.
    pub fn install_packed_attn(
        &mut self,
        layer: &mut Qwen3AttentionLayer,
        p: &str,
        idx: usize,
    ) -> Result<()> {
        let gpu = self.gpu;
        let shapes = super::attn_fp8_from_bf16::AttnShapes::of(self.config).projections();
        let boxes = if self.ct_on() && ct_install::attn_any(self.store, p) {
            let scheme = self.ct_scheme()?;
            ct_install::attn_boxes(
                gpu,
                self.store,
                self.config,
                p,
                shapes,
                &scheme,
                self.stream,
            )?
        } else {
            packed_install::attn_boxes(
                self.store,
                p,
                shapes,
                self.config.tp_world_size,
                self.config.adapter_max_rank > 0,
                |t| crate::port::gguf_q4::from_weight_tensor(gpu, t),
            )?
        };
        let [q, k, v, o] = self.accept(W4Role::Attn, idx, boxes, None)?;
        layer.try_set_w4_proj(q, k, v, o).with_context(|| {
            format!("attention layer {idx} ({p}): keep-packed W4 q/k/v/o install")
        })?;
        layer.prepare_w4_proj(self.gpu, self.stream)?;
        Ok(())
    }

    /// Fail the load, naming the layer, when a GDN layer's input projections arrive keep-packed:
    /// no install serves them (see [`packed_install::gdn_any_packed`]). `la` is `<layer>.linear_attn`.
    pub fn refuse_packed_gdn(&self, la: &str) -> Result<()> {
        if packed_install::gdn_any_packed(self.store, la) {
            return Err(packed_install::gdn_packed_refusal(la));
        }
        if self.ct_on() && ct_install::gdn_any(self.store, la) {
            return Err(ct_install::gdn_refusal(la));
        }
        Ok(())
    }

    /// Hook of the BF16 GDN arm, right after the layer is built with its BF16 `[Q|K|V|Z]`
    /// (`qkvz`, the loader's concat) and `out_proj_dense`. On the route both sources are
    /// quantized and freed, and the pair replaces the layer's BF16 slots.
    pub fn install_gdn(
        &mut self,
        layer: &mut Qwen3SsmLayer,
        lp: &str,
        idx: usize,
        qkvz: DenseWeight,
        out_proj: DenseWeight,
        residency: &mut DerivedResidency,
    ) -> Result<()> {
        let plan = self.plans.gdn;
        if !plan.on()
            || qkvz.weight.is_null()
            || out_proj.weight.is_null()
            || layer.w4_proj_installed()
        {
            return Ok(());
        }
        let ctx = self.ctx(lp);
        let Some(boxes) = gdn_affine4_from_bf16::install(&plan, &ctx, qkvz, out_proj)? else {
            return Ok(());
        };
        let [q, o] = self.accept(W4Role::Gdn, idx, boxes, Some(residency))?;
        // The fallible setter (lane L22a's R1): the BF16 sources are already freed, so a refusal
        // leaves the layer with no QKVZ weight and must fail the load. The bytes it returns are
        // those of BF16 slots the route already freed (0 on the layers the loader builds).
        let _released = layer
            .try_set_w4_proj(q, o)
            .with_context(|| format!("GDN layer {idx} ({lp}): W4 in_proj_qkvz/out_proj install"))?;
        // Reserve the shared scratch before the KV cache is sized and before any graph capture.
        layer.prepare_w4_proj(self.gpu, self.stream)?;
        Ok(())
    }

    /// Hook of the BF16 attention arm, right after the layer is built with BF16 q/k/v and its BF16
    /// O projection `o_dense`. On the route the four sources are quantized and freed, and the
    /// packed weights replace the layer's BF16 slots.
    pub fn install_attn(
        &mut self,
        layer: &mut Qwen3AttentionLayer,
        lp: &str,
        idx: usize,
        o_dense: Option<DenseWeight>,
        residency: &mut DerivedResidency,
    ) -> Result<()> {
        let plan = self.plans.attn;
        if !plan.on() {
            return Ok(());
        }
        let Some(o) = o_dense.filter(|o| !o.weight.is_null()) else {
            return Ok(());
        };
        let (q, k, v) = (layer.attn.q_proj, layer.attn.k_proj, layer.attn.v_proj);
        // The setter's refusals a fresh BF16 layer can meet, checked BEFORE the sources are freed:
        // a refusal after the free could not be undone.
        let bf16_missing = [q, k, v].iter().any(|w| w.weight.is_null());
        if bf16_missing || layer.k_eq_v || layer.mla.is_some() || layer.w4_proj_installed() {
            return Ok(());
        }
        let ctx = self.ctx(lp);
        let Some(boxes) = attn_affine4_from_bf16::install(&plan, &ctx, [q, k, v, o])? else {
            return Ok(());
        };
        let [q, k, v, o] = self.accept(W4Role::Attn, idx, boxes, Some(residency))?;
        // The fallible setter (lane L22b's R2): the BF16 sources are already freed, so a refusal
        // would leave the layer with no q/k/v/o weights at all and must fail the load.
        layer.try_set_w4_proj(q, k, v, o).with_context(|| {
            format!(
                "attention layer {idx} ({lp}): W4 q/k/v/o install after the BF16 sources were \
                 freed; {}=0 keeps BF16",
                Route::AttnAffine.env()
            )
        })?;
        // Reserve the shared scratch and this layer's staging region before the KV cache is sized
        // and before any graph capture.
        layer.prepare_w4_proj(self.gpu, self.stream)?;
        Ok(())
    }

    /// What each layer type installed, for `/metrics`.
    pub fn records(&self) -> Vec<RoleRecord> {
        W4Role::ALL
            .iter()
            .map(|&role| {
                let t = self.tally[slot(role)];
                RoleRecord {
                    role,
                    format: t.format,
                    layers: t.layers,
                    bytes: t.bytes,
                    per_row: self.checks.per_row(role),
                }
            })
            .collect()
    }

    /// End of the load: one summary line per installed layer type, a warning for a route that
    /// ran but found no qualifying layer, and the `/metrics` record (replacing the previous
    /// load's; cleared when nothing is 4-bit).
    pub fn finish(&self) {
        for (route, p) in self.plans.routes() {
            // At most one FFN route is on (the NVFP4 table declines under affine), so the slot
            // is that route's.
            if p.on() && self.tally[slot(p.role)].layers == 0 {
                tracing::warn!(
                    "[avarok] 4-bit route {} is ON but no layer qualified; every layer keeps its \
                     current path",
                    route.row()
                );
            }
        }
        let recs = self.records();
        for r in recs.iter().filter(|r| r.format.is_some()) {
            let t = &self.tally[slot(r.role)];
            tracing::info!("{}", summary_line(r.role, t, r.per_row));
        }
        w4_stats::publish(&recs);
    }
}

#[cfg(test)]
#[path = "w4_install_tests.rs"]
mod tests;
