// SPDX-License-Identifier: AGPL-3.0-only

//! Stable, config-derived model fingerprint for the SHARED paging peer.
//!
//! The paging peer (atlas-cache-peer) owns ONE residency map shared across
//! every fleet client, keyed purely by the u64 the client sends — so the
//! per-model namespace folded into each key is the ONLY thing preventing two
//! models from silently serving each other's recurrent state. That makes the
//! namespace a **durable on-disk contract**: the peer's NVMe swap file
//! outlives client rebuilds and toolchains, so the same model config must
//! derive the same u64 forever.
//!
//! ## Why FNV-1a/64, vendored
//!
//! `std::hash::DefaultHasher` documents its algorithm as unspecified and "not
//! to be relied upon over releases" — a toolchain bump may silently rotate
//! every persisted key (total cache miss) or collide with a stale namespace
//! (silent wrong-state corruption). FNV-1a/64 is fully specified (offset
//! basis `0xcbf29ce484222325`, prime `0x100000001b3`), consumes a byte stream
//! (endianness-free), and is vendored here (~10 lines) so no crate version
//! bump can ever change the value. The test file pins the primitive to the
//! published FNV reference vectors and the full fingerprint of known configs
//! to frozen literals. Collision quality is irrelevant at this input size (a
//! handful of fleet model configs, not an adversarial keyspace).
//!
//! ## Encoding contract
//!
//! The hash input is an injective canonical encoding: tagged records, u64s as
//! `[tag][8-byte LE]`, strings length-prefixed `[tag][4-byte LE len][bytes]`
//! (so `("ab","c")` never collides with `("a","bc")`). Field set and order
//! are FROZEN behind [`FP_VERSION`]; any change to either is a deliberate
//! fleet-wide cache flush and must bump the version. `kv_layer_dims` is
//! loader-populated and order-sensitive — derive() must run after loader
//! post-processing (it does: the only call sites are in
//! `TransformerModel::new`).
//!
//! The runtime KV-cache dtype (`--kv-cache-dtype`) is deliberately EXCLUDED:
//! SSM h/conv state is FP32 by construction regardless of KV dtype, so two
//! serves of one model with different KV dtypes produce byte-identical SSM
//! blobs and SHOULD share the warm cache. A future KV paging tier must fold
//! its own dtype/block_size mix-in at its own call site.
//!
//! Note: `wire()`'s splitmix fold is bijective per namespace, but two
//! DIFFERENT namespaces can map two (key, ns) pairs to one wire key with
//! ~2^-64 probability — acceptable, and NOT to be "strengthened" into a wire
//! -format change.

use std::num::NonZeroU64;

use anyhow::{Result, anyhow, bail};
use atlas_core::config::ModelConfig;

// SSOT: the durable-key hash primitives live in atlas-tier (pure, dep-free, a
// common ancestor of this crate and spark-storage's kv_paging). They were
// transcribed in three places; the constants are now defined exactly once.
pub(crate) use atlas_tier::hash::{FNV_OFFSET, fnv1a_64, mix64};

/// Bump = deliberate fleet-wide cache-key rotation (document it).
// v2 (2026-07-10): added hidden_size / num_attention_heads / intermediate_size /
// moe_intermediate_size / num_experts_per_tok (tags 0x16-0x1a). v1 omitted them, so
// two distinct models differing only in residual/FFN width collided. Bumping the
// version is a DELIBERATE, one-time fleet cache-key rotation (greenfield: no shim).
pub(crate) const FP_VERSION: u64 = 2;

fn put_u64(buf: &mut Vec<u8>, tag: u8, v: u64) {
    buf.push(tag);
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_str(buf: &mut Vec<u8>, tag: u8, s: &str) {
    buf.push(tag);
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// Stable identity of "the bytes this model's SSM tier produces", derived
/// from the loaded [`ModelConfig`] geometry + quantization identity +
/// `blob_bytes`. Non-zero by construction (a 0 namespace — the old silent
/// passthrough — is unrepresentable downstream).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ModelFingerprint(NonZeroU64);

impl ModelFingerprint {
    /// Derive the fingerprint and log it at INFO so operators can see and pin
    /// it. `ATLAS_MODEL_ID` (optional) is an extra salt for the one case
    /// geometry cannot distinguish: a fine-tune with byte-identical config —
    /// unset (the common case) it contributes an empty record and never
    /// rotates keys.
    pub(crate) fn derive(cfg: &ModelConfig, blob_bytes: usize) -> Result<Self> {
        let model_id = std::env::var("ATLAS_MODEL_ID").unwrap_or_default();
        let fp = Self::derive_with_id(cfg, blob_bytes, &model_id)?;
        tracing::info!(
            "SSM tier model fingerprint = {:#018x} (model_type={}, blob_bytes={blob_bytes}, \
             ATLAS_MODEL_ID={model_id:?}); pin with ATLAS_SSM_SWAP_NS / ATLAS_SSM_DECODE_NS",
            fp.get(),
            cfg.model_type,
        );
        // The fingerprint is GEOMETRY-ONLY. It cannot distinguish two checkpoints
        // with byte-identical config — a fine-tune, an RL variant, a continued
        // pre-train of one base. Those derive the same namespace and would share
        // one store's cache. We cannot fail fast (config carries no weight
        // identity), so warn loudly exactly when it matters: a SHARED store backs
        // the tier and the operator gave neither a salt nor an explicit ns.
        //
        // NOT just a peer: ATLAS_SSM_SWAP=1 is a LOCAL swap FILE, which two
        // processes pointed at one swap dir share exactly as dangerously. Saying
        // "peer" here sent an operator hunting for an RDMA peer that was never
        // configured (found by running the serve path, not by any unit test).
        let shared_store = std::env::var("ATLAS_SSM_SWAP").ok().as_deref() == Some("1")
            || std::env::var("ATLAS_SSM_DECODE_TIER").ok().as_deref() == Some("peer");
        let overridden = std::env::var_os("ATLAS_SSM_SWAP_NS").is_some()
            || std::env::var_os("ATLAS_SSM_DECODE_NS").is_some();
        if shared_store && model_id.is_empty() && !overridden {
            tracing::warn!(
                "a SHARED SSM cache store is selected (ATLAS_SSM_SWAP=1 local swap file, or \
                 ATLAS_SSM_DECODE_TIER=peer) but ATLAS_MODEL_ID is unset: the fingerprint is \
                 derived from config GEOMETRY ONLY, so two checkpoints with identical config \
                 (fine-tunes, RL variants, continued pre-train of one base) will SHARE cache \
                 keys and silently cross-serve recurrent state. Set ATLAS_MODEL_ID to a stable \
                 per-checkpoint string (or set ATLAS_SSM_SWAP_NS / ATLAS_SSM_DECODE_NS \
                 explicitly) when co-locating such models on one store."
            );
        }
        Ok(fp)
    }

    /// KV-paging fingerprint (`ATLAS_KV_PAGING`): the SAME canonical
    /// per-model encoding as the SSM tier, derived with the KV CONVENTION
    /// `blob_bytes = 0` (tag 0x40 = 0 marks the KV instance; SSM instances
    /// put their real, non-zero blob size there — so the two never share a
    /// value, and attention-only models with no SSM tier still get an fp).
    /// The KV tier folds its full block geometry + dtype + a per-client salt
    /// DOWNSTREAM at its own call site (`spark_storage::kv_paging::ns`),
    /// exactly as the module doc above prescribes — do NOT add KV fields
    /// here (that would rotate SSM keys).
    pub(crate) fn derive_kv(cfg: &ModelConfig) -> Result<Self> {
        let model_id = std::env::var("ATLAS_MODEL_ID").unwrap_or_default();
        let fp = Self::derive_with_id(cfg, 0, &model_id)?;
        tracing::info!(
            "KV paging model fingerprint = {:#018x} (model_type={}, \
             ATLAS_MODEL_ID={model_id:?}); folded into the ATLAS_KV_PAGING namespace",
            fp.get(),
            cfg.model_type,
        );
        Ok(fp)
    }

    /// Pure derivation (no env, no logging) — the canonical encoding.
    /// FROZEN: field set + order changes require an FP_VERSION bump.
    pub(crate) fn derive_with_id(
        cfg: &ModelConfig,
        blob_bytes: usize,
        model_id: &str,
    ) -> Result<Self> {
        // Defensive PCND bail: never legitimate after parse_config. Without a
        // real fingerprint a shared peer would need an explicit override.
        if cfg.model_type.is_empty() && cfg.num_hidden_layers == 0 {
            bail!(
                "cannot derive a model fingerprint: empty model_type and zero geometry; \
                 fix the model config or set ATLAS_SSM_SWAP_NS / ATLAS_SSM_DECODE_NS \
                 to explicit non-zero u64 namespaces"
            );
        }
        let (qm, qa, qf) = cfg.quantization_config.as_ref().map_or(("", "", ""), |q| {
            (
                q.quant_method.as_str(),
                q.quant_algo.as_str(),
                q.format.as_str(),
            )
        });
        let mut buf = Vec::with_capacity(256);
        put_u64(&mut buf, 0x00, FP_VERSION);
        put_str(&mut buf, 0x01, &cfg.model_type);
        // Quant identity: an NVFP4 build and a bf16 build of one checkpoint
        // share geometry but produce different recurrent state values.
        put_str(&mut buf, 0x02, qm);
        put_str(&mut buf, 0x03, qa);
        put_str(&mut buf, 0x04, qf);
        put_str(&mut buf, 0x05, model_id);
        for (tag, v) in [
            (0x10, cfg.num_hidden_layers),
            (0x11, cfg.num_ssm_layers()),
            (0x12, cfg.num_attention_layers()),
            (0x13, cfg.head_dim),
            (0x14, cfg.num_key_value_heads),
            (0x15, cfg.num_experts),
            // FP_VERSION 2: residual-stream and FFN widths. Omitting these was a
            // BLOCKING defect — two distinct models differing ONLY in
            // `hidden_size` / `num_attention_heads` hashed identically and would
            // have silently cross-served recurrent state on a shared paging peer
            // (the exact failure this fingerprint exists to prevent). `blob_bytes`
            // does NOT capture them: it is `num_ssm_layers * (h + conv)` bytes,
            // which is SSM-state-only and independent of the residual width.
            (0x16, cfg.hidden_size),
            (0x17, cfg.num_attention_heads),
            (0x18, cfg.intermediate_size),
            (0x19, cfg.moe_intermediate_size),
            (0x1a, cfg.num_experts_per_tok),
            // SSM state geometry — included alongside blob_bytes because the
            // blob size is a lossy product of these.
            (0x20, cfg.linear_num_key_heads),
            (0x21, cfg.linear_key_head_dim),
            (0x22, cfg.linear_num_value_heads),
            (0x23, cfg.linear_value_head_dim),
            (0x24, cfg.linear_conv_kernel_dim),
            (0x25, cfg.mamba_num_heads),
            (0x26, cfg.mamba_head_dim),
            (0x27, cfg.ssm_state_size),
            (0x28, cfg.n_groups),
            // SSM h/conv state element size: FP32 today (the ×4 hardcoded in
            // ssm_h_state_bytes/ssm_conv_state_bytes). Fingerprinted so a
            // future non-FP32 SSM state rotates keys instead of colliding.
            (0x30, 4),
            (0x40, blob_bytes),
        ] {
            put_u64(&mut buf, tag, v as u64);
        }
        // Heterogeneous-attention per-layer (kv_heads, head_dim) overrides
        // (Gemma-4). Loader-populated; canonical order = layer order.
        put_u64(&mut buf, 0x50, cfg.kv_layer_dims.len() as u64);
        for &(kvh, hd) in &cfg.kv_layer_dims {
            put_u64(&mut buf, 0x51, kvh as u64);
            put_u64(&mut buf, 0x52, hd as u64);
        }
        let h = fnv1a_64(&buf);
        // Zero-avoidance (p = 2^-64): keep NonZeroU64 total + deterministic.
        Ok(Self(
            NonZeroU64::new(h).unwrap_or(NonZeroU64::new(FNV_OFFSET).unwrap()),
        ))
    }

    pub(crate) fn get(self) -> u64 {
        self.0.get()
    }

    pub(crate) fn nonzero(self) -> NonZeroU64 {
        self.0
    }
}

/// Marconi swap namespace: `ATLAS_SSM_SWAP_NS` override (strict — junk or 0
/// is a startup ERROR, never a silent fallthrough) else the fingerprint.
pub(crate) fn resolve_swap_ns(fp: ModelFingerprint) -> Result<NonZeroU64> {
    resolve_ns_from(
        std::env::var("ATLAS_SSM_SWAP_NS").ok().as_deref(),
        "ATLAS_SSM_SWAP_NS",
        fp.nonzero(),
    )
}

/// Per-process decode CLIENT salt — resolved ONCE (`OnceLock`) so every decode
/// store built in this process folds the SAME salt: `cold_key` determinism
/// within a process is an invariant the spill FIFO / epoch guard rest on.
///
/// `ATLAS_SSM_DECODE_CLIENT_ID` pins it (strict parse; 0 is PERMITTED — the
/// salt is key material folded through `mix64`, not a sentinel); unset ⇒
/// fresh OS entropy per process. Entropy failure is a hard startup error,
/// never a silent zero-salt (a degraded shared salt would quietly reintroduce
/// cross-client key sharing).
fn decode_client_salt() -> Result<u64> {
    use std::sync::OnceLock;
    static SALT: OnceLock<u64> = OnceLock::new();
    if let Some(&s) = SALT.get() {
        return Ok(s);
    }
    let salt = match std::env::var("ATLAS_SSM_DECODE_CLIENT_ID").ok() {
        Some(raw) => parse_u64_strict("ATLAS_SSM_DECODE_CLIENT_ID", &raw)?,
        None => atlas_tier::entropy::random_u64()?,
    };
    // A racing thread may have won `get_or_init` meanwhile; both then return
    // the ONE stored value — the process-wide salt is still unique.
    Ok(*SALT.get_or_init(|| salt))
}

/// Env-free core of the decode namespace:
/// `mix64(mix64(fingerprint, DECODE_DOMAIN), client_salt)`.
///
/// Every layer is load-bearing: the DOMAIN separator keeps decode keys off the
/// same model's Marconi keys (both tiers share ONE peer residency whenever
/// blob_bytes match); the fingerprint keeps them off OTHER models' decode
/// spills; the CLIENT salt keeps them off other same-model PROCESSES' spills —
/// decode keys are SLOT COORDINATES (`cold_key(seq, logical)`), byte-identical
/// across same-model clients, so without the salt two clients on one peer
/// cross-serve and cross-delete each other's rollback state.
///
/// Zero-avoidance (each layer p = 2^-64) falls back exactly ONE layer: never
/// `fp` bare (that IS the Marconi swap namespace), never `DECODE_DOMAIN` alone
/// (model-blind). Total over all inputs.
pub(crate) fn derive_decode_ns_salted(fp: u64, salt: u64) -> NonZeroU64 {
    let base = NonZeroU64::new(mix64(fp, atlas_kernels::DECODE_DOMAIN)).unwrap_or_else(|| {
        NonZeroU64::new(atlas_kernels::DECODE_DOMAIN).expect("DECODE_DOMAIN is a non-zero constant")
    });
    NonZeroU64::new(mix64(base.get(), salt)).unwrap_or(base)
}

/// Decode namespace: `ATLAS_SSM_DECODE_NS` override (strict, wins UNSALTED —
/// the escape hatch fully determines the ns) else the per-process salted
/// derivation [`derive_decode_ns_salted`]. Decode blobs are process-ephemeral
/// (pid-named local arenas, all-Absent manager init, no store enumeration —
/// never recovered across runs), so the per-process rotation loses nothing;
/// the generated salt is INFO-logged so an operator can pin/reproduce it.
pub(crate) fn resolve_decode_ns(fp: ModelFingerprint) -> Result<NonZeroU64> {
    if let Ok(raw) = std::env::var("ATLAS_SSM_DECODE_NS") {
        // Mirror the ATLAS_MODEL_ID warn in `ModelFingerprint::derive`: the
        // override is the documented escape hatch, and exactly as dangerous.
        tracing::warn!(
            "ATLAS_SSM_DECODE_NS={raw:?} bypasses the per-process decode client salt: decode \
             keys are SLOT COORDINATES (not content hashes), so two processes sharing this \
             value on one peer WILL cross-serve and cross-delete each other's rollback state \
             (silent corruption + spurious 'cold MISS on live target'). Ensure a DISTINCT \
             value per process, or unset it and pin ATLAS_SSM_DECODE_CLIENT_ID instead."
        );
        if std::env::var_os("ATLAS_SSM_DECODE_CLIENT_ID").is_some() {
            tracing::warn!(
                "ATLAS_SSM_DECODE_CLIENT_ID is IGNORED while ATLAS_SSM_DECODE_NS is set \
                 (the explicit namespace fully determines the wire keys)"
            );
        }
        return parse_ns("ATLAS_SSM_DECODE_NS", &raw);
    }
    let salt = decode_client_salt()?;
    let ns = derive_decode_ns_salted(fp.get(), salt);
    tracing::info!(
        "SSM decode ns = {ns:#018x} (fp {:#018x} ⊕ DECODE_DOMAIN ⊕ client_salt \
         {salt:#018x}); decode keys are CLIENT-PRIVATE on a shared peer; pin \
         ATLAS_SSM_DECODE_CLIENT_ID={salt:#x} to reproduce this namespace",
        fp.get(),
    );
    Ok(ns)
}

/// Env-free core (unit-testable without process-global setenv races).
pub(crate) fn resolve_ns_from(
    override_raw: Option<&str>,
    var: &str,
    derived: NonZeroU64,
) -> Result<NonZeroU64> {
    match override_raw {
        Some(raw) => parse_ns(var, raw),
        None => Ok(derived),
    }
}

/// Strict u64 parser: decimal or `0x`-hex. Junk is a hard startup error, never
/// a silent fallthrough (PCND). Unlike [`parse_ns`], 0 is PERMITTED — callers
/// needing `NonZeroU64` layer that check on top.
pub(crate) fn parse_u64_strict(var: &str, raw: &str) -> Result<u64> {
    let s = raw.trim();
    let parsed = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => s.parse::<u64>(),
    };
    parsed.map_err(|e| anyhow!("{var}={raw:?} is not a valid u64 (decimal or 0x-hex): {e}"))
}

/// Strict override parser: decimal or `0x`-hex u64. Unparseable values are a
/// hard error (PCND — the old code silently `.ok()`-swallowed a mistyped
/// override into the model-blind default). 0 is a hard error: the ns=0
/// passthrough is removed, a shared peer must always be namespaced.
pub(crate) fn parse_ns(var: &str, raw: &str) -> Result<NonZeroU64> {
    let v = parse_u64_strict(var, raw)?;
    NonZeroU64::new(v).ok_or_else(|| {
        anyhow!(
            "{var}=0 is invalid: the ns=0 passthrough is removed (it silently \
             cross-served state between models on a shared peer); unset {var} \
             to use the derived model fingerprint (logged at INFO on startup)"
        )
    })
}

#[cfg(test)]
#[path = "fingerprint_tests.rs"]
mod tests;
