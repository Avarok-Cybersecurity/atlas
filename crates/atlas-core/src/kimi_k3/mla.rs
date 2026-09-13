// SPDX-License-Identifier: AGPL-3.0-only

//! Gated NoPE MLA CPU reference.
//!
//! Do not reuse `qwen3_attention` blindly: K3 sets `mla_use_nope=true` while
//! still allocating `qk_rope_head_dim` slots, and `mla_use_output_gate=true`
//! applies a full-rank `g_proj` sigmoid gate on the attention output.
//!
//! RoPE dims stay in the head (prod 128 nope + 64 rope = 192). NoPE means
//! those slots are **not rotated**, not that they are dropped.
//!
//! CUDA decode is `kernels/gb10/kimi-k3/bf16/mla_decode.cu` (`k3_mla_*`).
//! BoundLayer FullAttention stays on this CPU path unless `K3_CUDA_MLA=1`.

#![allow(clippy::needless_range_loop)]

use super::cache::MlaKv;
use super::situ::sigmoid;

#[derive(Clone, Copy, Debug)]
pub struct MlaConfig {
    pub heads: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub mla_use_nope: bool,
    pub mla_use_output_gate: bool,
}

impl MlaConfig {
    pub fn production() -> Self {
        Self {
            heads: 96,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            mla_use_nope: true,
            mla_use_output_gate: true,
        }
    }

    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// `inference-optimization/Kimi-K3-0.40B` text_config MLA dims.
    pub fn twin_0_40b() -> Self {
        Self {
            heads: 8,
            qk_nope_head_dim: 64,
            qk_rope_head_dim: 32,
            v_head_dim: 64,
            q_lora_rank: 256,
            kv_lora_rank: 128,
            mla_use_nope: true,
            mla_use_output_gate: true,
        }
    }
}

/// FullAttention BoundLayer uses CUDA `mla_decode` only when `K3_CUDA_MLA=1`.
/// Unset / `0` keeps this CPU mixer so aviation greedy cannot regress
/// until spark1 C1 is proven. Opposite polarity from `K3_CUDA_KDA`.
pub fn cuda_mla_enabled() -> bool {
    matches!(std::env::var("K3_CUDA_MLA").as_deref(), Ok("1"))
}

/// Optional RoPE on the rope **slice** of a packed `[nope | rope]` head.
/// NoPE leaves `q`/`k` unchanged.
pub fn maybe_rope(x: &mut [f32], nope: usize, rope: usize, pos: usize, theta: f32, use_nope: bool) {
    if use_nope || rope == 0 {
        return;
    }
    let dim = nope + rope;
    for head in x.chunks_exact_mut(dim) {
        let r = &mut head[nope..];
        for i in 0..rope / 2 {
            let freq = (pos as f32) / theta.powf(2.0 * i as f32 / rope as f32);
            let (s, c) = freq.sin_cos();
            let a = r[i];
            let b = r[i + rope / 2];
            r[i] = a * c - b * s;
            r[i + rope / 2] = a * s + b * c;
        }
    }
}

/// Causal scaled-dot-product attention. `q/k`: `[T, H, dq]`, `v`: `[T, H, dv]`.
pub fn sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    heads: usize,
    dq: usize,
    dv: usize,
) -> Vec<f32> {
    let scale = 1.0 / (dq as f32).sqrt();
    let mut out = vec![0.0f32; t * heads * dv];
    for h in 0..heads {
        for qi in 0..t {
            let qrow = &q[(qi * heads + h) * dq..(qi * heads + h) * dq + dq];
            let mut scores = vec![0.0f32; qi + 1];
            let mut m = f32::NEG_INFINITY;
            for kj in 0..=qi {
                let krow = &k[(kj * heads + h) * dq..(kj * heads + h) * dq + dq];
                let s: f32 = qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * scale;
                scores[kj] = s;
                if s > m {
                    m = s;
                }
            }
            let mut z = 0.0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                z += *s;
            }
            let orow = &mut out[(qi * heads + h) * dv..(qi * heads + h) * dv + dv];
            for kj in 0..=qi {
                let a = scores[kj] / z;
                let vrow = &v[(kj * heads + h) * dv..(kj * heads + h) * dv + dv];
                for d in 0..dv {
                    orow[d] += a * vrow[d];
                }
            }
        }
    }
    out
}

/// Decode SDPA: one query `[H, dq]` against cached K/V of length `t`.
pub fn sdpa_one(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    heads: usize,
    dq: usize,
    dv: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), heads * dq);
    assert_eq!(k.len(), t * heads * dq);
    assert_eq!(v.len(), t * heads * dv);
    let scale = 1.0 / (dq as f32).sqrt();
    let mut out = vec![0.0f32; heads * dv];
    for h in 0..heads {
        let qrow = &q[h * dq..(h + 1) * dq];
        let mut scores = vec![0.0f32; t];
        let mut m = f32::NEG_INFINITY;
        for kj in 0..t {
            let krow = &k[(kj * heads + h) * dq..(kj * heads + h) * dq + dq];
            let s: f32 = qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * scale;
            scores[kj] = s;
            if s > m {
                m = s;
            }
        }
        let mut z = 0.0f32;
        for s in &mut scores {
            *s = (*s - m).exp();
            z += *s;
        }
        let orow = &mut out[h * dv..(h + 1) * dv];
        for kj in 0..t {
            let a = scores[kj] / z;
            let vrow = &v[(kj * heads + h) * dv..(kj * heads + h) * dv + dv];
            for d in 0..dv {
                orow[d] += a * vrow[d];
            }
        }
    }
    out
}

/// Apply `sigmoid(g) ⊙ attn` when the output gate is on; otherwise identity.
pub fn apply_output_gate(attn: &[f32], g: &[f32], enabled: bool) -> Vec<f32> {
    if !enabled {
        return attn.to_vec();
    }
    assert_eq!(attn.len(), g.len());
    attn.iter().zip(g).map(|(a, gg)| a * sigmoid(*gg)).collect()
}

/// Gated NoPE attend: optionally skip RoPE, SDPA, optional output gate.
#[allow(clippy::too_many_arguments)]
pub fn gated_mla_attend(
    q: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    g: &[f32],
    t: usize,
    cfg: &MlaConfig,
    pos0: usize,
    theta: f32,
) -> Vec<f32> {
    for p in 0..t {
        let dim = cfg.qk_head_dim();
        let qh = &mut q[p * cfg.heads * dim..(p + 1) * cfg.heads * dim];
        let kh = &mut k[p * cfg.heads * dim..(p + 1) * cfg.heads * dim];
        maybe_rope(
            qh,
            cfg.qk_nope_head_dim,
            cfg.qk_rope_head_dim,
            pos0 + p,
            theta,
            cfg.mla_use_nope,
        );
        maybe_rope(
            kh,
            cfg.qk_nope_head_dim,
            cfg.qk_rope_head_dim,
            pos0 + p,
            theta,
            cfg.mla_use_nope,
        );
    }
    let attn = sdpa(q, k, v, t, cfg.heads, cfg.qk_head_dim(), cfg.v_head_dim);
    apply_output_gate(&attn, g, cfg.mla_use_output_gate)
}

/// One decode token: optional RoPE, append K/V, SDPA, optional output gate.
/// Projections stay in `mla_mixer`. CUDA `k3_mla_*` matches this order.
pub fn mla_decode_token(
    q: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    g: &[f32],
    kv: &mut MlaKv,
    cfg: &MlaConfig,
    pos: usize,
    theta: f32,
) -> Vec<f32> {
    maybe_rope(
        q,
        cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim,
        pos,
        theta,
        cfg.mla_use_nope,
    );
    maybe_rope(
        k,
        cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim,
        pos,
        theta,
        cfg.mla_use_nope,
    );
    kv.append(k, v);
    let attn = sdpa_one(
        q,
        &kv.k,
        &kv.v,
        kv.seq_len,
        cfg.heads,
        cfg.qk_head_dim(),
        cfg.v_head_dim,
    );
    apply_output_gate(&attn, g, cfg.mla_use_output_gate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg(nope: bool, gate: bool) -> MlaConfig {
        MlaConfig {
            heads: 1,
            qk_nope_head_dim: 2,
            qk_rope_head_dim: 2,
            v_head_dim: 2,
            q_lora_rank: 4,
            kv_lora_rank: 4,
            mla_use_nope: nope,
            mla_use_output_gate: gate,
        }
    }

    #[test]
    fn production_keeps_rope_slots_but_nope() {
        let p = MlaConfig::production();
        assert!(p.mla_use_nope);
        assert!(p.mla_use_output_gate);
        assert_eq!(p.qk_head_dim(), 192);
        assert_eq!(p.v_head_dim, 128);
    }

    #[test]
    fn nope_does_not_rotate() {
        let mut x = vec![1.0, 0.0, 1.0, 0.0];
        let orig = x.clone();
        maybe_rope(&mut x, 2, 2, 3, 10000.0, true);
        assert_eq!(x, orig);
        maybe_rope(&mut x, 2, 2, 3, 10000.0, false);
        assert_ne!(x, orig, "RoPE on must move the rope slice");
    }

    #[test]
    fn output_gate_mutates() {
        let attn = vec![1.0, 2.0];
        let g = vec![0.0, 10.0];
        let off = apply_output_gate(&attn, &g, false);
        let on = apply_output_gate(&attn, &g, true);
        assert_eq!(off, attn);
        assert!((on[0] - 0.5).abs() < 1e-6);
        assert!(on[1] > 1.9);
        assert_ne!(on, off);
    }

    #[test]
    fn gated_nope_path_runs() {
        let cfg = tiny_cfg(true, true);
        let mut q = vec![1.0, 0.0, 0.0, 1.0];
        let mut k = q.clone();
        let v = vec![0.5, -0.5];
        let g = vec![0.0, 0.0];
        let out = gated_mla_attend(&mut q, &mut k, &v, &g, 1, &cfg, 0, 10000.0);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn twin_0_40b_geometry() {
        let t = MlaConfig::twin_0_40b();
        assert_eq!(t.heads, 8);
        assert_eq!(t.qk_head_dim(), 96);
        assert_eq!(t.v_head_dim, 64);
        assert!(t.mla_use_nope);
        assert!(t.mla_use_output_gate);
    }

    #[test]
    fn decode_token_matches_attend_t1() {
        let cfg = tiny_cfg(true, true);
        let mut q = vec![1.0, 0.0, 0.0, 1.0];
        let mut k = q.clone();
        let v = vec![0.5, -0.5];
        let g = vec![0.0, 0.0];
        let mut q2 = q.clone();
        let mut k2 = k.clone();
        let attend = gated_mla_attend(&mut q2, &mut k2, &v, &g, 1, &cfg, 0, 10000.0);
        let mut kv = MlaKv::default();
        let dec = mla_decode_token(&mut q, &mut k, &v, &g, &mut kv, &cfg, 0, 10000.0);
        assert_eq!(attend, dec);
        assert_eq!(kv.seq_len, 1);
    }

    #[test]
    fn cuda_mla_env_default_off() {
        if std::env::var_os("K3_CUDA_MLA").is_some() {
            return;
        }
        assert!(
            !cuda_mla_enabled(),
            "FullAttention default is CPU MLA; K3_CUDA_MLA=1 is opt-in"
        );
    }

    #[test]
    fn cuda_mla_env_opt_in() {
        const THIS: &str = "kimi_k3::mla::tests::cuda_mla_env_opt_in";
        const MARKER: &str = "K3_CUDA_MLA_CHILD";
        if std::env::var_os(MARKER).is_some() {
            assert!(cuda_mla_enabled(), "K3_CUDA_MLA=1 must enable CUDA MLA");
            return;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", THIS])
            .env(MARKER, "1")
            .env("K3_CUDA_MLA", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "K3_CUDA_MLA child failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
