// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `kda_layer_microtest.rs` to keep it under the 500-LoC cap.
//! Test-only harness code; no serving path runs any of it.

#![allow(unused_imports)]

use crate::*;
use anyhow::{Context, Result, bail};
use half::bf16;
use kda_layer_cpu::*;
use serde_json::Value;
use spark_model::layers::glm5next_kda::binding::{
    self, AttnBlockKind, KdaDtype, KdaTensorSource, RawTensor,
};
use spark_model::layers::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace, KdaSeqState,
};
use spark_model::layers::glm5next_kda_ref as kref;
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::collections::BTreeMap;

/// A layer packet as a [`KdaTensorSource`]: layer-relative names over the raw safetensors bytes.
pub(crate) struct Packet {
    pub(crate) raw: Vec<u8>,
    pub(crate) base: usize,
    pub(crate) hdr: BTreeMap<String, (KdaDtype, Vec<usize>, usize, usize)>,
}

impl Packet {
    pub(crate) fn open(path: &str) -> Result<Self> {
        let raw = std::fs::read(path).with_context(|| format!("reading {path}"))?;
        let hn = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let j: Value = serde_json::from_slice(&raw[8..8 + hn])?;
        let mut hdr = BTreeMap::new();
        for (k, m) in j.as_object().unwrap() {
            if k == "__metadata__" {
                continue;
            }
            let dt = KdaDtype::parse(m["dtype"].as_str().unwrap())
                .with_context(|| format!("{k}: a KDA block must be BF16 / F32 only"))?;
            let shape: Vec<usize> = m["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let a = m["data_offsets"][0].as_u64().unwrap() as usize;
            let b = m["data_offsets"][1].as_u64().unwrap() as usize;
            hdr.insert(k.clone(), (dt, shape, a, b));
        }
        Ok(Self {
            raw,
            base: 8 + hn,
            hdr,
        })
    }
    pub(crate) fn f32s(&self, name: &str) -> Vec<f32> {
        let (dt, _, a, b) = &self.hdr[name];
        let by = &self.raw[self.base + a..self.base + b];
        match dt {
            KdaDtype::Bf16 => by
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            KdaDtype::F32 => by
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        }
    }
}

impl KdaTensorSource for Packet {
    fn get(&self, name: &str) -> Option<RawTensor<'_>> {
        let (dt, shape, a, b) = self.hdr.get(name)?;
        Some(RawTensor {
            dtype: *dt,
            shape: shape.clone(),
            bytes: &self.raw[self.base + a..self.base + b],
        })
    }
    fn names(&self) -> Vec<String> {
        self.hdr.keys().cloned().collect()
    }
}

/// Host mirror of a bound block, for the CPU oracle only.
pub(crate) fn host_weights(p: &Packet) -> Wts {
    let mut conv = p.f32s("self_attn.q_conv1d.weight");
    conv.extend(p.f32s("self_attn.k_conv1d.weight"));
    conv.extend(p.f32s("self_attn.v_conv1d.weight"));
    Wts {
        q: p.f32s("self_attn.q_proj.weight"),
        k: p.f32s("self_attn.k_proj.weight"),
        v: p.f32s("self_attn.v_proj.weight"),
        conv,
        f_a: p.f32s("self_attn.f_a_proj.weight"),
        f_b: p.f32s("self_attn.f_b_proj.weight"),
        dt_bias: p.f32s("self_attn.dt_bias"),
        a_log: p.f32s("self_attn.A_log"),
        b: p.f32s("self_attn.b_proj.weight"),
        g_a: p.f32s("self_attn.g_a_proj.weight"),
        g_b: p.f32s("self_attn.g_b_proj.weight"),
        o_norm: p.f32s("self_attn.o_norm.weight"),
        o: p.f32s("self_attn.o_proj.weight"),
    }
}

pub(crate) struct Gpu<'a> {
    pub(crate) g: &'a dyn GpuBackend,
    pub(crate) layer: Glm5NextKdaLayer,
    pub(crate) ws: Glm5NextKdaWorkspace,
    pub(crate) dm: Dims,
}

impl Gpu<'_> {
    /// Drive one forward and read every stage back. `pad_fill` primes the padded q/k/v tails —
    /// zero in production, poison for the Slice-5 guard regression.
    pub(crate) fn run(
        &self,
        hidden: &[f32],
        t: usize,
        decode: bool,
        conv_state4: &[f32],
        state: &[f32],
        pad_fill: f32,
    ) -> Result<(Stages, Vec<f32>, Vec<f32>)> {
        let g = self.g;
        let (qkv, cd, h, hid) = (self.dm.qkv(), self.dm.conv_dim(), self.dm.h, self.dm.hid);
        let ws = &self.ws;
        let st = KdaSeqState {
            conv: up_f32(g, conv_state4)?,
            recurrent: up_f32(g, state)?,
        };
        let dh = up_bf16(g, hidden)?;

        if decode {
            self.layer.decode(g, dh, &st, ws, 0)?;
        } else {
            self.layer
                .prefill_with_pad_fill(g, dh, t, &st, ws, pad_fill, 0)?;
        }
        g.synchronize(0)?;

        let t_pad = if decode {
            1
        } else {
            t.div_ceil(self.layer.cfg.chunk) * self.layer.cfg.chunk
        };
        let conv_out = down_bf16(g, ws.conv_out, t * cd)?;
        let pick = |off: usize| -> Vec<f32> {
            (0..t)
                .flat_map(|tt| conv_out[tt * cd + off..tt * cd + off + qkv].to_vec())
                .collect()
        };
        let core_full = down_f32(g, ws.core, t_pad * qkv)?;
        let out = Stages {
            qkv_proj: down_bf16(g, ws.qkv_proj, t * cd)?,
            q: pick(0),
            k: pick(qkv),
            v: pick(2 * qkv),
            gate: down_f32(g, ws.gate, t_pad * qkv)?[..t * qkv].to_vec(),
            beta: down_f32(g, ws.beta, t_pad * h)?[..t * h].to_vec(),
            core: core_full[..t * qkv].to_vec(),
            state: down_f32(g, st.recurrent, h * self.dm.d * self.dm.d)?,
            out_gate: down_bf16(g, ws.out_gate, t * qkv)?,
            o_norm: down_bf16(g, ws.o_norm_out, t * qkv)?,
            final_out: down_bf16(g, ws.final_out, t * hid)?,
            conv_state: down_f32(g, st.conv, cd * self.dm.ks)?,
            ref_layer_delta: None,
        };
        let (cs, rs) = (out.conv_state.clone(), out.state.clone());
        Ok((out, cs, rs))
    }
}

pub(crate) fn dwt(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DenseWeight> {
    Ok(DenseWeight {
        weight: up_bf16(gpu, v)?,
    })
}

/// Four regimes plus the pad-guard regression, for one bound layer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_suite(
    gpu: &dyn GpuBackend,
    layer: Glm5NextKdaLayer,
    ws: &Glm5NextKdaWorkspace,
    dm: Dims,
    cfg: Glm5NextKdaConfig,
    w: &Wts,
    golden: Option<&Value>,
    tag: &str,
) -> Result<bool> {
    let _ = ws;
    let layer_ws = Glm5NextKdaWorkspace::new(gpu, &cfg, 8)?;
    let g = Gpu {
        g: gpu,
        layer,
        ws: layer_ws,
        dm,
    };
    let cd = dm.conv_dim();
    let sz_state = dm.h * dm.d * dm.d;
    let mut ok = true;

    // Fixture draw order must match the generator exactly. The RAW fp32 hidden is what floor A
    // is fed: the generator's fp32 arm never rounds its input.
    let draw = |t: usize| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rng = Lcg(0x5EED_1A70);
        let hidden = rng.vec(t.max(8) * dm.hid)[..t * dm.hid].to_vec();
        let conv3 = rng.scaled(cd * (dm.ks - 1), 0.5);
        let rec = rng.scaled(sz_state, 0.1);
        (round_bf16(&hidden), hidden, conv3, rec)
    };
    // HF's `kernel-1` slots -> Atlas's `kernel`; slot 0 is shifted out before the conv.
    let widen = |c3: &[f32]| -> Vec<f32> {
        let mut s = vec![0.0f32; cd * dm.ks];
        for ch in 0..cd {
            for i in 0..dm.ks - 1 {
                s[ch * dm.ks + 1 + i] = c3[ch * (dm.ks - 1) + i];
            }
        }
        s
    };
    let tail3 = |s4: &[f32]| -> Vec<f32> {
        (0..cd)
            .flat_map(|ch| (1..dm.ks).map(move |i| (ch, i)))
            .map(|(ch, i)| s4[ch * dm.ks + i])
            .collect()
    };

    for (rname, t, decode) in [
        ("decode1", 1usize, true),
        ("prefill4", 4, false),
        ("prefill7", 7, false),
    ] {
        let (hidden, hidden_f32, c3, rec) = draw(t);
        let cs4 = if decode {
            widen(&c3)
        } else {
            vec![0.0f32; cd * dm.ks]
        };
        let st0 = if decode {
            rec.clone()
        } else {
            vec![0.0f32; sz_state]
        };

        let (gs, _, _) = g.run(&hidden, t, decode, &cs4, &st0, 0.0)?;
        let (mut cpu_cs, mut cpu_st) = (cs4.clone(), st0.clone());
        let cs = cpu_layer(
            w,
            dm,
            cfg,
            &hidden,
            t,
            &mut cpu_cs,
            &mut cpu_st,
            decode,
            CHUNK,
            false,
        );
        let (mut acs, mut ast) = (cs4.clone(), st0.clone());
        let ca = cpu_layer(
            w,
            dm,
            cfg,
            &hidden_f32,
            t,
            &mut acs,
            &mut ast,
            decode,
            CHUNK,
            true,
        );

        ok &= report(
            &format!("{tag} {rname}"),
            t,
            dm,
            &gs,
            &cs,
            Some(&ca),
            golden,
            rname,
            &tail3,
        )?;
    }

    // ── ragged prefill -> decode: the regime that carries state across formulations ─────
    {
        let (hidden, hidden_f32, _, _) = draw(7);
        let mut gcs = vec![0.0f32; cd * dm.ks];
        let mut gst = vec![0.0f32; sz_state];
        let (gp, cs_out, st_out) = g.run(&hidden, 7, false, &gcs, &gst, 0.0)?;
        gcs = cs_out;
        gst = st_out;
        let (mut ccs, mut cst) = (vec![0.0f32; cd * dm.ks], vec![0.0f32; sz_state]);
        let cp = cpu_layer(
            w, dm, cfg, &hidden, 7, &mut ccs, &mut cst, false, CHUNK, false,
        );
        let (mut acs, mut ast) = (vec![0.0f32; cd * dm.ks], vec![0.0f32; sz_state]);
        let ap = cpu_layer(
            w,
            dm,
            cfg,
            &hidden_f32,
            7,
            &mut acs,
            &mut ast,
            false,
            CHUNK,
            true,
        );

        let mut h2_f32 = Lcg(0xD3C0_DE01).vec(8 * dm.hid);
        h2_f32.truncate(dm.hid);
        let h2 = round_bf16(&h2_f32);
        let (gd, _, _) = g.run(&h2, 1, true, &gcs, &gst, 0.0)?;
        let cd_st = cpu_layer(w, dm, cfg, &h2, 1, &mut ccs, &mut cst, true, CHUNK, false);
        let ad = cpu_layer(
            w, dm, cfg, &h2_f32, 1, &mut acs, &mut ast, true, CHUNK, true,
        );

        ok &= report(
            &format!("{tag} prefill7_decode1 (prefill leg)"),
            7,
            dm,
            &gp,
            &cp,
            Some(&ap),
            golden,
            "prefill7_decode1",
            &tail3,
        )?;
        ok &= report_decode_leg(
            &format!("{tag} prefill7_decode1 (decode leg)"),
            dm,
            &gd,
            &cd_st,
            Some(&ad),
            golden,
            &tail3,
        )?;

        // ── PAD-CORRUPTION REGRESSION (Slice 5) ────────────────────────────────
        let zc = vec![0.0f32; cd * dm.ks];
        let zs = vec![0.0f32; sz_state];
        let (pg, pcs, pst) = g.run(&hidden, 7, false, &zc, &zs, 7.5)?;
        let d_out = maxabs(&pg.final_out, &gp.final_out);
        let d_state = maxabs(&pst, &gst);
        let d_conv = maxabs(&pcs, &gcs);
        let d_core = maxabs(&pg.core, &gp.core);
        let (pd, _, _) = g.run(&h2, 1, true, &pcs, &pst, 0.0)?;
        let d_dec = maxabs(&pd.final_out, &gd.final_out);
        let clean =
            d_out == 0.0 && d_state == 0.0 && d_conv == 0.0 && d_core == 0.0 && d_dec == 0.0;
        println!(
            "\n  {tag}: PAD-CORRUPTION REGRESSION — T=7 real, {} padded positions filled with 7.5",
            CHUNK - 7
        );
        println!(
            "    prefill out {d_out:.3e} · core {d_core:.3e} · CARRIED STATE {d_state:.3e} · \
             conv state {d_conv:.3e} · NEXT decode {d_dec:.3e}  [{}]",
            if clean {
                "ok, kernels self-guard past T"
            } else {
                "FAIL"
            }
        );
        ok &= clean;
    }

    Ok(ok)
}
