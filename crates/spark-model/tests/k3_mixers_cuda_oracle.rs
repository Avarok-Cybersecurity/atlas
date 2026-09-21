// SPDX-License-Identifier: AGPL-3.0-only

//! K3 TP8 mixer geometry on actual CUDA, against independent core CPU kernels.
//! Uses synthetic projected inputs, not weights or a full model. The cached
//! prefix/decode handoff is exercised; this does not certify a fused prefill.
//! Both wrappers return FP32: tolerance is 2e-6 absolute + 2e-5 relative,
//! well below BF16 rounding, allowing FMA/exp/sqrt implementation differences
//! across sequential 128/192-element reductions. No per-run tolerance tuning.
//! Build with AVAROK_TARGET_MODEL=kimi-k3 AVAROK_TARGET_QUANT=mxfp4 and matching
//! AVAROK_TARGET_HW. On an idle GPU, keep those variables and run:
//! K3_ORACLE_GPU_ORDINAL=0 timeout 180s cargo test -p spark-model \
//! --test k3_mixers_cuda_oracle -- --ignored --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use anyhow::{Context, Result, ensure};
use avarok_core::kimi_k3::{
    KdaConfig, KdaState, MlaConfig, MlaKv, kda_decode_token, kda_from, mla_decode_token, mla_from,
};
use spark_model::kimi_k3::kda_cuda::{
    K3KdaDecodeKernels, KdaDeviceState, launch_k3_kda_decode_token_on_device,
};
use spark_model::kimi_k3::mla_cuda::{
    K3MlaDecodeKernels, MlaDeviceKv, launch_k3_mla_decode_token_on_device,
};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

fn values(len: usize, salt: usize, amplitude: f32) -> Vec<f32> {
    let mut state = 0x8a73_aed9_bc12_3405u64 ^ salt as u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state >> 32) % 257) as f32 - 128.0) * (amplitude / 128.0)
        })
        .collect()
}

fn close(label: &str, actual: &[f32], expected: &[f32]) -> Result<()> {
    ensure!(actual.len() == expected.len(), "{label}: size mismatch");
    let mut max_abs = 0.0f32;
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        max_abs = max_abs.max((a - b).abs());
        ensure!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-6 + 2e-5 * b.abs(),
            "{label}[{i}]: CUDA={a} CPU={b}"
        );
    }
    println!(
        "{label}: {} FP32 values, max_abs_error={max_abs}",
        actual.len()
    );
    Ok(())
}

fn exact(label: &str, actual: &[f32], expected: &[f32]) -> Result<()> {
    ensure!(
        actual == expected,
        "{label}: exact state/continuation mismatch"
    );
    Ok(())
}

fn download(gpu: &dyn GpuBackend, ptr: DevicePtr, len: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0; len * 4];
    gpu.copy_d2h(ptr, &mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn upload(gpu: &dyn GpuBackend, ptr: DevicePtr, values: &[f32]) -> Result<()> {
    gpu.copy_h2d(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        ptr,
    )
}

struct OwnedKda<'a> {
    gpu: &'a dyn GpuBackend,
    inner: Option<KdaDeviceState>,
}

impl<'a> OwnedKda<'a> {
    fn new(gpu: &'a dyn GpuBackend, state: &KdaState) -> Result<Self> {
        Ok(Self {
            gpu,
            inner: Some(KdaDeviceState::alloc_and_upload(gpu, state)?),
        })
    }
    fn state(&self) -> &KdaDeviceState {
        self.inner.as_ref().unwrap()
    }
}

impl Drop for OwnedKda<'_> {
    fn drop(&mut self) {
        if let Some(state) = self.inner.take() {
            let _ = state.free(self.gpu);
        }
    }
}

fn kda_oracle(gpu: &dyn GpuBackend, stream: u64, cfg: &KdaConfig) -> Result<()> {
    let kernels = K3KdaDecodeKernels::resolve(gpu)?;
    let mut cpu = KdaState::new(cfg);
    cpu.conv = values(cfg.conv_elems(), 41, 0.3);
    cpu.recurrent = values(cfg.recurrent_elems(), 42, 0.08);
    let device = OwnedKda::new(gpu, &cpu)?;
    let mut restored: Option<OwnedKda<'_>> = None;
    let weights = values(cfg.conv_elems(), 43, 0.4);
    for token in 0..8 {
        let x = values(cfg.conv_dim(), 101 + token, 0.7);
        let gate: Vec<_> = values(cfg.qkv_dim(), 201 + token, 0.3)
            .iter()
            .map(|g| g - 0.4)
            .collect();
        let beta = values(cfg.heads, 301 + token, 1.5);
        let expected = kda_decode_token(&x, &weights, &gate, &beta, cfg, &mut cpu);
        if token == 0 {
            let wrong = kda_decode_token(&x, &weights, &gate, &beta, cfg, &mut KdaState::new(cfg));
            ensure!(
                close("negative control: cleared KDA history", &wrong, &expected).is_err(),
                "KDA fixture is insensitive to missing history"
            );
        }
        let actual = launch_k3_kda_decode_token_on_device(
            gpu,
            &kernels,
            &x,
            &weights,
            &gate,
            &beta,
            cfg,
            device.state(),
            stream,
        )?;
        close(&format!("KDA output token {token}"), &actual, &expected)?;
        let mut snapshot = KdaState::new(cfg);
        device.state().download(gpu, &mut snapshot)?;
        exact("KDA conv history", &snapshot.conv, &cpu.conv)?;
        close("KDA recurrent state", &snapshot.recurrent, &cpu.recurrent)?;
        if let Some(ref second) = restored {
            let continuation = launch_k3_kda_decode_token_on_device(
                gpu,
                &kernels,
                &x,
                &weights,
                &gate,
                &beta,
                cfg,
                second.state(),
                stream,
            )?;
            exact(
                "KDA restored vs uninterrupted output",
                &continuation,
                &actual,
            )?;
            let mut second_snapshot = KdaState::new(cfg);
            second.state().download(gpu, &mut second_snapshot)?;
            exact("KDA restored conv", &second_snapshot.conv, &snapshot.conv)?;
            exact(
                "KDA restored recurrent",
                &second_snapshot.recurrent,
                &snapshot.recurrent,
            )?;
        }
        if token == 2 {
            restored = Some(OwnedKda::new(gpu, &snapshot)?);
        }
    }
    println!(
        "PASS KDA TP8 heads={} dim={} tokens=8 nonzero history + restored continuation + cleared-history negative control",
        cfg.heads, cfg.head_dim
    );
    Ok(())
}

struct OwnedMla<'a> {
    gpu: &'a dyn GpuBackend,
    inner: MlaDeviceKv,
}

impl<'a> OwnedMla<'a> {
    fn new(gpu: &'a dyn GpuBackend, host: &MlaKv, cfg: &MlaConfig) -> Result<Self> {
        let mut owned = Self {
            gpu,
            inner: MlaDeviceKv::alloc(
                gpu,
                40,
                cfg.heads * cfg.qk_head_dim(),
                cfg.heads * cfg.v_head_dim,
            )?,
        };
        upload(gpu, owned.inner.k, &host.k)?;
        upload(gpu, owned.inner.v, &host.v)?;
        owned.inner.seq_len = host.seq_len;
        Ok(owned)
    }
    fn snapshot(&self, cfg: &MlaConfig) -> Result<MlaKv> {
        Ok(MlaKv {
            seq_len: self.inner.seq_len,
            k: download(
                self.gpu,
                self.inner.k,
                self.inner.seq_len * cfg.heads * cfg.qk_head_dim(),
            )?,
            v: download(
                self.gpu,
                self.inner.v,
                self.inner.seq_len * cfg.heads * cfg.v_head_dim,
            )?,
        })
    }
}

impl Drop for OwnedMla<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.free(self.inner.k);
        let _ = self.gpu.free(self.inner.v);
    }
}

fn mla_oracle(gpu: &dyn GpuBackend, stream: u64, cfg: &MlaConfig, theta: f32) -> Result<()> {
    ensure!(
        cfg.mla_use_nope && cfg.mla_use_output_gate,
        "expected official gated NoPE config"
    );
    let kernels = K3MlaDecodeKernels::resolve(gpu)?;
    let mut cpu = MlaKv::default();
    let dq = cfg.heads * cfg.qk_head_dim();
    let dv = cfg.heads * cfg.v_head_dim;
    // Materialized 31-token NoPE prefix, then append across positions 31/32/33.
    // It is not a fused-prefill invocation: this tests the state handoff.
    for pos in 0..31 {
        cpu.append(&values(dq, 400 + pos, 0.7), &values(dv, 500 + pos, 0.9));
    }
    let mut device = OwnedMla::new(gpu, &cpu, cfg)?;
    let mut restored: Option<OwnedMla<'_>> = None;
    for pos in 31..37 {
        let q = values(dq, 600 + pos, 0.8);
        let k = values(dq, 700 + pos, 0.7);
        let v = values(dv, 800 + pos, 0.9);
        let gate = values(dv, 900 + pos, 2.5);
        let before = cpu.clone();
        let expected = mla_decode_token(
            &mut q.clone(),
            &mut k.clone(),
            &v,
            &gate,
            &mut cpu,
            cfg,
            pos,
            theta,
        );
        if pos == 31 {
            let wrong_gate: Vec<_> = gate.iter().map(|g| g + 3.0).collect();
            let wrong = mla_decode_token(
                &mut q.clone(),
                &mut k.clone(),
                &v,
                &wrong_gate,
                &mut before.clone(),
                cfg,
                pos,
                theta,
            );
            ensure!(
                close("negative control: wrong MLA gate", &wrong, &expected).is_err(),
                "MLA fixture is insensitive to gate corruption"
            );
            let wrong = mla_decode_token(
                &mut q.clone(),
                &mut k.clone(),
                &v,
                &gate,
                &mut MlaKv::default(),
                cfg,
                pos,
                theta,
            );
            ensure!(
                close("negative control: dropped MLA prefix", &wrong, &expected).is_err(),
                "MLA fixture is insensitive to missing prefix"
            );
        }
        let actual = launch_k3_mla_decode_token_on_device(
            gpu,
            &kernels,
            &q,
            &k,
            &v,
            &gate,
            &mut device.inner,
            cfg,
            pos,
            theta,
            stream,
        )?;
        close(&format!("MLA output position {pos}"), &actual, &expected)?;
        let snapshot = device.snapshot(cfg)?;
        ensure!(snapshot.seq_len == pos + 1, "MLA sequence length");
        exact("MLA appended NoPE keys", &snapshot.k, &cpu.k)?;
        exact("MLA appended values", &snapshot.v, &cpu.v)?;
        if let Some(ref mut second) = restored {
            let continuation = launch_k3_mla_decode_token_on_device(
                gpu,
                &kernels,
                &q,
                &k,
                &v,
                &gate,
                &mut second.inner,
                cfg,
                pos,
                theta,
                stream,
            )?;
            exact(
                "MLA restored vs uninterrupted output",
                &continuation,
                &actual,
            )?;
            let second_snapshot = second.snapshot(cfg)?;
            exact("MLA restored keys", &second_snapshot.k, &snapshot.k)?;
            exact("MLA restored values", &second_snapshot.v, &snapshot.v)?;
        }
        if pos == 32 {
            restored = Some(OwnedMla::new(gpu, &snapshot, cfg)?);
        }
    }
    println!(
        "PASS MLA TP8 heads={} qk_dim={} v_dim={} prefix=31 decode=6 + restored continuation + gate/prefix negative controls",
        cfg.heads,
        cfg.qk_head_dim(),
        cfg.v_head_dim
    );
    Ok(())
}

#[test]
#[ignore = "requires an explicitly selected idle CUDA device and compiled K3 MXFP4 kernels"]
fn k3_tp8_mixers_match_core_cpu() -> Result<()> {
    let ordinal = std::env::var("K3_ORACLE_GPU_ORDINAL")
        .context("set K3_ORACLE_GPU_ORDINAL explicitly")?
        .parse::<usize>()?;
    let target = avarok_kernels::ptx_for_exact_target("kimi-k3", "mxfp4")
        .context("compile exact K3 MXFP4 target")?;
    println!("K3 mixer oracle PTX architecture: {}", target.ptx_arch);
    let gpu = AvarokCudaBackend::new(ordinal, &target.modules)?;
    let stream = gpu.create_stream()?;
    let mut config = avarok_core::config::parse_config(include_str!(
        "../../../docs/k3/fixtures/moonshotai-Kimi-K3-config.json"
    ))?;
    ensure!(
        config.num_attention_heads > 0
            && config.linear_num_key_heads > 0
            && config.num_attention_heads % 8 == 0
            && config.linear_num_key_heads % 8 == 0,
        "official head counts must divide TP8"
    );
    config.num_attention_heads /= 8;
    config.linear_num_key_heads /= 8;
    config.tp_world_size = 8;
    config.tp_rank = 0;
    kda_oracle(&gpu, stream, &kda_from(&config))?;
    mla_oracle(&gpu, stream, &mla_from(&config), config.rope_theta as f32)
}
