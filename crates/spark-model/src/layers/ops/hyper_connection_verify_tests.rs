// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Verify must preserve the decode projection shape even when elementwise
/// stages process several rows. Run with the existing real-weight HC fixture.
#[test]
#[ignore = "requires GB10 and ATLAS_HC_TEST_DATA"]
fn hc_verify_batched_stages_match_serial_bytes() {
    let f = Fixture::load();
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build the qwen3.8-flash-next/nvfp4 kernel target");
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).unwrap();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let streams = upload(g, &f.bytes("streams"));
    let scratch = g.alloc(64 * (2 * f.hc * f.h + f.rank + f.hc) * 2).unwrap();
    for site in ["attn", "mlp"] {
        let w = site_weights(g, &f, site, true);
        for rows in [2usize, 3, 4] {
            assert!(rows <= f.tokens);
            let outputs: Vec<_> = (0..2)
                .map(|_| {
                    (
                        g.alloc(rows * f.h * 2).unwrap(),
                        g.alloc(rows * f.hc * 4).unwrap(),
                    )
                })
                .collect();
            for (i, &(y, inj)) in outputs.iter().enumerate() {
                let batch = if i == 0 { rows } else { 1 };
                for t in (0..rows).step_by(batch) {
                    ops::hc_pre_gemm(
                        g,
                        streams.offset(t * f.hc * f.h * 4),
                        &w,
                        y.offset(t * f.h * 2),
                        inj.offset(t * f.hc * 4),
                        scratch,
                        batch as u32,
                        f.h as u32,
                        f.hc as u32,
                        f.eps,
                        true,
                        true,
                        true,
                        stream,
                    )
                    .unwrap();
                }
            }
            g.synchronize(stream).unwrap();
            for (label, a, b, bytes) in [
                ("hidden", outputs[0].0, outputs[1].0, rows * f.h * 2),
                ("injection", outputs[0].1, outputs[1].1, rows * f.hc * 4),
            ] {
                let mut actual = vec![0; bytes];
                let mut expected = vec![0; bytes];
                g.copy_d2h(a, &mut actual).unwrap();
                g.copy_d2h(b, &mut expected).unwrap();
                let differences = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
                assert_eq!(differences, 0, "{site} {rows} rows {label}");
                g.free(a).unwrap();
                g.free(b).unwrap();
            }
        }
    }
}
