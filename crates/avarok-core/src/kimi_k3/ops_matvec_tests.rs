// SPDX-License-Identifier: AGPL-3.0-only

use super::matvec;

#[test]
fn interleaved_rows_preserve_scalar_reduction_bits_and_tails() {
    let values = [
        0.0, -0.0, 1.0e10, -1.0e10, 1.0e-10, -1.0e-10, 0.33333334, -7.25,
    ];
    for out in [0, 1, 2, 3, 4, 5, 7, 8, 9, 33] {
        for inn in [0, 1, 3, 31, 32, 129, 1024] {
            let w: Vec<f32> = (0..out * inn)
                .map(|i| values[(i * 7 + i / 3) % values.len()])
                .collect();
            let x: Vec<f32> = (0..inn)
                .map(|i| values[(i * 3 + 1) % values.len()])
                .collect();
            let got = matvec(&w, &x, out, inn);
            for o in 0..out {
                let mut expected = 0.0f32;
                for i in 0..inn {
                    expected += w[o * inn + i] * x[i];
                }
                assert_eq!(
                    got[o].to_bits(),
                    expected.to_bits(),
                    "out={out} inn={inn} row={o}"
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "matvec weight")]
fn matvec_rejects_invalid_weight_shape() {
    matvec(&[1.0; 7], &[1.0; 2], 4, 2);
}
