// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub(super) fn grammar_argmax(
    gpu: &dyn GpuBackend,
    logits: DevicePtr,
    vocab: usize,
    bitmask: &[i32],
) -> Result<u32> {
    let mut bytes = vec![0; vocab * 2];
    gpu.copy_d2h(logits, &mut bytes)?;
    allowed_argmax(&bytes, bitmask)
}

fn allowed_argmax(bytes: &[u8], bitmask: &[i32]) -> Result<u32> {
    let mut best: Option<(u32, f32)> = None;
    for (token, value) in bytes.chunks_exact(2).enumerate() {
        if bitmask
            .get(token / 32)
            .is_none_or(|word| word & (1i32 << (token % 32)) == 0)
        {
            continue;
        }
        let value = f32::from_bits(u32::from(u16::from_le_bytes([value[0], value[1]])) << 16);
        if value.is_finite() && best.is_none_or(|(_, previous)| value > previous) {
            best = Some((token as u32, value));
        }
    }
    best.map(|(token, _)| token)
        .ok_or_else(|| anyhow::anyhow!("Qwen MTP grammar has no allowed finite draft logit"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn logits(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn grammar_draft_reads_bf16_and_excludes_higher_forbidden_logits() {
        let gpu = MockGpuBackend::new();
        let values = logits(&[100.0, 2.0, 3.0, 3.0]);
        let ptr = gpu.alloc(values.len()).unwrap();
        gpu.copy_h2d(&values, ptr).unwrap();
        assert_eq!(grammar_argmax(&gpu, ptr, 4, &[0b1110]).unwrap(), 2);
    }

    #[test]
    fn grammar_draft_handles_signed_mask_words_and_missing_words() {
        let mut values = vec![100.0; 34];
        values[31] = 2.0;
        assert_eq!(allowed_argmax(&logits(&values), &[i32::MIN]).unwrap(), 31);
    }

    #[test]
    fn grammar_draft_refuses_empty_or_nonfinite_allowed_set() {
        assert!(allowed_argmax(&logits(&[1.0]), &[0]).is_err());
        assert!(allowed_argmax(&logits(&[f32::NAN, f32::NEG_INFINITY]), &[3]).is_err());
    }
}
