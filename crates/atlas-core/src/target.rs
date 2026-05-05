// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel target descriptor for (Hardware, Model_quantization) tuples.
//!
//! Every set of Atlas kernels is hyperoptimized for a specific target.
//! This module provides the `KernelTarget` type that serves as the
//! indexing key for PTX module sets, benchmarks, and kernel dispatch.

/// Describes a (Hardware, Model, Quantization) target for kernel selection.
///
/// Each unique `KernelTarget` maps to a distinct set of PTX modules
/// that have been hyperoptimized for that specific combination.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelTarget {
    /// SM architecture identifier (e.g., "sm_121", "sm_100a").
    pub arch: &'static str,
    /// Model identifier (e.g., "qwen3-next-80b-a3b").
    pub model: &'static str,
    /// Quantization scheme (e.g., "nvfp4", "fp8", "bf16").
    pub quant: &'static str,
}

impl KernelTarget {
    /// GB10 + Qwen3-Next-80B-A3B + NVFP4.
    pub const GB10_QWEN3_NVFP4: Self = Self {
        arch: "sm_121",
        model: "qwen3-next-80b-a3b",
        quant: "nvfp4",
    };

    /// GB10 + Qwen3.5-35B-A3B + NVFP4.
    pub const GB10_QWEN35_NVFP4: Self = Self {
        arch: "sm_121",
        model: "qwen3.5-35b-a3b",
        quant: "nvfp4",
    };

    /// GB10 + Qwen3.5-122B-A10B + NVFP4.
    pub const GB10_QWEN35_122B_NVFP4: Self = Self {
        arch: "sm_121",
        model: "qwen3.5-122b-a10b",
        quant: "nvfp4",
    };

    /// Check if this target's model name contains the given substring.
    pub fn model_contains(&self, substring: &str) -> bool {
        self.model.contains(substring)
    }
}

impl std::fmt::Display for KernelTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.arch, self.model, self.quant)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_target_display() {
        let t = KernelTarget::GB10_QWEN3_NVFP4;
        assert_eq!(t.to_string(), "(sm_121, qwen3-next-80b-a3b, nvfp4)");
    }

    #[test]
    fn targets_are_comparable() {
        let a = KernelTarget::GB10_QWEN3_NVFP4;
        let b = KernelTarget {
            arch: "sm_100a",
            model: "llama-70b",
            quant: "fp8",
        };
        assert_ne!(a, b);
        assert_eq!(a, KernelTarget::GB10_QWEN3_NVFP4);
    }
}
