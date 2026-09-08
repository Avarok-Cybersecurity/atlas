// SPDX-License-Identifier: AGPL-3.0-only

//! Type-safe kernel argument builder for CUDA + Metal kernel launches.
//!
//! Replaces manual `Vec<*mut c_void>` construction with a builder
//! pattern that prevents parameter type/order mismatches AND records
//! per-arg type information so the metal backend can dispatch buffer
//! args via `setBuffer:offset:atIndex:` and scalar args via
//! `setBytes:length:atIndex:` (cuda's untyped `cuLaunchKernel` cannot
//! distinguish the two; metal cannot conflate them).
//!
//! # Usage
//!
//! ```ignore
//! KernelLaunch::new(gpu, kernel)
//!     .grid([num_tokens, 1, 1])
//!     .block([256, 1, 1])
//!     .arg_ptr(input)
//!     .arg_u32(hidden_size)
//!     .arg_f32(eps)
//!     .launch(stream)?;
//! ```
//!
//! Internally, every arg is recorded with its kind (Buffer / Scalar)
//! and its native byte width. `launch()` materializes a typed
//! `KernelArg` slice and calls `GpuBackend::launch_typed`. The
//! cuda backend's default `launch_typed` impl flattens that back
//! into the legacy `void**` shape; the metal backend overrides
//! `launch_typed` to thread the type info through to the encoder.

use anyhow::Result;

use crate::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};

/// Per-arg metadata: which slot of `storage` it lives at, and how
/// many native bytes it occupies. Buffer args set `is_buffer = true`
/// and ignore `byte_len`. Scalar args set `is_buffer = false` and
/// record `byte_len = sizeof::<T>()`.
struct ArgKind {
    is_buffer: bool,
    /// Byte count for scalar args (4 for u32/i32/f32, 8 for u64, 128 for a
    /// `CUtensorMap`). Unused when `is_buffer` is true. `u16`, not `u8`: a TMA
    /// descriptor does not fit in 255 bytes' worth of headroom by much, and a
    /// silently-wrapped length here is the same class of bug as the truncation
    /// this type now guards against.
    byte_len: u16,
    /// Starting slot in `storage`. An arg is NOT one slot: a by-value struct
    /// occupies `ceil(byte_len/8)` CONSECUTIVE slots, so `launch()` can no
    /// longer index `storage` by the arg's position in `kinds`.
    slot: u32,
}

/// Builder for type-safe kernel launches across CUDA + Metal.
///
/// Accumulates grid dimensions, block dimensions, and typed kernel
/// arguments. `launch()` packages the args as `&[KernelArg]` and
/// calls `GpuBackend::launch_typed`.
pub struct KernelLaunch<'a> {
    gpu: &'a dyn GpuBackend,
    kernel: KernelHandle,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    /// Route through `GpuBackend::launch_cooperative_typed` instead of
    /// `launch_typed`. Set by `cooperative()` for kernels that `grid.sync()`.
    cooperative: bool,
    /// Backing storage: each parameter's bytes stored in a u64 slot
    /// (LE-packed for scalars; raw u64 GPU address for pointers).
    /// Pointers into this vec remain stable because we never
    /// reallocate after the initial capacity reservation.
    storage: Vec<u64>,
    /// Parallel array recording per-arg kind so `launch()` can build
    /// a typed `KernelArg` slice.
    kinds: Vec<ArgKind>,
}

impl<'a> KernelLaunch<'a> {
    pub fn new(gpu: &'a dyn GpuBackend, kernel: KernelHandle) -> Self {
        Self {
            gpu,
            kernel,
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shared_mem: 0,
            cooperative: false,
            storage: Vec::with_capacity(16),
            kinds: Vec::with_capacity(16),
        }
    }

    pub fn grid(mut self, grid: [u32; 3]) -> Self {
        self.grid = grid;
        self
    }

    pub fn block(mut self, block: [u32; 3]) -> Self {
        self.block = block;
        self
    }

    pub fn shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem = bytes;
        self
    }

    /// Launch cooperatively (`cuLaunchCooperativeKernel`): all blocks are
    /// co-resident, making in-kernel `grid.sync()` legal — required by the
    /// EXL3 trellis GEMM/GEMV kernels. The caller must size the grid from
    /// `GpuBackend::sm_count` (a grid that cannot be fully resident FAILS the
    /// launch), and a kernel wanting > 48 KB dynamic smem must have had
    /// `GpuBackend::set_kernel_max_dynamic_smem` called on it once at kernel
    /// resolution — the cooperative path does not auto-raise the cap.
    pub fn cooperative(mut self) -> Self {
        self.cooperative = true;
        self
    }

    /// Add a DevicePtr (u64) argument.
    pub fn arg_ptr(mut self, p: DevicePtr) -> Self {
        let slot = self.storage.len() as u32;
        self.storage.push(p.0);
        self.kinds.push(ArgKind {
            is_buffer: true,
            byte_len: 0,
            slot,
        });
        self
    }

    /// Add a 128-byte `CUtensorMap` by value, for a kernel parameter declared
    /// `__grid_constant__ const CUtensorMap`.
    ///
    /// TMA descriptors are the one argument on this path that is not
    /// pointer-or-scalar sized: the driver copies all 128 bytes into the
    /// parameter buffer, so the bytes must land in `ceil(128/8) = 16`
    /// CONSECUTIVE slots contributing ONE param entry. See
    /// `gpu::pack_kernel_args`, and `a_128_byte_arg_is_not_truncated`.
    pub fn arg_tensormap(mut self, map: &[u8; 128]) -> Self {
        let slot = self.storage.len() as u32;
        for c in map.chunks(8) {
            let mut w = [0u8; 8];
            w.copy_from_slice(c);
            self.storage.push(u64::from_le_bytes(w));
        }
        self.kinds.push(ArgKind {
            is_buffer: false,
            byte_len: 128,
            slot,
        });
        self
    }

    /// Add a u32 argument.
    pub fn arg_u32(mut self, v: u32) -> Self {
        let slot = self.storage.len() as u32;
        self.storage.push(v as u64);
        self.kinds.push(ArgKind {
            is_buffer: false,
            byte_len: 4,
            slot,
        });
        self
    }

    /// Add a u64 argument.
    pub fn arg_u64(mut self, v: u64) -> Self {
        let slot = self.storage.len() as u32;
        self.storage.push(v);
        self.kinds.push(ArgKind {
            is_buffer: false,
            byte_len: 8,
            slot,
        });
        self
    }

    /// Add an i32 argument.
    pub fn arg_i32(mut self, v: i32) -> Self {
        // Store as u64, preserving the i32 bits in the low 4 bytes.
        let slot = self.storage.len() as u32;
        self.storage.push(v as u32 as u64);
        self.kinds.push(ArgKind {
            is_buffer: false,
            byte_len: 4,
            slot,
        });
        self
    }

    /// Add an f32 argument.
    pub fn arg_f32(mut self, v: f32) -> Self {
        let slot = self.storage.len() as u32;
        self.storage.push(f32::to_bits(v) as u64);
        self.kinds.push(ArgKind {
            is_buffer: false,
            byte_len: 4,
            slot,
        });
        self
    }

    /// Execute the kernel launch via `GpuBackend::launch_typed`.
    ///
    /// Builds a typed `KernelArg` slice from the recorded storage +
    /// kinds. The cuda backend's default `launch_typed` flattens this
    /// back into the legacy `void**` shape; the metal backend
    /// overrides `launch_typed` to use `setBuffer:` / `setBytes:` per
    /// arg. The storage vec is not reallocated between building the
    /// args and launching, so all byte slices remain valid.
    pub fn launch(self, stream: u64) -> Result<()> {
        // Build typed args. The `&[u8]` slices borrow from `self.storage`
        // (specifically the low N bytes of each u64 slot, LE-packed).
        let mut args: Vec<KernelArg<'_>> = Vec::with_capacity(self.kinds.len());
        for kind in self.kinds.iter() {
            let slot = &self.storage[kind.slot as usize];
            if kind.is_buffer {
                args.push(KernelArg::Buffer(DevicePtr(*slot)));
            } else {
                // SAFETY: slot is a valid u64 in self.storage; we slice
                // its first `byte_len` bytes (LE) and the slice's
                // lifetime is bounded by the borrow of self.storage,
                // which lives until the end of this function.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        slot as *const u64 as *const u8,
                        kind.byte_len as usize,
                    )
                };
                args.push(KernelArg::Bytes(bytes));
            }
        }
        let r = if self.cooperative {
            self.gpu.launch_cooperative_typed(
                self.kernel,
                self.grid,
                self.block,
                self.shared_mem,
                stream,
                &args,
            )
        } else {
            self.gpu.launch_typed(
                self.kernel,
                self.grid,
                self.block,
                self.shared_mem,
                stream,
                &args,
            )
        };
        // ATLAS_DEBUG_SYNC_KERNELS (PCND, default-off): synchronize after
        // each launch so an async CUDA fault surfaces AT the culprit launch
        // (with grid/block) instead of at a later, unrelated sync point.
        // Diagnostic only — leave unset in production (one stream sync per
        // launch is a large slowdown). With RUST_BACKTRACE=1 the propagated
        // error pinpoints the calling op.
        if r.is_ok() && self.gpu.debug_sync_kernels() {
            self.gpu.synchronize(stream).map_err(|e| {
                let bt = std::backtrace::Backtrace::force_capture();
                anyhow::anyhow!(
                    "ATLAS_DEBUG_SYNC_KERNELS: async GPU fault immediately after kernel launch \
                     grid={:?} block={:?} shared_mem={}: {e}\nLAUNCH BACKTRACE:\n{bt}",
                    self.grid,
                    self.block,
                    self.shared_mem
                )
            })?;
        }
        r
    }
}

// `ATLAS_DEBUG_SYNC_KERNELS` is now resolved once when the backend is built
// and read through `GpuBackend::debug_sync_kernels` — the launch path is far
// too hot for a per-call getenv, and a static was the wrong way to avoid one.

/// Convenience: divide and round up.
pub fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::mock::MockGpuBackend;

    #[test]
    fn test_kernel_launch_builder() {
        let gpu = MockGpuBackend::new();
        let kernel = gpu.kernel("test", "test_kernel").unwrap();

        let result = KernelLaunch::new(&gpu, kernel)
            .grid([4, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(DevicePtr(0x1000))
            .arg_u32(42)
            .arg_f32(1.5)
            .launch(0);

        assert!(result.is_ok());
        assert_eq!(gpu.launch_count(), 1);
    }

    /// A 128-byte by-value struct (the `CUtensorMap` case) must survive the
    /// launch path intact and occupy 16 CONSECUTIVE slots with ONE param entry.
    ///
    /// The packing this guards used to be `b.len().min(8)`: it truncated to the
    /// first 8 bytes and launched anyway. That is unobservable from the caller
    /// and produces a kernel silently reading garbage, so the test asserts the
    /// bytes round-trip, not merely that the call succeeded.
    #[test]
    fn a_128_byte_arg_is_not_truncated() {
        use crate::gpu::{KernelArg, pack_kernel_args};
        let map: Vec<u8> = (0..128u16).map(|i| (i * 7 % 251) as u8).collect();
        let args = [
            KernelArg::Buffer(DevicePtr(0xDEAD_BEEF)),
            KernelArg::Bytes(&map),
            KernelArg::Bytes(&42u32.to_le_bytes()),
        ];
        let (storage, starts) = pack_kernel_args(&args);

        assert_eq!(
            starts.len(),
            3,
            "one param entry per argument, not per slot"
        );
        assert_eq!(starts, vec![0, 1, 17], "the map occupies slots 1..=16");
        assert_eq!(storage.len(), 18);
        assert_eq!(storage[0], 0xDEAD_BEEF);

        // Every one of the 128 bytes must be readable, contiguously, from the
        // map's starting slot — that is exactly what the kernel will do.
        let round_trip: Vec<u8> = storage[starts[1]..starts[1] + 16]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        assert_eq!(
            round_trip, map,
            "128-byte struct arg was corrupted or truncated"
        );
        assert_eq!(
            storage[starts[2]] as u32, 42,
            "the arg after it is still intact"
        );
    }

    /// End-to-end through the BUILDER: a tensormap between two ordinary args
    /// must not disturb either, and must present as one param entry.
    ///
    /// `launch()` used to index `storage` by an arg's POSITION in `kinds`, which
    /// is only correct while every arg is exactly one slot. A 16-slot arg makes
    /// that indexing silently read the wrong slot for every argument AFTER it.
    #[test]
    fn a_tensormap_arg_does_not_shift_the_args_around_it() {
        let gpu = MockGpuBackend::new();
        let kernel = gpu.kernel("test", "tma_kernel").unwrap();
        let map = [0xABu8; 128];

        let b = KernelLaunch::new(&gpu, kernel)
            .arg_ptr(DevicePtr(0x1000))
            .arg_tensormap(&map)
            .arg_u32(7);

        assert_eq!(b.kinds.len(), 3, "one param entry per arg, not per slot");
        assert_eq!(b.storage.len(), 1 + 16 + 1);
        assert_eq!(b.kinds[0].slot, 0);
        assert_eq!(b.kinds[1].slot, 1);
        assert_eq!(b.kinds[1].byte_len, 128);
        assert_eq!(
            b.kinds[2].slot, 17,
            "the arg after the map must not be shifted"
        );
        assert_eq!(b.storage[b.kinds[2].slot as usize] as u32, 7);
        assert!(b.launch(0).is_ok());
    }

    /// A zero-length byte arg still needs a slot to point at.
    #[test]
    fn an_empty_byte_arg_still_gets_one_slot() {
        use crate::gpu::{KernelArg, pack_kernel_args};
        let (storage, starts) =
            pack_kernel_args(&[KernelArg::Bytes(&[]), KernelArg::Bytes(&[9u8])]);
        assert_eq!(starts, vec![0, 1]);
        assert_eq!(storage.len(), 2);
        assert_eq!(storage[1], 9);
    }

    /// `.cooperative()` must change the ROUTE, not just set a bit: the launch
    /// has to reach `launch_cooperative_typed` with grid/block/smem AND the
    /// typed args intact. A grid.sync() kernel that silently went down the
    /// eager path would deadlock on real hardware, which no arg assertion
    /// alone can catch — hence asserting `cooperative` on the recorded launch.
    #[test]
    fn cooperative_routes_to_the_cooperative_path_with_args_intact() {
        use crate::gpu::mock::MockArg;
        let gpu = MockGpuBackend::new();
        let kernel = gpu.kernel("exl3_gemm", "exl3_gemm_k4_s2").unwrap();

        KernelLaunch::new(&gpu, kernel)
            .grid([48, 1, 1])
            .block([512, 1, 1])
            .shared_mem(90 * 1024)
            .cooperative()
            .arg_ptr(DevicePtr(0x2000))
            .arg_i32(2560)
            .launch(7)
            .unwrap();

        assert_eq!(gpu.cooperative_launch_count(), 1);
        let launches = gpu.launches_snapshot();
        let l = &launches[0];
        assert!(l.cooperative);
        assert_eq!(l.grid, [48, 1, 1]);
        assert_eq!(l.block, [512, 1, 1]);
        assert_eq!(l.shared_mem, 90 * 1024, "dynamic smem must survive the route");
        assert_eq!(l.stream, 7);
        assert_eq!(l.args[0], MockArg::Buffer(DevicePtr(0x2000)));
        assert_eq!(l.args[1], MockArg::Bytes(2560i32.to_le_bytes().to_vec()));
    }

    /// Without `.cooperative()` the builder must keep taking the eager path —
    /// the flag defaults off, and an accidental cooperative launch of an
    /// ordinary kernel is its own bug class (grid-size launch failures).
    #[test]
    fn launches_stay_eager_unless_cooperative_is_requested() {
        let gpu = MockGpuBackend::new();
        let kernel = gpu.kernel("test", "plain_kernel").unwrap();
        KernelLaunch::new(&gpu, kernel)
            .grid([4, 1, 1])
            .block([256, 1, 1])
            .arg_u32(1)
            .launch(0)
            .unwrap();
        assert_eq!(gpu.cooperative_launch_count(), 0);
        assert!(!gpu.launches_snapshot()[0].cooperative);
    }

    /// The one-time smem opt-in must reach the backend with the kernel handle
    /// and byte count it was given (the real impl writes a sticky per-function
    /// driver attribute; the mock records the call).
    #[test]
    fn max_dynamic_smem_opt_in_is_recorded_per_kernel() {
        let gpu = MockGpuBackend::new();
        let kernel = gpu.kernel("exl3_gemm", "exl3_gemm_k4_s2").unwrap();
        gpu.set_kernel_max_dynamic_smem(kernel, 90 * 1024).unwrap();
        assert_eq!(gpu.max_dynamic_smem_calls(), vec![(kernel.0, 90 * 1024)]);
    }

    #[test]
    fn test_div_ceil() {
        assert_eq!(div_ceil(10, 3), 4);
        assert_eq!(div_ceil(9, 3), 3);
        assert_eq!(div_ceil(1, 256), 1);
        assert_eq!(div_ceil(256, 256), 1);
        assert_eq!(div_ceil(257, 256), 2);
    }
}
