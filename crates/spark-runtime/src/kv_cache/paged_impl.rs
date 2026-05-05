// SPDX-License-Identifier: AGPL-3.0-only
//
// `PagedKvCache` impl block. Split from `kv_cache.rs` so the parent file
// keeps the public types small enough to read at a glance. The struct
// definition lives in the parent.

use anyhow::{Result, bail};

use super::{KvCacheConfig, KvCacheDtype, LayerPool, PagedKvCache};
use crate::gpu::{DevicePtr, GpuBackend};

impl PagedKvCache {
    /// Allocate the KV cache pool on the GPU.
    pub fn new(config: KvCacheConfig, num_blocks: usize, gpu: &dyn GpuBackend) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        let mut total_bytes: usize = 0;
        for i in 0..config.num_layers {
            let layer_block_bytes = config.block_bytes_for_layer(i);
            let pool_bytes = num_blocks * layer_block_bytes;
            let k_pool = gpu.alloc(pool_bytes)?;
            let v_pool = gpu.alloc(pool_bytes)?;
            total_bytes += pool_bytes * 2;
            layers.push(LayerPool {
                k_pool,
                v_pool,
                block_stride: layer_block_bytes,
                dtype: config.dtype_for_layer(i),
            });
        }

        let free_blocks: Vec<u32> = (0..num_blocks as u32).rev().collect();
        let block_ref_counts = vec![0u32; num_blocks];

        let has_mixed = !config.layer_dtypes.is_empty()
            && config.layer_dtypes.iter().any(|d| *d != config.dtype);
        if has_mixed {
            let hp_count = config
                .layer_dtypes
                .iter()
                .filter(|d| **d != config.dtype)
                .count();
            tracing::info!(
                "KV cache: {} blocks × {} layers ({} high-precision) = {:.1} GB total (mixed dtype)",
                num_blocks,
                config.num_layers,
                hp_count,
                total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        } else {
            tracing::info!(
                "KV cache: {} blocks × {} layers × {} bytes/block = {:.1} GB total",
                num_blocks,
                config.num_layers,
                config.block_bytes_kv(),
                (num_blocks * config.num_layers * config.block_bytes_kv()) as f64
                    / (1024.0 * 1024.0 * 1024.0),
            );
        }

        Ok(Self {
            layers,
            num_blocks,
            free_blocks,
            block_ref_counts,
            config,
        })
    }

    /// Allocate a free block. Returns block index.
    pub fn alloc_block(&mut self) -> Result<u32> {
        let idx = self
            .free_blocks
            .pop()
            .ok_or_else(|| anyhow::anyhow!("KV cache exhausted: no free blocks"))?;
        self.block_ref_counts[idx as usize] = 1;
        Ok(idx)
    }

    /// Zero all KV data in a block across all layers.
    /// Prevents stale KV data from previous sequences from leaking into
    /// new sequences via paged attention reads beyond the current seq_len.
    pub fn zero_block(
        &self,
        block_idx: u32,
        gpu: &dyn crate::gpu::GpuBackend,
        stream: u64,
    ) -> anyhow::Result<()> {
        for layer in &self.layers {
            let k_offset = block_idx as usize * layer.block_stride;
            let v_offset = block_idx as usize * layer.block_stride;
            gpu.memset_async(layer.k_pool.offset(k_offset), 0, layer.block_stride, stream)?;
            gpu.memset_async(layer.v_pool.offset(v_offset), 0, layer.block_stride, stream)?;
        }
        Ok(())
    }

    /// Try to allocate a free block without failing. Returns None if exhausted.
    pub fn try_alloc_block(&mut self) -> Option<u32> {
        let idx = self.free_blocks.pop()?;
        self.block_ref_counts[idx as usize] = 1;
        Some(idx)
    }

    /// Increment reference count on a block (for prefix cache sharing).
    pub fn inc_ref(&mut self, block_idx: u32) {
        debug_assert!((block_idx as usize) < self.num_blocks);
        self.block_ref_counts[block_idx as usize] += 1;
    }

    /// Decrement reference count. Returns true if block was freed (count hit 0).
    pub fn dec_ref(&mut self, block_idx: u32) -> bool {
        let idx = block_idx as usize;
        debug_assert!(idx < self.num_blocks);
        debug_assert!(
            self.block_ref_counts[idx] > 0,
            "dec_ref on block with 0 refs"
        );
        self.block_ref_counts[idx] -= 1;
        if self.block_ref_counts[idx] == 0 {
            self.free_blocks.push(block_idx);
            true
        } else {
            false
        }
    }

    /// Free a previously allocated block (decrements ref, frees if count hits 0).
    pub fn free_block(&mut self, block_idx: u32) {
        self.dec_ref(block_idx);
    }

    /// Free all blocks in a block table.
    pub fn free_blocks(&mut self, block_table: &[u32]) {
        for &idx in block_table {
            self.free_block(idx);
        }
    }

    /// Return a block to the free pool directly, bypassing ref counting.
    /// Used by eviction: the radix tree already removed its reference.
    pub fn return_evicted_block(&mut self, block_idx: u32) {
        debug_assert!((block_idx as usize) < self.num_blocks);
        self.block_ref_counts[block_idx as usize] = 0;
        self.free_blocks.push(block_idx);
    }

    /// Current reference count for a block.
    pub fn ref_count(&self, block_idx: u32) -> u32 {
        self.block_ref_counts[block_idx as usize]
    }

    /// Number of free blocks.
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get K cache pointer for a layer and block.
    pub fn k_cache_ptr(&self, layer_idx: usize, block_idx: u32) -> DevicePtr {
        let layer = &self.layers[layer_idx];
        layer.k_pool.offset(block_idx as usize * layer.block_stride)
    }

    /// Get V cache pointer for a layer and block.
    pub fn v_cache_ptr(&self, layer_idx: usize, block_idx: u32) -> DevicePtr {
        let layer = &self.layers[layer_idx];
        layer.v_pool.offset(block_idx as usize * layer.block_stride)
    }

    /// Get the full K cache pool pointer for a layer (for paged decode kernel).
    pub fn k_pool_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.layers[layer_idx].k_pool
    }

    /// Get the full V cache pool pointer for a layer.
    pub fn v_pool_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.layers[layer_idx].v_pool
    }

    /// Cache stride in elements (for FP8/BF16 kernels that need explicit stride).
    /// Same for all layers (element count is dtype-independent).
    pub fn cache_stride(&self) -> usize {
        self.config.cache_stride_elements()
    }

    /// Block stride in bytes (for NVFP4 kernels), using the uniform dtype.
    pub fn block_stride_bytes(&self) -> usize {
        self.config.block_bytes()
    }

    /// Block stride in bytes for a specific attention layer.
    pub fn block_stride_bytes_for_layer(&self, layer_idx: usize) -> usize {
        self.layers[layer_idx].block_stride
    }

    /// NVFP4 data section size in bytes per block (uniform).
    pub fn nvfp4_data_bytes(&self) -> usize {
        self.config.nvfp4_data_bytes()
    }

    /// Turbo4 data section bytes (same layout as NVFP4: 4-bit packed).
    pub fn turbo4_data_bytes(&self) -> usize {
        self.config.turbo4_data_bytes()
    }

    /// Turbo3 data section bytes (3-bit packed).
    pub fn turbo3_data_bytes(&self) -> usize {
        self.config.turbo3_data_bytes()
    }

    /// Turbo8 data section bytes (FP8 E4M3 per element).
    pub fn turbo8_data_bytes(&self) -> usize {
        self.config.turbo8_data_bytes()
    }

    /// Turbo4 scale section bytes (same layout as NVFP4: FP8 per-group).
    pub fn turbo4_scale_bytes(&self) -> usize {
        self.config.turbo4_scale_bytes()
    }

    /// Cache configuration (read-only). Used by attention layers to query
    /// the `--high-speed-swap` HBM-shrink cap (`cache_blocks_per_seq`).
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Effective KV cache dtype for a specific attention layer.
    pub fn dtype_for_layer(&self, layer_idx: usize) -> KvCacheDtype {
        self.layers[layer_idx].dtype
    }

    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn dtype(&self) -> KvCacheDtype {
        self.config.dtype
    }

    /// Number of attention layers.
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Read K and V data for one block at one layer from GPU to host.
    ///
    /// Returns `(k_data, v_data)` where each is `block_stride` bytes.
    pub fn read_block(
        &self,
        layer_idx: usize,
        block_idx: u32,
        gpu: &dyn GpuBackend,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let stride = self.layers[layer_idx].block_stride;
        let k_ptr = self.k_cache_ptr(layer_idx, block_idx);
        let v_ptr = self.v_cache_ptr(layer_idx, block_idx);

        let mut k_data = vec![0u8; stride];
        let mut v_data = vec![0u8; stride];
        gpu.copy_d2h(k_ptr, &mut k_data)?;
        gpu.copy_d2h(v_ptr, &mut v_data)?;

        Ok((k_data, v_data))
    }

    /// Write K and V data for one block at one layer from host to GPU.
    pub fn write_block(
        &self,
        layer_idx: usize,
        block_idx: u32,
        k_data: &[u8],
        v_data: &[u8],
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let k_ptr = self.k_cache_ptr(layer_idx, block_idx);
        let v_ptr = self.v_cache_ptr(layer_idx, block_idx);
        gpu.copy_h2d(k_data, k_ptr)?;
        gpu.copy_h2d(v_data, v_ptr)?;
        Ok(())
    }

    /// Compute how many blocks can fit given available GPU memory.
    /// Accounts for mixed dtypes when layer_dtypes is set.
    pub fn compute_num_blocks(config: &KvCacheConfig, available_bytes: usize) -> Result<usize> {
        let bytes_per_block = config.block_bytes_kv_all_layers();
        if bytes_per_block == 0 {
            bail!("KV cache block size is zero");
        }
        Ok(available_bytes / bytes_per_block)
    }
}
