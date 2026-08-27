// SPDX-License-Identifier: AGPL-3.0-only

//! Which scale layouts count as per-row.
//!
//! The predicate decides whether a checkpoint's FP8 tensor goes to the
//! row-wise cuBLASLt GEMM or stays on the block-scaled kernels. Getting it
//! wrong in the permissive direction is the dangerous one: a block grid fed to
//! the row-wise path, or a per-row buffer fed to `w8a16`, is smaller than the
//! index space the consumer walks, so it reads in-bounds garbage instead of
//! faulting.

use std::collections::HashMap;

use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{proj_is_fp8_per_row, scale_is_per_row};

fn projection_store(
    weight_dtype: WeightDtype,
    weight_shape: Vec<usize>,
    scale_shape: Option<Vec<usize>>,
) -> WeightStore {
    let mut tensors = HashMap::from([(
        "proj.weight".to_string(),
        WeightTensor {
            ptr: DevicePtr::NULL,
            shape: weight_shape,
            dtype: weight_dtype,
        },
    )]);
    if let Some(shape) = scale_shape {
        tensors.insert(
            "proj.weight_scale".to_string(),
            WeightTensor {
                ptr: DevicePtr::NULL,
                shape,
                dtype: WeightDtype::BF16,
            },
        );
    }
    WeightStore::from_map(tensors)
}

/// `[N]` and `[N,1]` are the two spellings a per-channel export uses.
#[test]
fn per_row_shapes_are_accepted() {
    assert!(scale_is_per_row(4096, &[4096], 4096), "[N]");
    assert!(scale_is_per_row(4096, &[4096, 1], 4096), "[N,1]");
}

/// A per-TENSOR scalar belongs to the block/broadcast arm.
#[test]
fn a_scalar_scale_is_not_per_row() {
    assert!(!scale_is_per_row(4096, &[1], 1));
    assert!(!scale_is_per_row(4096, &[], 1));
}

/// A `[N/128, K/128]` block grid is not per-row — this is the case that would
/// silently misread if it were let through.
#[test]
fn a_block_grid_is_not_per_row() {
    // 4096x5120 weight -> 32 x 40 grid = 1280 elements, nothing like N.
    assert!(!scale_is_per_row(4096, &[32, 40], 1280));
}

/// Right element count, wrong axis: `[1,N]` is a per-COLUMN vector. It would
/// pass a naive element-count check and multiply the wrong dimension.
#[test]
fn a_column_vector_is_rejected_even_with_n_elements() {
    assert!(!scale_is_per_row(4096, &[1, 4096], 4096));
}

#[test]
fn projection_metadata_gates_the_rowwise_loader_path() {
    assert!(proj_is_fp8_per_row(
        &projection_store(WeightDtype::FP8E4M3, vec![4, 8], Some(vec![4, 1])),
        "proj"
    ));
    assert!(!proj_is_fp8_per_row(
        &projection_store(WeightDtype::BF16, vec![4, 8], Some(vec![4, 1])),
        "proj"
    ));
    assert!(!proj_is_fp8_per_row(
        &projection_store(WeightDtype::FP8E4M3, vec![32], Some(vec![4, 1])),
        "proj"
    ));
    assert!(!proj_is_fp8_per_row(
        &projection_store(WeightDtype::FP8E4M3, vec![4, 8], None),
        "proj"
    ));
}
