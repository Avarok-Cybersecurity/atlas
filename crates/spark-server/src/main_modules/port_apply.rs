// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: validate and publish the port sprint's (`port/consumer-opt`)
//! serve flags and row values to the crates that read them.
//!
//! Called once from `serve_flags::publish_kernel_flags`, immediately before
//! the boot `target defaults` line: after `validate_serve_args`, before any
//! model is built and before any port row resolver is first read, so a setter
//! published here is in force for the whole serve. `validate` runs first; an
//! error there is a startup error, never a panic inside a resolver's
//! `OnceLock`.
//!
//! One line per lane at each marker. Crates below spark-model (spark-runtime)
//! never read rows: a lane's `@port:apply` line resolves its row here
//! (`spark_model::layers::ops::port_rows::<row>::row()`) and passes the plain
//! value into the lane's own setter. Lines are inserted immediately BEFORE the
//! markers below by the coordinator's fragment apply pass.

use crate::cli::serve_args_port::PortArgs;

/// Publish every port flag and row value. Empty until a lane adds a line.
#[allow(unused_variables)]
pub(crate) fn apply(args: &PortArgs) -> anyhow::Result<()> {
    spark_runtime::weights::keep_packed_iq4::set_enabled(
        spark_model::layers::ops::port_rows::gguf_native_iq4::row(),
    );
    spark_model::port::affine_opts::apply(spark_model::port::affine_opts::AffineCli {
        bits: args.w4_bits,
        group: args.w4_group,
        clip_search: args.w4_clip_search,
    });
    spark_model::port::packed_head::set_requested(args.lm_head_packed);
    if args.lm_head_packed {
        spark_runtime::weights::keep_packed_iq4::set_keep_head(true);
    }
    spark_runtime::pinned_pool::set_default_chunk_mb(args.pinned_pool_chunk_mb);
    spark_runtime::radix_tree::set_lru_index_enabled(
        spark_model::layers::ops::port_rows::radix_lru_index::row(),
    );
    spark_runtime::radix_tree::set_cow_tail_enabled(
        spark_model::layers::ops::port_rows::prefix_cow_tail::row(),
    );
    // @port:apply
    Ok(())
}

/// Reject a port flag or row value that cannot be served (an unknown string
/// row value, a contradictory flag pair). Empty until a lane adds a line.
#[allow(unused_variables)]
pub(crate) fn validate(args: &PortArgs) -> anyhow::Result<()> {
    spark_model::port::affine_opts::validate(spark_model::port::affine_opts::AffineCli {
        bits: args.w4_bits,
        group: args.w4_group,
        clip_search: args.w4_clip_search,
    })?;
    spark_runtime::pinned_pool::validate_chunk_mb(args.pinned_pool_chunk_mb)?;
    // @port:validate
    Ok(())
}
