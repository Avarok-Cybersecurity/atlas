// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).
//! GPU oracle for the bounded view's table kernel (`kernels/ada/common/kv_view_table_ada.cu`,
//! row `kv_view_selection`) against its host twin `spark_model::port::kv_view::view::build_row`.
//!
//! WHAT THIS PINS (each case prints PASS/FAIL; any FAIL exits non-zero):
//!  * **kernel == host**, every table entry and every length, for random views over random
//!    real tables: one row with the arena as its capacity (the single-sequence graph), a
//!    batched step of 8 rows with padding rows, and verify-style groups of 5 rows sharing one
//!    table with lengths one apart;
//!  * **identity**: with no descriptor active every row's table and length equal the real ones;
//!  * **most specific wins**: two matching descriptors, the one with the larger last page
//!    applies;
//!  * **a planted control**: one anchor of an active descriptor changed in the real table must
//!    turn that row (and only that row) back to its real table, so a comparator that cannot
//!    fail does not read as a pass;
//!  * **the extent**: entries past each row's view are left untouched (sentinel bytes).
//!
//! Run on the Ada card, in a build of the ada target:
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!       --example kv_view_table_microtest --features cuda,gpu-examples
use anyhow::{Context, Result, ensure};
use spark_model::port::kv_view::device::{DESC_ENTRIES, KERNEL_FN, KERNEL_MODULE};
use spark_model::port::kv_view::view::{DESC_WORDS, Desc, ViewShape, build_row, pages_of};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const BS: u32 = 16;
const SEL_CAP: u32 = 1032;
const SENTINEL: i32 = -7;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

fn bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn ints(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One step: `rows` real tables (each `stride` entries) and lengths, and the active views.
struct Step {
    rows: Vec<Vec<u32>>,
    lens: Vec<u32>,
    stride: u32,
    row_cap: u32,
    descs: Vec<Desc>,
}

fn run(gpu: &dyn GpuBackend, k: KernelHandle, st: &Step) -> Result<(Vec<i32>, Vec<i32>)> {
    let n = st.rows.len();
    let stride = st.stride as usize;
    let mut real = vec![0i32; n * stride];
    for (r, row) in st.rows.iter().enumerate() {
        for (j, &b) in row.iter().enumerate() {
            real[r * stride + j] = b as i32;
        }
    }
    let lens: Vec<i32> = st.lens.iter().map(|&l| l as i32).collect();
    let mut desc = vec![0i32; DESC_ENTRIES * DESC_WORDS];
    let mut pool = vec![0i32; DESC_ENTRIES * SEL_CAP as usize];
    for (e, d) in st.descs.iter().enumerate() {
        desc[e * DESC_WORDS..(e + 1) * DESC_WORDS].copy_from_slice(&d.words());
        for (j, &p) in d.shape.sel_pages.iter().enumerate() {
            pool[e * SEL_CAP as usize + j] = p as i32;
        }
    }
    let alloc = |v: &[i32]| -> Result<DevicePtr> {
        let p = gpu.alloc(v.len().max(1) * 4)?;
        gpu.copy_h2d(&bytes(v), p)?;
        Ok(p)
    };
    let (d_real, d_lens, d_desc, d_pool) =
        (alloc(&real)?, alloc(&lens)?, alloc(&desc)?, alloc(&pool)?);
    let d_out = alloc(&vec![SENTINEL; n * stride])?;
    let d_out_lens = alloc(&vec![SENTINEL; n])?;
    let stream = gpu.default_stream();
    KernelLaunch::new(gpu, k)
        .grid([n as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(d_real)
        .arg_ptr(d_lens)
        .arg_u32(n as u32)
        .arg_u32(st.stride)
        .arg_u32(st.row_cap)
        .arg_u32(BS)
        .arg_ptr(d_desc)
        .arg_u32(DESC_ENTRIES as u32)
        .arg_ptr(d_pool)
        .arg_u32(SEL_CAP)
        .arg_ptr(d_out)
        .arg_ptr(d_out_lens)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut out = vec![0u8; n * stride * 4];
    let mut out_lens = vec![0u8; n * 4];
    gpu.copy_d2h(d_out, &mut out)?;
    gpu.copy_d2h(d_out_lens, &mut out_lens)?;
    for p in [d_real, d_lens, d_desc, d_pool, d_out, d_out_lens] {
        gpu.free(p)?;
    }
    Ok((ints(&out), ints(&out_lens)))
}

/// Compare the kernel with the host twin; returns the number of mismatches.
fn check(name: &str, gpu: &dyn GpuBackend, k: KernelHandle, st: &Step) -> Result<usize> {
    let (out, out_lens) = run(gpu, k, st)?;
    let refs: Vec<Option<&Desc>> = st.descs.iter().map(Some).collect();
    let stride = st.stride as usize;
    let mut bad = 0usize;
    for (r, row) in st.rows.iter().enumerate() {
        let (want, want_len) = build_row(row, st.lens[r], BS, &refs);
        let got = &out[r * stride..r * stride + want.len()];
        let got_len = out_lens[r];
        let tail_untouched = out[r * stride + want.len()..(r + 1) * stride]
            .iter()
            .all(|&v| v == SENTINEL);
        let same = got.iter().zip(&want).all(|(&g, &w)| g == w as i32);
        if !same || got_len != want_len as i32 || !tail_untouched {
            bad += 1;
            if bad <= 3 {
                eprintln!(
                    "  row {r}: len {} -> kernel {got_len} host {want_len}; tables equal {same}; \
                     extent untouched {tail_untouched}",
                    st.lens[r]
                );
            }
        }
    }
    println!(
        "{} {name}: {} rows",
        if bad == 0 { "PASS" } else { "FAIL" },
        st.rows.len()
    );
    Ok(bad)
}

fn random_view(g: &mut Lcg, row: &[u32], len: u32) -> Desc {
    let pages = pages_of(len, BS);
    let tail_page = 1 + g.next() % (pages - 1).max(1);
    let sel: Vec<u32> = (0..tail_page)
        .filter(|_| g.next() % 4 == 0)
        .take(SEL_CAP as usize)
        .collect();
    Desc::bind(
        ViewShape {
            sel_pages: sel,
            tail_page,
        },
        row,
        len,
        BS,
    )
    .expect("a random view binds")
}

fn random_row(g: &mut Lcg, pages: u32) -> Vec<u32> {
    let base = g.next() % 100_000;
    (0..pages).map(|i| base + i * 3 + g.next() % 3).collect()
}

fn main() -> Result<()> {
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let k = gpu
        .kernel(KERNEL_MODULE, KERNEL_FN)
        .context("build with AVAROK_TARGET_HW=ada")?;
    let mut g = Lcg(0x32_2026_1112);
    let mut bad = 0usize;

    // One row, the arena as its capacity, a long context.
    for _ in 0..20 {
        let pages = 64 + g.next() % 16_000;
        let len = (pages - 1) * BS + 1 + g.next() % BS;
        let row = random_row(&mut g, pages);
        let d = random_view(&mut g, &row, len);
        let st = Step {
            rows: vec![row],
            lens: vec![len],
            stride: pages,
            row_cap: 4 << 20,
            descs: vec![d],
        };
        bad += check("single row", &gpu, k, &st)?;
    }

    // A batched step: 8 rows, two padding, three with views.
    let stride = 4097u32;
    let mut rows = Vec::new();
    let mut lens = Vec::new();
    let mut descs = Vec::new();
    for r in 0..8 {
        let pages = 32 + g.next() % (stride - 32);
        let len = if r == 2 || r == 6 {
            0
        } else {
            (pages - 1) * BS + 1 + g.next() % BS
        };
        let row = random_row(&mut g, pages);
        if len > 0 && r % 2 == 1 {
            descs.push(random_view(&mut g, &row, len));
        }
        rows.push(row);
        lens.push(len);
    }
    let batched = Step {
        rows,
        lens,
        stride,
        row_cap: stride,
        descs,
    };
    bad += check("batched with padding", &gpu, k, &batched)?;

    // Verify-style: 5 rows of one sequence, lengths one apart.
    let pages = 900u32;
    let row = random_row(&mut g, pages);
    let base_len = (pages - 1) * BS - 4;
    let d = random_view(&mut g, &row, base_len);
    let st = Step {
        rows: vec![row.clone(); 5],
        lens: (0..5).map(|i| base_len + i).collect(),
        stride: pages,
        row_cap: pages,
        descs: vec![d.clone()],
    };
    bad += check("verify group", &gpu, k, &st)?;

    // Identity: no descriptor.
    let st = Step {
        rows: vec![row.clone()],
        lens: vec![base_len],
        stride: pages,
        row_cap: pages,
        descs: vec![],
    };
    bad += check("identity", &gpu, k, &st)?;

    // Most specific wins.
    let early = random_view(&mut g, &row, base_len - 200 * BS);
    let st = Step {
        rows: vec![row.clone()],
        lens: vec![base_len],
        stride: pages,
        row_cap: pages,
        descs: vec![early, d.clone()],
    };
    bad += check("most specific", &gpu, k, &st)?;

    // Planted control: a view that drops pages; break one anchor of row 1, which must
    // fall back to its real table while row 0 keeps the view.
    let ctrl = Desc::bind(
        ViewShape {
            sel_pages: (0..4).collect(),
            tail_page: 100,
        },
        &row,
        base_len,
        BS,
    )
    .expect("the control view binds");
    let mut broken = row.clone();
    broken[ctrl.shape.tail_page as usize] ^= 0x4000_0000;
    let st = Step {
        rows: vec![row.clone(), broken.clone()],
        lens: vec![base_len, base_len],
        stride: pages,
        row_cap: pages,
        descs: vec![ctrl],
    };
    let (out, out_lens) = run(&gpu, k, &st)?;
    let control = out_lens[1] == base_len as i32
        && out[pages as usize..pages as usize + 3]
            == [broken[0] as i32, broken[1] as i32, broken[2] as i32]
        && out_lens[0] != base_len as i32;
    println!(
        "{} planted control: a broken anchor turns only its own row dense",
        if control { "PASS" } else { "FAIL" }
    );
    if !control {
        bad += 1;
    }
    ensure!(bad == 0, "{bad} case(s) failed");
    println!("kv_view_table_microtest: all cases PASS");
    Ok(())
}
