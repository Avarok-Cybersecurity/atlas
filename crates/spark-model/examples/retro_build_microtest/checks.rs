// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the correctness cells of `retro_build_microtest`: each k-means stage against the
//! CPU mirror, the whole build against the mirror and L33's oracle (determinism, CSR structure,
//! value sums, centroids, assignment agreement), an appended decode-time update, and the zone
//! selection against the reference and `oracle_eval`.

use anyhow::{Context, Result};
use spark_model::port::retro::build::{self, Job, geometry, launch, reference};
use spark_model::port::retro::config::{self, RetroParams};
use spark_model::port::retro::state::IndexCounts;
use spark_model::port::retro::{oracle, oracle_eval, select};
use spark_runtime::gpu::GpuBackend;

use crate::fixtures::{
    BS, D, Dev, H, Kind, Layer, QH, Report, Rng, bf16_bytes, d2h, f32s, i32s, initial_job, members,
    read_all, read_clusters, rel_max_abs, run_build, same_bits, span,
};

/// Stage by stage against the mirror: mean, init, one segmented assignment, one CSR, one update.
pub fn stages(
    gpu: &dyn GpuBackend,
    l: &Layer,
    init_len: u32,
    d: &Dev,
    rep: &mut Report,
) -> Result<()> {
    let job = initial_job(init_len)?;
    let k = launch::Kernels::resolve(gpu).context("retro_kmeans_ada missing")?;
    let lay = geometry::scratch_layout(H, &job, l.len.div_ceil(BS));
    let s = launch::Scratch::at(d.scratch, &lay);
    let bt: Vec<u8> = l.table.iter().flat_map(|b| b.to_le_bytes()).collect();
    gpu.copy_h2d(&bt, s.block_table)?;
    let pool = build::pool_args(&l.kv, s.block_table);
    let out = launch::IndexOut::of(&d.idx, &job);
    let st = gpu.default_stream();
    launch::mean(gpu, &k, &pool, &job, &s, st)?;
    launch::init(gpu, &k, &pool, &job, &out, &s, st)?;
    launch::assign(gpu, &k, &pool, &job, &out, &s, true, st)?;
    let train = launch::CsrOut::training(&s, &job);
    launch::csr(gpu, &k, H, &job, &s, train, st)?;
    gpu.synchronize(st)?;
    let (n, nc, h) = (job.n_tokens as usize, job.n_centroids as usize, H as usize);
    let means = f32s(&d2h(gpu, s.mean, h * D * 4)?);
    let asg = i32s(&d2h(gpu, s.assign, h * n * 4)?);
    let cs = i32s(&d2h(gpu, s.cstart, h * (nc + 1) * 4)?);
    let sz = i32s(&d2h(gpu, s.csize, h * nc * 4)?);
    let mem = i32s(&d2h(gpu, s.members, h * n * 4)?);
    let inits: Vec<Vec<f32>> = read_all(gpu, &d.idx, nc)?
        .into_iter()
        .map(|c| c.cent)
        .collect();
    launch::update(gpu, &k, &pool, &job, &out, &s, train, false, st)?;
    gpu.synchronize(st)?;
    let updated = read_all(gpu, &d.idx, nc)?;
    let (mut ok_mean, mut ok_init, mut ok_csr, mut ok_upd) = (true, true, true, true);
    let (mut miss, mut ties) = (0usize, 0usize);
    for hh in 0..h {
        let keys = span(&l.keys[hh], &job);
        let m = reference::mean(&keys, job.n_tokens);
        ok_mean &= same_bits(&m, &means[hh * D..(hh + 1) * D]);
        let c = reference::center(&keys, &m);
        let ci = reference::init(&c, &job);
        ok_init &= same_bits(&ci, &inits[hh]);
        let a = reference::assign(&c, &ci, &job, true);
        let ag = &asg[hh * n..(hh + 1) * n];
        for t in (0..n).filter(|&t| a[t] != ag[t]) {
            let row = &c[t * D..(t + 1) * D];
            let score = |i: i32| match usize::try_from(i) {
                Ok(i) => reference::dot(row, &ci[i * D..(i + 1) * D]),
                Err(_) => f32::NEG_INFINITY,
            };
            let (x, y) = (score(a[t]), score(ag[t]));
            if (x - y).abs() <= 1e-3 * x.abs().max(1.0) {
                ties += 1;
            } else {
                miss += 1;
            }
        }
        let (rs, rz, rm) = reference::csr(ag, job.n_centroids, job.first_pos);
        ok_csr &= cs[hh * (nc + 1)..hh * (nc + 1) + nc] == rs[..]
            && sz[hh * nc..(hh + 1) * nc] == rz[..]
            && mem[hh * n..hh * n + rm.len()] == rm[..];
        let (mut cu, mut vs) = (ci.clone(), vec![0.0f32; nc * D]);
        reference::update(
            &c,
            None,
            &m,
            &mut cu,
            &mut vs,
            (&rs, &rz, &rm),
            job.first_pos,
            false,
        );
        ok_upd &= same_bits(&cu, &updated[hh].cent);
    }
    rep.check(ok_mean, "key mean bit-equal to the mirror");
    rep.check(ok_init, "initial centroids bit-equal to the mirror");
    rep.check(
        miss == 0,
        &format!("segmented assignment: {miss} real mismatches, {ties} near ties"),
    );
    rep.check(ok_csr, "stable CSR equal to the mirror");
    rep.check(ok_upd, "normalized centroid update bit-equal to the mirror");
    Ok(())
}

/// The whole initial build: determinism, structure, and agreement with the mirror and the oracle.
pub fn end_to_end(
    gpu: &dyn GpuBackend,
    l: &Layer,
    init_len: u32,
    d: &mut Dev,
    kind: Kind,
    rep: &mut Report,
) -> Result<Job> {
    let job = initial_job(init_len)?;
    let nc = job.n_centroids as usize;
    run_build(gpu, l, d, &job)?;
    let first = read_all(gpu, &d.idx, nc)?;
    run_build(gpu, l, d, &job)?;
    let second = read_all(gpu, &d.idx, nc)?;
    let det = first.iter().zip(&second).all(|(a, b)| {
        same_bits(&a.cent, &b.cent) && same_bits(&a.vsum, &b.vsum) && a.lists == b.lists
    });
    rep.check(det, "two builds bitwise equal");
    let heads: Vec<(Vec<f32>, Vec<f32>)> = (0..H as usize)
        .map(|h| (span(&l.keys[h], &job), span(&l.vals[h], &job)))
        .collect();
    let mirror = reference::build(&heads, &job);
    // L33's f64 oracle is slow at length; it checks the Gaussian layers up to 16K tokens.
    let with_l33 = kind == Kind::Gaussian && job.n_tokens <= 16_384;
    let l33 = if with_l33 {
        let spec = oracle::KmeansSpec {
            centroids: nc,
            segments: job.n_segments as usize,
            iters: job.iters as usize,
        };
        let kvs: Vec<oracle::HeadKv> = (0..H as usize)
            .map(|h| oracle::HeadKv {
                k: &l.keys[h],
                v: &l.vals[h],
                head_dim: D,
            })
            .collect();
        oracle::build_heads(&kvs, job.first_pos as usize, job.end_pos() as usize, spec)
    } else {
        Vec::new()
    };
    let (mut structure, mut vs_ok, mut worst) = (true, true, 0.0f32);
    let (mut agree, mut agree_l33) = (0usize, 0usize);
    for (h, (o, r)) in first.iter().zip(&mirror).enumerate() {
        let mut seen = vec![false; job.n_tokens as usize];
        for c in 0..nc {
            let m = &o.lists[c];
            structure &= m.len() == o.size[c] as usize && m.windows(2).all(|w| w[0] < w[1]);
            for &p in m {
                let t = (p as u32).wrapping_sub(job.first_pos) as usize;
                structure &= t < seen.len() && !std::mem::replace(&mut seen[t], true);
            }
            let rm = members(r, c);
            agree += m.iter().filter(|&p| rm.contains(p)).count();
            if let Some(oh) = l33.get(h) {
                agree_l33 += m
                    .iter()
                    .filter(|&&p| oh.lists[c].contains(&(p as u32)))
                    .count();
            }
            if m[..] == rm[..] && !m.is_empty() {
                let rows = c * D..(c + 1) * D;
                vs_ok &= same_bits(&o.vsum[rows.clone()], &r.value_sum[rows.clone()]);
                worst = worst.max(rel_max_abs(&o.cent[rows.clone()], &r.centroids[rows]));
            }
        }
        structure &= seen.iter().all(|&s| s);
    }
    let total = f64::from(job.n_tokens) * f64::from(H);
    let pct = |x: usize| 100.0 * x as f64 / total;
    rep.check(
        structure,
        "CSR is a permutation of the span, members ascending, sizes match",
    );
    rep.check(
        kind != Kind::Gaussian || agree as f64 >= 0.999 * total,
        &format!("assignment agrees with the mirror on {:.4}%", pct(agree)),
    );
    if with_l33 {
        rep.check(
            agree_l33 as f64 >= 0.99 * total,
            &format!(
                "assignment agrees with L33's oracle on {:.4}%",
                pct(agree_l33)
            ),
        );
    }
    rep.check(vs_ok, "value sums bit-equal on clusters with equal members");
    rep.check(
        worst <= 1e-2,
        &format!("equal-member centroids: worst max-abs relative error {worst:.2e}"),
    );
    Ok(job)
}

/// A decode-time update: 1,024 more tokens clustered and appended behind the initial clusters.
pub fn update(
    gpu: &dyn GpuBackend,
    l: &Layer,
    d: &mut Dev,
    init: &Job,
    rep: &mut Report,
) -> Result<()> {
    let p = RetroParams::default();
    let first = init.end_pos();
    if first + p.update_interval > l.len {
        return Ok(());
    }
    let (co, nc0) = (init.n_centroids, init.n_centroids as usize);
    let job = Job::new(
        first,
        p.update_interval,
        p.update_centroids(),
        1,
        p.kmeans_iters,
        co,
        init.n_tokens,
    )?;
    let before = read_clusters(gpu, &d.idx, 0, 0, nc0)?;
    run_build(gpu, l, d, &job)?;
    let after = read_clusters(gpu, &d.idx, 0, 0, nc0)?;
    let added = read_clusters(gpu, &d.idx, 0, nc0, job.n_centroids as usize)?;
    let r = reference::build_head(&span(&l.keys[0], &job), &span(&l.vals[0], &job), &job);
    let lists_ok =
        (0..job.n_centroids as usize).all(|c| added.lists[c].len() == members(&r, c).len());
    let total: i32 = added.size.iter().sum();
    let untouched = same_bits(&before.cent, &after.cent) && before.lists == after.lists;
    rep.check(untouched, "update left the earlier clusters untouched");
    rep.check(
        lists_ok && total as u32 == job.n_tokens,
        "update appended its clusters at the offsets",
    );
    Ok(())
}

/// `n` random selections on the built index against the reference and `oracle_eval::select`.
pub fn selection(
    gpu: &dyn GpuBackend,
    d: &mut Dev,
    job: &Job,
    n: usize,
    rep: &mut Report,
) -> Result<()> {
    let p = RetroParams::default();
    let nc = job.n_centroids;
    let np = config::nprobe(nc, p.retrieval_budget);
    let es = config::estimation_clusters(nc, p.estimation_budget, np);
    d.idx.counts = IndexCounts {
        n_centroids: nc,
        nprobe: np,
        es,
        indexed_end: job.end_pos(),
        sink: job.first_pos,
        listed: job.n_tokens,
    };
    gpu.copy_h2d(&d.idx.counts.to_bytes(), d.idx.header)?;
    let heads = read_all(gpu, &d.idx, nc as usize)?;
    let q_dev = gpu.alloc(QH as usize * D * 2)?;
    let (rc, ec, st) = (
        d.cfg.retrieval_cap as usize,
        d.cfg.es_cap as usize,
        gpu.default_stream(),
    );
    let mut r = Rng(0xC0FFEE);
    let (mut exact, mut ties, mut miss) = (0usize, 0usize, 0usize);
    let cfg = d.cfg.clone();
    for _ in 0..n {
        let qv: Vec<f32> = (0..QH as usize * D)
            .map(|_| reference::bf16(2.0 * r.uni()))
            .collect();
        gpu.copy_h2d(&bf16_bytes(&qv), q_dev)?;
        select::select_zones(gpu, &d.idx, q_dev, &cfg, &mut d.zl, st)?;
        gpu.synchronize(st)?;
        let ret = i32s(&d2h(gpu, d.zl.retrieval_ids, H as usize * rc * 4)?);
        let est = i32s(&d2h(gpu, d.zl.estimation_ids, H as usize * ec * 4)?);
        let cnt = i32s(&d2h(gpu, d.zl.counts, H as usize * 8)?);
        for (h, hc) in heads.iter().enumerate() {
            let ids = |v: &[i32], base: usize, k: i32| -> Vec<u32> {
                v[base..base + k.max(0) as usize]
                    .iter()
                    .map(|&x| x as u32)
                    .collect()
            };
            let (gr, ge) = (
                ids(&ret, h * rc, cnt[2 * h]),
                ids(&est, h * ec, cnt[2 * h + 1]),
            );
            let qg = &qv[h * 4 * D..(h + 1) * 4 * D];
            let inp = select::reference::Input {
                centroids: &hc.cent,
                cluster_size: &hc.size,
                cap: nc as usize,
                nc: nc as usize,
                kv_heads: 1,
                group: 4,
                rows: 1,
                q: qg,
                scale: 0.0625,
                nprobe: np,
                es,
                r_cap: rc as u32,
                e_cap: ec as u32,
            };
            let mine = &select::reference::select(&inp)[0];
            let oh = oracle::HeadIndex {
                head_dim: D,
                centroids: hc.cent.clone(),
                value_sum: Vec::new(),
                sizes: hc.size.iter().map(|&s| s as u32).collect(),
                lists: Vec::new(),
            };
            let score = oracle_eval::score_clusters(&oh, nc as usize, qg, 4, 0.0625);
            let theirs = oracle_eval::select(&score, np as usize, es as usize);
            let near = |a: &[u32], b: &[u32]| {
                a.len() == b.len()
                    && a.iter().zip(b).all(|(&x, &y)| {
                        let (sx, sy) = (score[x as usize], score[y as usize]);
                        x == y || (sx - sy).abs() <= 1e-5 * sx.max(sy)
                    })
            };
            let same = |a: &[u32], b: &[u32], c: &[u32], e: &[u32]| a == c && b == e;
            if same(&gr, &ge, &mine.retrieval, &mine.estimation)
                && same(&gr, &ge, &theirs.retrieval, &theirs.estimation)
            {
                exact += 1;
            } else if near(&gr, &theirs.retrieval)
                && near(&ge, &theirs.estimation)
                && near(&gr, &mine.retrieval)
                && near(&ge, &mine.estimation)
            {
                ties += 1;
            } else {
                miss += 1;
            }
        }
    }
    let what =
        format!("selection: {exact} head-queries exact, {ties} near ties, {miss} mismatches");
    rep.check(miss == 0, &what);
    gpu.free(q_dev)
}
