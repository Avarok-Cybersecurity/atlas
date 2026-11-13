// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from rayline-ai/vllm-arc (vllm_arc/arc_heads.py and vllm_arc/heads.py, commit 4ee6a9f), Copyright
// 2026 Atlas Futures, Inc., licensed under the Apache License 2.0. Translated to Rust; see stage_one/mod.rs.
//! Responsibility: build a [`StageOne`] from tensors or from a release directory: the same refusals as the reference
//! loader (pinned hashes, one consistent release, no unported branch, no unknown or missing tensor, checked shapes).

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{Candidate, Dims, SUPPORTED, Stage1Tensors, StageOne, UNPORTED};
use crate::route_gate::error::{GResult, GateError, key_err, value_err};
use crate::route_gate::tensorio::{Tensor, TensorFile};

pub const MANIFEST: &str = "arc_release.json";
pub const CONFIG: &str = "config.json";
pub const WEIGHTS: &str = "model.safetensors";
const PREFIX: &str = "stage_one.";

fn unsupported<T>(msg: impl Into<String>) -> GResult<T> {
    Err(GateError::Stage1Unsupported(msg.into()))
}

/// Lower-case hex SHA-256 of a file, streamed.
pub fn release_sha256(path: impl AsRef<Path>) -> GResult<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn read_json(path: &Path) -> GResult<Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

/// Same refusals as the reference loader: the manifest pins `config.json` and the weights, and both name one release.
/// Returns the parsed `config.json`.
pub fn check_release(release_dir: impl AsRef<Path>, verify_hashes: bool) -> GResult<Value> {
    let dir = release_dir.as_ref();
    let manifest = read_json(&dir.join(MANIFEST))?;
    let cfg = read_json(&dir.join(CONFIG))?;
    if verify_hashes {
        for name in [CONFIG, WEIGHTS] {
            let pinned = manifest
                .get("files")
                .and_then(|f| f.get(name))
                .and_then(Value::as_str);
            let ok = match pinned {
                Some(p) => release_sha256(dir.join(name))? == p,
                None => false,
            };
            if !ok {
                return Err(GateError::ReleaseRefused(format!(
                    "{name} is not the file {MANIFEST} pins"
                )));
            }
        }
    }
    let arc = cfg.get("arc").cloned().unwrap_or(Value::Null);
    if arc.get("release") != manifest.get("name")
        || arc.get("package_sha256") != manifest.get("package_sha256")
    {
        return Err(GateError::ReleaseRefused(format!(
            "{CONFIG} and {MANIFEST} name different releases"
        )));
    }
    let has_head = match arc.get("stage_one") {
        None | Some(Value::Null) => false,
        Some(Value::Object(o)) => !o.is_empty(),
        Some(_) => true,
    };
    if !has_head {
        return Err(GateError::ReleaseRefused(
            "the release carries no Stage-1 head".to_string(),
        ));
    }
    Ok(cfg)
}

fn rank_dim(t: &BTreeMap<String, Tensor>, name: &str, rank: usize, axis: usize) -> GResult<usize> {
    let tensor = &t[name];
    if tensor.shape.len() != rank {
        return unsupported(format!(
            "tensor {name:?} has shape {:?}, expected rank {rank}",
            tensor.shape
        ));
    }
    Ok(tensor.shape[axis])
}

fn to_usize(name: &str, data: &[f64]) -> GResult<Vec<usize>> {
    data.iter()
        .map(|&v| {
            if v < 0.0 || v.fract() != 0.0 {
                unsupported(format!(
                    "tensor {name:?} holds {v}, expected a non-negative index"
                ))
            } else {
                Ok(v as usize)
            }
        })
        .collect()
}

impl StageOne {
    /// Builds the head from the `stage_one.*` tensors (prefix stripped). `candidates` and `regimes` come from the
    /// release's `config.json`; without them the head names them `candidate-i` and `regime-i`.
    pub fn new(
        tensors: BTreeMap<String, Tensor>,
        candidates: Option<Vec<Candidate>>,
        regimes: Option<Vec<String>>,
    ) -> GResult<Self> {
        let mut t = tensors;
        t.remove("auxiliary");
        for name in t.keys() {
            if let Some((_, why)) = UNPORTED.iter().find(|(n, _)| n == name) {
                return unsupported(format!(
                    "tensor {name:?} selects an unported branch ({why})"
                ));
            }
        }
        let unknown: Vec<&String> = t
            .keys()
            .filter(|k| !SUPPORTED.contains(&k.as_str()))
            .collect();
        if !unknown.is_empty() {
            return unsupported(format!(
                "unknown Stage-1 tensors {unknown:?}: refusing to guess what they do"
            ));
        }
        let missing: Vec<&str> = SUPPORTED
            .iter()
            .copied()
            .filter(|k| !t.contains_key(*k))
            .collect();
        if !missing.is_empty() {
            return unsupported(format!("Stage-1 tensors missing: {missing:?}"));
        }
        let dims = Self::dims_and_check(&t)?;
        let n_cl = dims.c * dims.l;
        let alpha_len = t["ord_alpha"].numel();
        if alpha_len != 1 && alpha_len != n_cl {
            return unsupported(format!(
                "ord_alpha has {alpha_len} values; expected 1 or {n_cl}"
            ));
        }
        let ens_point = to_usize("ens_point", &t["ens_point"].data)?;
        let ens_draws = to_usize("ens_draws", &t["ens_draws"].data)?;
        let ens_owner = to_usize("ens_owner", &t["ens_owner"].data)?;
        if ens_point.iter().any(|&i| i >= dims.m) {
            return unsupported("ens_point indexes outside the member axis");
        }
        if ens_draws.iter().any(|&i| i >= dims.m) {
            return unsupported("ens_draws indexes outside the member axis");
        }
        let mut take = |name: &str| t.remove(name).map(|x| x.data).unwrap_or_default();
        let flag = |v: Vec<f64>| v.into_iter().map(|x| x != 0.0).collect::<Vec<bool>>();
        let tensors = Stage1Tensors {
            mean: take("mean"),
            scale: take("scale"),
            difficulty: take("difficulty"),
            logit_bias: take("logit_bias"),
            regime: take("regime"),
            v: take("v"),
            u: take("u"),
            w1: take("w1"),
            c1: take("c1"),
            w2: take("w2"),
            c2: take("c2"),
            embed: take("embed"),
            w_usage: take("w_usage"),
            b_usage: take("b_usage"),
            smear: take("smear"),
            low: take("low"),
            high: take("high"),
            usage_ok: flag(take("usage_ok")),
            p_ok: flag(take("p_ok")),
            ord_mean: take("ord_mean"),
            ord_axes: take("ord_axes"),
            ord_center: take("ord_center"),
            ord_spread: take("ord_spread"),
            ord_w1: take("ord_w1"),
            ord_e1: take("ord_e1"),
            ord_b1: take("ord_b1"),
            ord_w2: take("ord_w2"),
            ord_level: take("ord_level"),
            ord_b: take("ord_b"),
            ord_seen: flag(take("ord_seen")),
            ord_pool_glm: take("ord_pool_glm"),
            ord_pool_net: take("ord_pool_net"),
            ord_pool_n: to_usize("ord_pool_n", &take("ord_pool_n"))?,
            ord_recal_x: take("ord_recal_x"),
            ord_recal_c: take("ord_recal_c"),
            ord_recal_n: to_usize("ord_recal_n", &take("ord_recal_n"))?,
            ord_alpha: take("ord_alpha"),
            cal_a: take("cal_a"),
            cal_b: take("cal_b"),
            cal_platt: flag(take("cal_platt")),
            ens_point,
            ens_draws,
            ens_owner,
        };
        let candidates = candidates.unwrap_or_else(|| {
            (0..dims.c)
                .map(|i| Candidate::named(&format!("candidate-{i}")))
                .collect()
        });
        if candidates.len() != dims.c {
            return unsupported(format!(
                "{} candidates in the config but the head scores {}",
                candidates.len(),
                dims.c
            ));
        }
        let regimes =
            regimes.unwrap_or_else(|| (0..dims.reg).map(|i| format!("regime-{i}")).collect());
        if regimes.len() != dims.reg {
            return unsupported("regime names and the regime tensor disagree");
        }
        Ok(Self {
            dims,
            candidates,
            regimes,
            t: tensors,
        })
    }

    fn dims_and_check(t: &BTreeMap<String, Tensor>) -> GResult<Dims> {
        let m = rank_dim(t, "difficulty", 2, 0)?;
        let d = rank_dim(t, "difficulty", 2, 1)?;
        let c = rank_dim(t, "logit_bias", 2, 1)?;
        let l = rank_dim(t, "b_usage", 3, 2)?;
        let h = rank_dim(t, "w1", 3, 2)?;
        let e = rank_dim(t, "w2", 3, 2)?;
        let dd = rank_dim(t, "ord_axes", 2, 0)?;
        let p = rank_dim(t, "ord_pool_glm", 3, 2)?;
        let k = rank_dim(t, "ord_recal_x", 3, 2)?;
        let reg = rank_dim(t, "regime", 2, 1)?;
        let r = rank_dim(t, "u", 3, 2)?;
        let draws = rank_dim(t, "ens_draws", 1, 0)?;
        let points = rank_dim(t, "ens_point", 1, 0)?;
        let ho = rank_dim(t, "ord_w1", 3, 2)?;
        let dims = Dims {
            m,
            d,
            c,
            l,
            h,
            e,
            reg,
            r,
            dd,
            ho,
            p,
            k,
            draws,
            points,
        };
        let want: Vec<(&str, Vec<usize>)> = vec![
            ("mean", vec![d]),
            ("scale", vec![d]),
            ("difficulty", vec![m, d]),
            ("logit_bias", vec![m, c]),
            ("regime", vec![m, reg]),
            ("v", vec![m, r, d]),
            ("u", vec![m, c, r]),
            ("w1", vec![m, d, h]),
            ("c1", vec![m, h]),
            ("w2", vec![m, h, e]),
            ("c2", vec![m, e]),
            ("embed", vec![m, c, e]),
            ("w_usage", vec![m, c, l, d]),
            ("b_usage", vec![m, c, l]),
            ("smear", vec![m, c, l]),
            ("low", vec![m, c, l]),
            ("high", vec![m, c, l]),
            ("usage_ok", vec![m, c, l]),
            ("p_ok", vec![m, c]),
            ("ord_mean", vec![d]),
            ("ord_axes", vec![dd, d]),
            ("ord_center", vec![l, dd]),
            ("ord_spread", vec![l, dd]),
            ("ord_w1", vec![l, dd, ho]),
            ("ord_e1", vec![l, c, ho]),
            ("ord_b1", vec![l, ho]),
            ("ord_w2", vec![l, ho]),
            ("ord_level", vec![l, c]),
            ("ord_b", vec![l]),
            ("ord_seen", vec![l, c]),
            ("ord_pool_glm", vec![c, l, p]),
            ("ord_pool_net", vec![c, l, p]),
            ("ord_pool_n", vec![c, l]),
            ("ord_recal_x", vec![c, l, k]),
            ("ord_recal_c", vec![c, l, k]),
            ("ord_recal_n", vec![c, l]),
            ("cal_a", vec![points, c]),
            ("cal_b", vec![points, c]),
            ("cal_platt", vec![points, c]),
            ("ens_owner", vec![draws]),
        ];
        for (name, shape) in want {
            if t[name].shape != shape {
                return unsupported(format!(
                    "tensor {name:?} has shape {:?}, expected {shape:?}",
                    t[name].shape
                ));
            }
        }
        Ok(dims)
    }

    /// Loads the head of a release directory (after the pin and consistency checks).
    pub fn from_release(release_dir: impl AsRef<Path>, verify_hashes: bool) -> GResult<Self> {
        let dir = release_dir.as_ref();
        let cfg = check_release(dir, verify_hashes)?;
        let file = TensorFile::open(dir.join(WEIGHTS))?;
        let tensors = file.read_prefix(PREFIX, true)?;
        if tensors.is_empty() {
            return Err(GateError::ReleaseRefused(
                "the release carries no stage_one.* tensors".to_string(),
            ));
        }
        let meta = &cfg["arc"]["stage_one"];
        let candidates = meta
            .get("candidates")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(Candidate::from_json).collect::<Vec<_>>());
        let regimes = meta.get("regimes").and_then(Value::as_array).map(|a| {
            a.iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect::<Vec<_>>()
        });
        Self::new(tensors, candidates, regimes)
    }

    /// Display names of the candidates (`model` or `model@effort`).
    pub fn candidate_names(&self) -> Vec<String> {
        self.candidates
            .iter()
            .map(Candidate::display_name)
            .collect()
    }

    /// Index of the candidate whose name equals `query` (exact, else a unique substring).
    pub fn candidate_index(&self, query: &str) -> GResult<usize> {
        let names = self.candidate_names();
        if let Some(i) = names.iter().position(|n| n == query) {
            return Ok(i);
        }
        let hits: Vec<usize> = names
            .iter()
            .enumerate()
            .filter(|(_, n)| n.contains(query))
            .map(|(i, _)| i)
            .collect();
        if hits.len() != 1 {
            return key_err(format!(
                "candidate {query:?} matches {} of {} names",
                hits.len(),
                names.len()
            ));
        }
        Ok(hits[0])
    }

    /// Regime indexes for the given names.
    pub fn regime_index(&self, names: &[&str]) -> GResult<Vec<usize>> {
        let mut unknown: Vec<&str> = names
            .iter()
            .copied()
            .filter(|n| !self.regimes.iter().any(|r| r == n))
            .collect();
        unknown.sort_unstable();
        unknown.dedup();
        if !unknown.is_empty() {
            return value_err(format!(
                "regimes {unknown:?} are not this head's ({:?})",
                self.regimes
            ));
        }
        Ok(names
            .iter()
            .map(|n| self.regimes.iter().position(|r| r == n).unwrap_or(0))
            .collect())
    }
}
