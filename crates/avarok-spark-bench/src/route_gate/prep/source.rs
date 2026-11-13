// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. Local dataset readers for the routing gate; no upstream code.
//! Responsibility: read the local dataset files the corpus is built from (`.jsonl`, `.jsonl.gz`, `.json`, `.csv`;
//! parquet is not read here, convert it once with any parquet tool) and locate a source's files under a data dir.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use super::BUCKET_SOURCES;

/// Lower-case hex SHA-256 of a file.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
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

/// Parses RFC 4180 CSV text into rows of fields (quotes, doubled quotes, embedded commas and newlines).
pub fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    let mut any = false;
    while let Some(c) = chars.next() {
        any = true;
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' if field.is_empty() => quoted = true,
            ',' => row.push(std::mem::take(&mut field)),
            '\r' => {}
            '\n' => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
                any = false;
            }
            _ => field.push(c),
        }
    }
    if any {
        row.push(field);
        rows.push(row);
    }
    rows
}

/// `csv.DictReader` over `text`: the first row names the fields unless `headerless` gives them; blank rows are skipped.
fn dict_rows(text: &str, headerless: Option<&[&str]>) -> Vec<Value> {
    let mut rows = parse_csv(text)
        .into_iter()
        .filter(|r| !(r.len() == 1 && r[0].is_empty()));
    let names: Vec<String> = match headerless {
        Some(n) => n.iter().map(|s| (*s).to_string()).collect(),
        None => rows.next().unwrap_or_default(),
    };
    rows.map(|r| {
        let mut m = Map::new();
        for (k, v) in names.iter().zip(r) {
            m.insert(k.clone(), Value::String(v));
        }
        Value::Object(m)
    })
    .collect()
}

fn json_lines(text: &str) -> Result<Vec<Value>> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(Into::into))
        .collect()
}

/// Records of one `.jsonl` / `.jsonl.gz` / `.json` (list or `{"data": [...]}`) / `.csv` file.
pub fn read_records(path: &Path, headerless: Option<&[&str]>) -> Result<Vec<Value>> {
    let low = path.to_string_lossy().to_lowercase();
    if low.ends_with(".jsonl.gz") {
        let mut text = String::new();
        flate2::read::GzDecoder::new(std::fs::File::open(path)?).read_to_string(&mut text)?;
        return json_lines(&text);
    }
    if low.ends_with(".jsonl") {
        return json_lines(&std::fs::read_to_string(path)?);
    }
    if low.ends_with(".json") {
        let data: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let list = match data {
            Value::Object(mut o) if o.contains_key("data") => {
                o.remove("data").unwrap_or(Value::Null)
            }
            other => other,
        };
        return match list {
            Value::Array(a) => Ok(a),
            _ => bail!("{}: expected a list of records", path.display()),
        };
    }
    if low.ends_with(".csv") {
        return Ok(dict_rows(&std::fs::read_to_string(path)?, headerless));
    }
    if low.ends_with(".parquet") {
        bail!(
            "{}: parquet is not read here; convert it once to .jsonl with any parquet tool",
            path.display()
        );
    }
    bail!("unsupported file type: {}", path.display())
}

const EXTS: [&str; 5] = [".jsonl", ".jsonl.gz", ".json", ".csv", ".parquet"];

fn has_ext(name: &str) -> bool {
    let low = name.to_lowercase();
    EXTS.iter().any(|e| low.ends_with(e))
}

/// A source's records, files and combined sha256.
pub struct Loaded {
    pub records: Vec<Value>,
    pub files: Vec<PathBuf>,
    pub sha256: String,
}

/// Records for one source under `data_dir`, or `None` when no file is present.
pub fn load_source(data_dir: &Path, source: &str) -> Result<Option<Loaded>> {
    let sub = data_dir.join(source);
    let mut files: Vec<PathBuf> = Vec::new();
    if sub.is_dir() {
        for e in std::fs::read_dir(&sub)? {
            let p = e?.path();
            if has_ext(
                &p.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            ) {
                files.push(p);
            }
        }
        files.sort();
    } else {
        for ext in EXTS {
            let hit = data_dir.join(format!("{source}{ext}"));
            if hit.is_file() {
                files = vec![hit];
                break;
            }
        }
        if files.is_empty() {
            // "mmlu_*.jsonl" must not pick up the mmlu_pro file: skip names that belong to a longer source name
            let longer: Vec<&str> = BUCKET_SOURCES
                .iter()
                .copied()
                .filter(|s| *s != source && s.starts_with(&format!("{source}_")))
                .collect();
            if data_dir.is_dir() {
                for e in std::fs::read_dir(data_dir)? {
                    let p = e?.path();
                    let name = p
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    if name.starts_with(&format!("{source}_"))
                        && has_ext(&name)
                        && !longer.iter().any(|o| name.starts_with(o))
                    {
                        files.push(p);
                    }
                }
            }
            files.sort();
        }
    }
    if files.is_empty() {
        return Ok(None);
    }
    let headerless: Option<&[&str]> =
        if source == "mmlu" && files[0].to_string_lossy().to_lowercase().ends_with(".csv") {
            Some(&["question", "A", "B", "C", "D", "answer"])
        } else {
            None
        };
    let mut records = Vec::new();
    let mut digests = String::new();
    for p in &files {
        records.extend(read_records(p, headerless)?);
        digests.push_str(&sha256_file(p)?);
    }
    let sha256 = format!("{:x}", Sha256::digest(digests.as_bytes()));
    Ok(Some(Loaded {
        records,
        files,
        sha256,
    }))
}
