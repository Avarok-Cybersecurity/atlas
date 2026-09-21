// SPDX-License-Identifier: AGPL-3.0-only
//! Bounded, duplicate-rejecting index input at the checkpoint I/O boundary.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde::de::{MapAccess, Visitor};
use std::collections::HashMap;
use std::fmt;
use std::io::Read;
use std::path::Path;

const MAX_INDEX_BYTES: u64 = 128 * 1024 * 1024;
#[derive(Deserialize)]
struct Index {
    #[serde(deserialize_with = "unique_map")]
    weight_map: HashMap<String, String>,
}
fn unique_map<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<HashMap<String, String>, D::Error> {
    struct Unique;
    impl<'de> Visitor<'de> for Unique {
        type Value = HashMap<String, String>;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("unique tensor-to-shard entries")
        }
        fn visit_map<M: MapAccess<'de>>(
            self,
            mut map: M,
        ) -> std::result::Result<Self::Value, M::Error> {
            let mut result = HashMap::new();
            while let Some((key, value)) = map.next_entry::<String, String>()? {
                if result.insert(key.clone(), value).is_some() {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate tensor index key {key}"
                    )));
                }
            }
            Ok(result)
        }
    }
    d.deserialize_map(Unique)
}

pub(super) fn read(root: &Path) -> Result<Option<HashMap<String, String>>> {
    let path = root.join("model.safetensors.index.json");
    if !path.exists() {
        return Ok(None);
    }
    let path = path.canonicalize()?;
    ensure!(path.starts_with(root), "K3 index escapes checkpoint root");
    let file = std::fs::File::open(path)?;
    ensure!(
        file.metadata()?.len() <= MAX_INDEX_BYTES,
        "K3 index exceeds 128 MiB admission limit"
    );
    let mut bytes = Vec::new();
    file.take(MAX_INDEX_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_INDEX_BYTES,
        "K3 index grew beyond admission limit"
    );
    let index: Index = serde_json::from_slice(&bytes).context("invalid K3 safetensors index")?;
    Ok(Some(index.weight_map))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duplicate_tensor_keys_are_not_silently_replaced() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"x":"a.safetensors","x":"b.safetensors"}}"#,
        )
        .unwrap();
        assert!(read(dir.path()).is_err());
    }
    #[test]
    fn oversized_index_is_refused_before_reading() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::File::create(dir.path().join("model.safetensors.index.json"))
            .unwrap()
            .set_len(MAX_INDEX_BYTES + 1)
            .unwrap();
        assert!(read(dir.path()).is_err());
    }
}
