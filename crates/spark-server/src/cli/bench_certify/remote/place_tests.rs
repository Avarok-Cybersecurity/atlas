// SPDX-License-Identifier: AGPL-3.0-only
//! Placement is exercised on a COMMITTED record and its real signature,
//! copied into a scratch repository that carries the committed signer keys —
//! so "verifies" here is the same verification CI performs.
use super::*;
use std::path::Path;

struct Scratch {
    root: PathBuf,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

/// A scratch root with the signer registry, a fetched copy of the newest
/// committed decode-floor record, and its git sha.
fn fixture(tag: &str) -> (Scratch, Vec<FetchedFile>, String) {
    let ws = workspace();
    let root = std::env::temp_dir().join(format!("certify-place-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join(".github/record-signers")).unwrap();
    for e in std::fs::read_dir(ws.join(".github/record-signers"))
        .unwrap()
        .flatten()
    {
        std::fs::copy(
            e.path(),
            root.join(".github/record-signers").join(e.file_name()),
        )
        .unwrap();
    }
    let dir = ws.join(".benchmarks/decode-floor");
    let newest = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .max()
        .unwrap();
    let name = newest.file_name().unwrap().to_string_lossy().into_owned();
    let fetched = root.join("fetched");
    std::fs::create_dir_all(fetched.join(".benchmarks/decode-floor")).unwrap();
    std::fs::create_dir_all(fetched.join(".certify/x")).unwrap();
    let rec_to = fetched.join(".benchmarks/decode-floor").join(&name);
    let sig_to = fetched
        .join(".benchmarks/decode-floor")
        .join(format!("{name}.sig"));
    std::fs::copy(&newest, &rec_to).unwrap();
    std::fs::copy(format!("{}.sig", newest.display()), &sig_to).unwrap();
    std::fs::write(fetched.join(".certify/x/decode-floor.log"), "log").unwrap();
    let sha = avarok_plugin::gate::read_record(&newest).unwrap().git_sha;
    let f = |name: &str, rel: &str, path: PathBuf| FetchedFile {
        name: name.into(),
        relative_path: rel.into(),
        path,
        bytes: 0,
        sha256: String::new(),
    };
    let files = vec![
        f(&name, &format!(".benchmarks/decode-floor/{name}"), rec_to),
        f(
            &format!("{name}.sig"),
            &format!(".benchmarks/decode-floor/{name}.sig"),
            sig_to,
        ),
        f(
            "decode-floor.log",
            ".certify/x/decode-floor.log",
            fetched.join(".certify/x/decode-floor.log"),
        ),
    ];
    (Scratch { root }, files, sha)
}

fn expect<'a>(anchor: &'a str) -> Expect<'a> {
    Expect {
        unit_id: "decode-floor",
        shard: None,
        log_stem: "decode-floor",
        anchor,
        hardware: "gb10",
    }
}

#[test]
fn a_genuine_record_is_placed_with_its_signature_and_log() {
    let (s, files, sha) = fixture("ok");
    let log_dir = s.root.join("logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let placed = place(&s.root, &log_dir, &files, &expect(&sha)).expect("placed");
    assert!(
        placed
            .record
            .starts_with(s.root.join(".benchmarks/decode-floor"))
    );
    assert!(placed.record.exists() && placed.signature.exists());
    assert_eq!(placed.log.as_ref().map(|l| l.exists()), Some(true));
    // A second placement of the same record is refused: never overwrite.
    let e = place(&s.root, &log_dir, &files, &expect(&sha)).unwrap_err();
    assert!(format!("{e:#}").contains("already exists"), "{e:#}");
}

#[test]
fn a_tampered_signature_leaves_nothing_behind() {
    let (s, files, sha) = fixture("tamper");
    let sig = &files[1].path;
    let mut text = std::fs::read_to_string(sig).unwrap();
    // Flip one character inside the base64 signature.
    let i = text.find("\"sig\":\"").unwrap() + 8;
    let c = text.as_bytes()[i];
    let flipped = if c == b'A' { 'B' } else { 'A' };
    text.replace_range(i..=i, &flipped.to_string());
    std::fs::write(sig, text).unwrap();
    let e = place(&s.root, &s.root, &files, &expect(&sha)).unwrap_err();
    assert!(format!("{e:#}").contains("does not verify"), "{e:#}");
    // NEGATIVE CONTROL on the cleanup: the record dir holds nothing.
    let left = std::fs::read_dir(s.root.join(".benchmarks/decode-floor"))
        .map(|d| d.count())
        .unwrap_or(0);
    assert_eq!(left, 0);
}

#[test]
fn the_record_must_be_for_this_unit_at_the_anchor_on_this_class() {
    let (s, files, sha) = fixture("expect");
    let mut wrong_unit = expect(&sha);
    wrong_unit.unit_id = "ttft-warm-gate";
    // sort_files refuses first: the paths do not belong to that unit.
    let e = place(&s.root, &s.root, &files, &wrong_unit).unwrap_err();
    assert!(
        format!("{e:#}").contains("not a record, signature or log of ttft-warm-gate"),
        "{e:#}"
    );
    let mut wrong_anchor = expect(&sha);
    wrong_anchor.anchor = "0000000000000000000000000000000000000000";
    let e = place(&s.root, &s.root, &files, &wrong_anchor).unwrap_err();
    assert!(format!("{e:#}").contains("not the anchor"), "{e:#}");
    let mut wrong_class = expect(&sha);
    wrong_class.hardware = "h100";
    let e = place(&s.root, &s.root, &files, &wrong_class).unwrap_err();
    assert!(format!("{e:#}").contains("measured on class gb10"), "{e:#}");
    assert!(
        !s.root.join(".benchmarks").exists(),
        "nothing placed on refusal"
    );
}

#[test]
fn sort_files_wants_exactly_one_record_and_its_own_signature() {
    let f = |rel: &str| FetchedFile {
        name: Path::new(rel)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
        relative_path: rel.into(),
        path: PathBuf::from(rel),
        bytes: 0,
        sha256: String::new(),
    };
    let ok = [f(".benchmarks/g/r.json"), f(".benchmarks/g/r.json.sig")];
    assert!(sort_files(&ok, "g").is_ok());
    assert!(
        sort_files(&[f(".benchmarks/g/r.json")], "g")
            .unwrap_err()
            .to_string()
            .contains("no signature")
    );
    assert!(
        sort_files(&[f(".benchmarks/g/r.json.sig")], "g")
            .unwrap_err()
            .to_string()
            .contains("no record")
    );
    let two = [
        f(".benchmarks/g/a.json"),
        f(".benchmarks/g/b.json"),
        f(".benchmarks/g/a.json.sig"),
    ];
    assert!(
        sort_files(&two, "g")
            .unwrap_err()
            .to_string()
            .contains("two records")
    );
    let mismatch = [f(".benchmarks/g/a.json"), f(".benchmarks/g/b.json.sig")];
    assert!(
        sort_files(&mismatch, "g")
            .unwrap_err()
            .to_string()
            .contains("does not belong")
    );
    let foreign = [
        f(".benchmarks/g/r.json"),
        f(".benchmarks/g/r.json.sig"),
        f("Cargo.toml"),
    ];
    assert!(
        sort_files(&foreign, "g")
            .unwrap_err()
            .to_string()
            .contains("Cargo.toml")
    );
}
