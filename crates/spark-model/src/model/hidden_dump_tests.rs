// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the prefill hidden-state dump writer (`hidden_dump.rs`): header plus payload
//! round trip, the env parser, and the dtype widening. CPU only.

use super::*;

/// A scratch directory removed on drop, unique per process and tag.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "avarok_hidden_dump_test_{}_{tag}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn site() -> DumpSite {
    DumpSite {
        prefill_path: "prefill_chunk",
        token_count: 5,
        proc_count: 5,
        seq_len_start: 0,
    }
}

fn meta(kind: &'static str, layer: Option<usize>, rows: usize, cols: usize) -> DumpMeta {
    DumpMeta {
        kind,
        layer,
        device_dtype: "bf16",
        rows,
        cols,
        position_start: 4,
        positions: if rows == 1 { "last" } else { "all" },
        site: site(),
    }
}

#[test]
fn header_and_payload_round_trip_bit_exact() {
    let dir = Scratch::new("roundtrip");
    let values = [
        1.5f32,
        -0.0,
        f32::MIN_POSITIVE,
        1.0e30,
        -2.25,
        f32::from_bits(0x7fc0_0001),
    ];
    let m = meta("layer", Some(11), 2, 3);
    let json_path = write_dump(&dir.0, &layer_stem(11), &m, &values).unwrap();
    assert_eq!(json_path, dir.0.join("hidden_L11.json"));

    let (header, back) = read_dump(&json_path).unwrap();
    assert_eq!(header["version"], FORMAT_VERSION);
    assert_eq!(header["name"], "hidden_L11");
    assert_eq!(header["kind"], "layer");
    assert_eq!(header["layer"], 11);
    assert_eq!(header["bin"], "hidden_L11.bin");
    assert_eq!(header["dtype"], "f32");
    assert_eq!(header["endian"], "little");
    assert_eq!(header["device_dtype"], "bf16");
    assert_eq!(header["shape"], serde_json::json!([2, 3]));
    assert_eq!(header["bytes"], 24);
    assert_eq!(header["positions"], "all");
    assert_eq!(header["position_start"], 4);
    assert_eq!(header["token_count"], 5);
    assert_eq!(header["proc_count"], 5);
    assert_eq!(header["seq_len_start"], 0);
    assert_eq!(header["prefill_path"], "prefill_chunk");

    assert_eq!(back.len(), values.len());
    for (a, b) in values.iter().zip(&back) {
        assert_eq!(a.to_bits(), b.to_bits(), "payload must survive bit for bit");
    }
    // Little endian on disk, whatever the host: 1.5f32 is 0x3fc00000.
    let raw = std::fs::read(dir.0.join("hidden_L11.bin")).unwrap();
    assert_eq!(&raw[..4], &[0x00, 0x00, 0xc0, 0x3f]);
}

#[test]
fn non_layer_dumps_have_a_null_layer_and_leave_no_tmp_files() {
    let dir = Scratch::new("final");
    let m = meta("final_norm", None, 1, 4);
    let json_path = write_dump(&dir.0, "final_norm", &m, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let (header, back) = read_dump(&json_path).unwrap();
    assert!(header["layer"].is_null());
    assert_eq!(header["kind"], "final_norm");
    assert_eq!(header["positions"], "last");
    assert_eq!(back, vec![1.0, 2.0, 3.0, 4.0]);
    let mut names: Vec<String> = std::fs::read_dir(&dir.0)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, vec!["final_norm.bin", "final_norm.json"]);
}

#[test]
fn a_second_dump_replaces_the_first() {
    let dir = Scratch::new("overwrite");
    let m = meta("logits", None, 1, 2);
    write_dump(&dir.0, "logits", &m, &[1.0, 2.0]).unwrap();
    let json_path = write_dump(&dir.0, "logits", &m, &[3.0, 4.0]).unwrap();
    let (_, back) = read_dump(&json_path).unwrap();
    assert_eq!(back, vec![3.0, 4.0]);
}

#[test]
fn a_shape_that_does_not_match_the_payload_is_refused_and_writes_nothing() {
    let dir = Scratch::new("badshape");
    let m = meta("layer", Some(0), 2, 3);
    let err = write_dump(&dir.0, "hidden_L0", &m, &[0.0; 5]).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert!(!dir.0.join("hidden_L0.bin").exists());
    assert!(!dir.0.join("hidden_L0.json").exists());
}

#[test]
fn bf16_widens_exactly_and_f32_reads_back() {
    // bf16 is the top half of an f32: 0x3fc0 is 1.5, 0xc020 is -2.5, 0x0000 is 0.
    let bytes = [0xc0, 0x3f, 0x20, 0xc0, 0x00, 0x00];
    assert_eq!(bf16_le_to_f32(&bytes), vec![1.5, -2.5, 0.0]);
    let f = [0x00, 0x00, 0xc0, 0x3f];
    assert_eq!(f32_le_to_f32(&f), vec![1.5]);
}

#[test]
fn plan_is_off_without_a_directory() {
    assert_eq!(HiddenDumpPlan::parse("", Some("11"), None), None);
    assert_eq!(HiddenDumpPlan::parse("   ", None, None), None);
}

#[test]
fn plan_parses_layers_all_positions_and_reports_junk() {
    let p = HiddenDumpPlan::parse("/tmp/d", Some(" 23, 11 ,11,x,,-1"), Some("1")).unwrap();
    assert_eq!(p.dir, PathBuf::from("/tmp/d"));
    assert!(p.wants_layer(11) && p.wants_layer(23));
    assert!(!p.wants_layer(0) && !p.wants_layer(12));
    assert!(p.all_positions);
    assert_eq!(p.ignored, vec!["x".to_string(), "-1".to_string()]);

    let none = HiddenDumpPlan::parse("/tmp/d", None, Some("0")).unwrap();
    assert!(!none.wants_layer(0), "no layer list means no layer dump");
    assert!(!none.all_positions);

    let all = HiddenDumpPlan::parse("/tmp/d", Some("all"), None).unwrap();
    assert!(all.wants_layer(0) && all.wants_layer(999));
}
