// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn f(name: &str, size: u64) -> RemoteFile {
    RemoteFile {
        name: name.into(),
        size: Some(size),
    }
}

/// A realistic sharded NVFP4 repo, including the things that make an
/// unfiltered mirror cost double.
fn repo() -> Vec<RemoteFile> {
    vec![
        f("config.json", 1_200),
        f("tokenizer.json", 9_000_000),
        f("tokenizer_config.json", 4_000),
        f("generation_config.json", 200),
        f("model-00002-of-00002.safetensors", 8_000_000_000),
        f("model-00001-of-00002.safetensors", 9_000_000_000),
        f("model.safetensors.index.json", 60_000),
        f("README.md", 5_000),
        f(".gitattributes", 1_000),
        f("original/consolidated.pth", 30_000_000_000),
        f("pytorch_model.bin", 30_000_000_000),
        f("onnx/model.onnx", 15_000_000_000),
        f("eval_results.json", 2_000),
    ]
}

#[test]
fn only_what_the_loader_reads_is_downloaded() {
    let names: Vec<String> = select(&repo()).into_iter().map(|f| f.name).collect();
    for want in [
        "config.json",
        "tokenizer.json",
        "model-00001-of-00002.safetensors",
        "model.safetensors.index.json",
    ] {
        assert!(names.iter().any(|n| n == want), "missing {want}: {names:?}");
    }
    for skip in [
        "original/consolidated.pth",
        "pytorch_model.bin",
        "onnx/model.onnx",
        "README.md",
        ".gitattributes",
        "eval_results.json",
    ] {
        assert!(!names.iter().any(|n| n == skip), "should skip {skip}");
    }
}

#[test]
fn the_unfiltered_repo_really_is_much_larger() {
    // The number that justifies the filter existing at all.
    let all: u64 = repo().iter().filter_map(|f| f.size).sum();
    let planned = total_bytes(&select(&repo()));
    assert!(
        planned * 4 < all,
        "filter should save most of the bytes: {planned} vs {all}"
    );
}

#[test]
fn small_files_come_first_then_shards_ascending() {
    let plan = select(&repo());
    let first = &plan[0].name;
    assert!(
        !first.ends_with(".safetensors"),
        "a wrong-model abort must not cost a shard first: {first}"
    );
    let shards: Vec<&str> = plan
        .iter()
        .filter(|f| f.name.ends_with(".safetensors"))
        .map(|f| f.name.as_str())
        .collect();
    assert_eq!(
        shards,
        vec![
            "model-00002-of-00002.safetensors",
            "model-00001-of-00002.safetensors"
        ],
        "shards ascend by size, not by name"
    );
}

#[test]
fn a_gguf_only_repo_yields_no_weights() {
    // Common, and it must be refused BEFORE any bytes move — otherwise the
    // download "succeeds" having fetched a tokenizer and the load fails later
    // with something far less clear.
    let gguf = vec![
        f("config.json", 1_000),
        f("model-Q4_K_M.gguf", 20_000_000_000),
        f("tokenizer.json", 900_000),
    ];
    let plan = select(&gguf);
    assert!(!has_weights(&plan), "no safetensors means nothing to load");
    assert!(
        !plan.iter().any(|f| f.name.ends_with(".gguf")),
        "and the gguf is not downloaded either"
    );
}

#[test]
fn an_index_json_alone_is_not_weights() {
    // The index names the shards; it is not a shard. Treating it as weights
    // would let a metadata-only revision pass the pre-flight.
    let plan = select(&[f("model.safetensors.index.json", 60_000)]);
    assert!(!has_weights(&plan));
}

#[test]
fn nested_metadata_is_not_mistaken_for_the_models_own() {
    assert!(wanted("config.json"));
    assert!(!wanted("vision_tower/config.json"));
    assert!(!wanted("original/params.json"));
}

#[test]
fn unknown_sizes_do_not_corrupt_the_total() {
    // The plain model endpoint omits sizes. A missing size must read as "not
    // counted", never as zero-that-looks-complete or a panic.
    let plan = vec![
        RemoteFile {
            name: "model.safetensors".into(),
            size: None,
        },
        f("config.json", 1_000),
    ];
    assert_eq!(total_bytes(&plan), 1_000);
}

#[test]
fn selection_is_stable_regardless_of_listing_order() {
    let mut shuffled = repo();
    shuffled.reverse();
    assert_eq!(select(&repo()), select(&shuffled));
}

#[test]
fn a_name_that_climbs_out_of_the_snapshot_is_refused() {
    // `rfilename` is whatever the repo owner typed. The downloader joins it
    // onto the snapshot dir, and `Path::join` resolves `..` against the
    // parent, so before this check a published repo could write a shard
    // anywhere the server process could reach — including over another
    // model's weights. The extension filter alone did not stop it: every
    // name below ends in `.safetensors` and so passed `is_weight`.
    for evil in [
        "../../../../etc/cron.d/x.safetensors",
        "../model.safetensors",
        "subdir/../../escape.safetensors",
        "/etc/ld.so.preload.safetensors",
        "//rooted/model.safetensors",
        "./model.safetensors",
        "a//b/model.safetensors",
        "..\\windows\\model.safetensors",
        "C:/windows/model.safetensors",
    ] {
        assert!(!wanted(evil), "must refuse {evil:?}");
    }
}

#[test]
fn a_weight_in_an_ordinary_subdirectory_still_downloads() {
    // The containment check must not cost the legitimate layout: weights do
    // live in subdirectories, which is why `/` cannot simply be banned.
    assert!(wanted("model.safetensors"));
    assert!(wanted("weights/model-00001-of-00002.safetensors"));
    assert!(wanted("a/b/c/model.safetensors.index.json"));
    // A dotfile is not a traversal component.
    assert!(wanted(".hidden/model.safetensors"));
}

#[test]
fn a_traversing_name_is_dropped_from_the_plan_not_just_from_wanted() {
    // `select` is what the downloader actually calls; the guard has to hold
    // there, not only in the predicate it delegates to.
    let listing = vec![
        f("config.json", 1_000),
        f("model.safetensors", 10),
        f("../../escape.safetensors", 10),
    ];
    let names: Vec<String> = select(&listing).into_iter().map(|f| f.name).collect();
    assert!(!names.iter().any(|n| n.contains("..")), "{names:?}");
    assert_eq!(names.len(), 2);
}
