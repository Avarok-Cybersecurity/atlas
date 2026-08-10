// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn map(y: &Yaml) -> &BTreeMap<String, Yaml> {
    y.as_map().expect("a mapping")
}

fn scalar<'a>(y: &'a Yaml, key: &str) -> &'a str {
    map(y)
        .get(key)
        .expect("key present")
        .as_str()
        .expect("scalar")
}

#[test]
fn scalars_nested_maps_and_quotes() {
    let y = parse(
        r#"
recipe_version: "2"
model: Qwen/Qwen3.6-27B
max_nodes: 1

metadata:
  maintainer: avarok
  category: agent
"#,
    )
    .expect("parses");
    assert_eq!(scalar(&y, "recipe_version"), "2", "quotes are stripped");
    assert_eq!(scalar(&y, "model"), "Qwen/Qwen3.6-27B");
    assert_eq!(scalar(&y, "max_nodes"), "1");
    let meta = map(&y)["metadata"].clone();
    assert_eq!(scalar(&meta, "maintainer"), "avarok");
}

#[test]
fn a_literal_block_keeps_its_newlines_and_its_hashes() {
    // The `command:` blocks contain `#`; stripping comments inside one would
    // silently truncate the command that actually gets run.
    let y = parse("description: |\n  line one\n  # not a comment\n  line three\nmodel: m\n")
        .expect("parses");
    assert_eq!(
        scalar(&y, "description"),
        "line one\n# not a comment\nline three"
    );
    assert_eq!(scalar(&y, "model"), "m", "the block ended at the dedent");
}

#[test]
fn comments_and_blank_lines_are_ignored_outside_blocks() {
    let y = parse("# header\n\nmodel: m   # trailing\n\nport: 8888\n").expect("parses");
    assert_eq!(scalar(&y, "model"), "m");
    assert_eq!(scalar(&y, "port"), "8888");
}

#[test]
fn a_hash_inside_quotes_is_not_a_comment() {
    let y = parse("tag: \"a # b\"\n").expect("parses");
    assert_eq!(scalar(&y, "tag"), "a # b");
}

#[test]
fn sequences_and_empty_flow_maps() {
    let y = parse("mods:\n  - mods/diffusiongemma\n  - mods/other\nenv: {}\n").expect("parses");
    assert_eq!(
        map(&y)["mods"],
        Yaml::List(vec![
            Yaml::Scalar("mods/diffusiongemma".into()),
            Yaml::Scalar("mods/other".into()),
        ])
    );
    assert_eq!(map(&y)["env"], Yaml::Map(BTreeMap::new()));
}

#[test]
fn an_unreadable_construct_is_an_error_not_a_skip() {
    // The whole point: a shape this reader does not model must stop the parse
    // rather than quietly dropping a key that changes the serve config.
    assert!(parse("model: m\n\tport: 8888\n").is_err(), "stray indent");
    assert!(parse("just a bare line\n").is_err(), "no key");
    assert!(
        parse("model:\n").is_err(),
        "key with neither value nor block"
    );
    assert!(parse(": value\n").is_err(), "empty key");
}

#[test]
fn the_document_must_be_a_mapping() {
    assert!(parse("- one\n- two\n").is_err());
}

/// The census test: every real recipe must parse, and expose the blocks the
/// rest of the feature reads. Fixtures are vendored, so this never hits the
/// network and cannot drift when the upstream repo moves.
#[test]
fn all_vendored_recipes_parse() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recipes");
    let mut count = 0;
    let mut stack = vec![dir.clone()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("fixtures dir exists") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "yaml") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("read");
            let y = parse(&text).unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
            let m = map(&y);
            for required in ["recipe_version", "model", "container", "defaults"] {
                assert!(
                    m.contains_key(required),
                    "{}: missing {required}",
                    path.display()
                );
            }
            assert!(
                m["defaults"].as_map().is_some(),
                "{}: defaults must be a mapping",
                path.display()
            );
            count += 1;
        }
    }
    assert_eq!(count, 25, "the vendored corpus is 25 recipes");
}
