# Harness aggregate — tier `ssmfix1200` (N=3)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_ssmfix1200_*.json`.
Runs: [1, 2, 3]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 3 | 1.333 ± 1.247 | 1.000 | 3.000 | [0.000, 3.000] | 2/3 |
| cargo_toml_valid | rate | 3 | 0.333 ± 0.471 | 0.000 | 1.000 | [0.000, 1.000] | 1/3 |
| cargo_toml_present | rate | 3 | 0.667 ± 0.471 | 1.000 | 1.000 | [0.000, 1.000] | 2/3 |
| tool_calls_total | count | 3 | 7.333 ± 3.682 | 7.000 | 12.000 | [3.000, 12.000] | 3/3 |
| write_calls | count | 3 | 2.333 ± 1.247 | 2.000 | 4.000 | [1.000, 4.000] | 3/3 |
| drift_empty_path | count | 3 | 1.667 ± 0.943 | 1.000 | 3.000 | [1.000, 3.000] | 3/3 |
| drift_path_outside_target | count | 3 | 0.333 ± 0.471 | 0.000 | 1.000 | [0.000, 1.000] | 1/3 |
| drift_path_literal_space | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| drift_lean_prefix | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| drift_bash_as_content | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| drift_xml_attr_leak | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| drift_toml_newlines_collapsed | count | 3 | 0.667 ± 0.471 | 1.000 | 1.000 | [0.000, 1.000] | 2/3 |
| atlas_ws1_mask_fires | count | 3 | 9.333 ± 3.859 | 11.000 | 13.000 | [4.000, 13.000] | 3/3 |
| atlas_b1_drift_fires | count | 3 | 3.000 ± 2.828 | 1.000 | 7.000 | [1.000, 7.000] | 3/3 |
| atlas_tier5c_retries | count | 3 | 0.333 ± 0.471 | 0.000 | 1.000 | [0.000, 1.000] | 1/3 |
| atlas_a2_fuzzy_fires | count | 3 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/3 |
| atlas_tool_call_lines | count | 3 | 7.000 ± 3.266 | 7.000 | 11.000 | [3.000, 11.000] | 3/3 |
| wall_time_s | count | 3 | 536.447 ± 300.818 | 635.867 | 844.959 | [128.514, 844.959] | 3/3 |
