# Harness aggregate — tier `mtpworks` (N=9)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_mtpworks_*.json`.
Runs: [1, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 9 | 1.556 ± 1.066 | 1.000 | 3.000 | [0.889, 2.333] | 8/9 |
| cargo_toml_valid | rate | 9 | 0.778 ± 0.416 | 1.000 | 1.000 | [0.444, 1.000] | 7/9 |
| cargo_toml_present | rate | 9 | 0.889 ± 0.314 | 1.000 | 1.000 | [0.667, 1.000] | 8/9 |
| tool_calls_total | count | 9 | 10.333 ± 4.807 | 11.000 | 16.000 | [6.889, 13.222] | 8/9 |
| write_calls | count | 9 | 3.111 ± 1.449 | 4.000 | 5.000 | [2.111, 4.000] | 8/9 |
| drift_empty_path | count | 9 | 0.333 ± 0.471 | 0.000 | 1.000 | [0.111, 0.667] | 3/9 |
| drift_path_outside_target | count | 9 | 1.667 ± 1.054 | 2.000 | 3.000 | [1.000, 2.333] | 7/9 |
| drift_path_literal_space | count | 9 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/9 |
| drift_lean_prefix | count | 9 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/9 |
| drift_bash_as_content | count | 9 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/9 |
| drift_xml_attr_leak | count | 9 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/9 |
| drift_toml_newlines_collapsed | count | 9 | 0.222 ± 0.416 | 0.000 | 1.000 | [0.000, 0.556] | 2/9 |
| atlas_ws1_mask_fires | count | 9 | 2.556 ± 1.571 | 3.000 | 5.000 | [1.556, 3.556] | 8/9 |
| atlas_b1_drift_fires | count | 9 | 0.333 ± 0.471 | 0.000 | 1.000 | [0.000, 0.667] | 3/9 |
| atlas_tier5c_retries | count | 9 | 0.111 ± 0.314 | 0.000 | 1.000 | [0.000, 0.333] | 1/9 |
| atlas_a2_fuzzy_fires | count | 9 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/9 |
| atlas_tool_call_lines | count | 9 | 10.778 ± 5.692 | 11.000 | 21.000 | [7.000, 14.444] | 8/9 |
| wall_time_s | count | 9 | 262.697 ± 104.785 | 283.045 | 360.073 | [187.735, 324.862] | 9/9 |
