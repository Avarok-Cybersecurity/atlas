# Harness aggregate — tier `lmbf16` (N=7)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_lmbf16_*.json`.
Runs: [1, 2, 3, 4, 5, 6, 7]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 7 | 3.143 ± 5.743 | 1.000 | 17.000 | [0.286, 7.857] | 4/7 |
| cargo_toml_valid | rate | 7 | 0.286 ± 0.452 | 0.000 | 1.000 | [0.000, 0.714] | 2/7 |
| cargo_toml_present | rate | 7 | 0.429 ± 0.495 | 0.000 | 1.000 | [0.143, 0.857] | 3/7 |
| tool_calls_total | count | 7 | 5.000 ± 2.268 | 4.000 | 9.000 | [3.429, 6.714] | 7/7 |
| write_calls | count | 7 | 1.143 ± 0.990 | 1.000 | 3.000 | [0.429, 1.857] | 5/7 |
| drift_empty_path | count | 7 | 0.571 ± 0.728 | 0.000 | 2.000 | [0.143, 1.143] | 3/7 |
| drift_path_outside_target | count | 7 | 0.143 ± 0.350 | 0.000 | 1.000 | [0.000, 0.429] | 1/7 |
| drift_path_literal_space | count | 7 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/7 |
| drift_lean_prefix | count | 7 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/7 |
| drift_bash_as_content | count | 7 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/7 |
| drift_xml_attr_leak | count | 7 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/7 |
| drift_toml_newlines_collapsed | count | 7 | 0.429 ± 0.728 | 0.000 | 2.000 | [0.000, 1.000] | 2/7 |
| atlas_ws1_mask_fires | count | 7 | 7.429 ± 4.271 | 8.000 | 14.000 | [4.286, 10.714] | 7/7 |
| atlas_b1_drift_fires | count | 7 | 1.714 ± 1.829 | 1.000 | 6.000 | [0.714, 3.286] | 6/7 |
| atlas_tier5c_retries | count | 7 | 0.286 ± 0.452 | 0.000 | 1.000 | [0.000, 0.571] | 2/7 |
| atlas_a2_fuzzy_fires | count | 7 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/7 |
| atlas_tool_call_lines | count | 7 | 4.714 ± 2.119 | 4.000 | 8.000 | [3.143, 6.286] | 7/7 |
| wall_time_s | count | 7 | 315.461 ± 79.008 | 360.068 | 360.075 | [251.822, 360.072] | 7/7 |
