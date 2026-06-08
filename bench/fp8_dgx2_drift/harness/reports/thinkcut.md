# Harness aggregate — tier `thinkcut` (N=9)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_thinkcut_*.json`.
Runs: [1, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 9 | 0.889 ± 1.197 | 0.000 | 3.000 | [0.222, 1.778] | 4/9 |
| cargo_toml_valid | rate | 9 | 0.222 ± 0.416 | 0.000 | 1.000 | [0.000, 0.556] | 2/9 |
| cargo_toml_present | rate | 9 | 0.444 ± 0.497 | 0.000 | 1.000 | [0.111, 0.778] | 4/9 |
| tool_calls_total | count | 9 | 4.000 ± 1.633 | 4.000 | 7.000 | [3.000, 5.111] | 9/9 |
| write_calls | count | 9 | 0.667 ± 0.816 | 0.000 | 2.000 | [0.111, 1.222] | 4/9 |
| drift_empty_path | count | 9 | 0.444 ± 0.685 | 0.000 | 2.000 | [0.000, 0.889] | 3/9 |
| drift_path_outside_target | count | 9 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/9 |
| drift_path_literal_space | count | 9 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/9 |
| drift_lean_prefix | count | 9 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/9 |
| drift_bash_as_content | count | 9 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/9 |
| drift_xml_attr_leak | count | 9 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/9 |
| drift_toml_newlines_collapsed | count | 9 | 0.222 ± 0.416 | 0.000 | 1.000 | [0.000, 0.556] | 2/9 |
| atlas_ws1_mask_fires | count | 9 | 5.889 ± 4.122 | 4.000 | 17.000 | [3.889, 8.889] | 9/9 |
| atlas_b1_drift_fires | count | 9 | 0.556 ± 0.685 | 0.000 | 2.000 | [0.111, 1.000] | 4/9 |
| atlas_tier5c_retries | count | 9 | 0.111 ± 0.314 | 0.000 | 1.000 | [0.000, 0.333] | 1/9 |
| atlas_a2_fuzzy_fires | count | 9 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/9 |
| atlas_tool_call_lines | count | 9 | 3.889 ± 1.523 | 4.000 | 7.000 | [3.000, 4.889] | 9/9 |
| wall_time_s | count | 9 | 253.378 ± 94.040 | 248.274 | 360.083 | [192.059, 313.361] | 9/9 |
