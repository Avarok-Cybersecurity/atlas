# Harness aggregate — tier `wsnvfp4on` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_wsnvfp4on_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 1006.500 ± 118.176 | 929.000 | 1197.000 | [930.100, 1084.300] | 10/10 |
| cargo_toml_valid | rate | 10 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 10/10 |
| cargo_toml_present | rate | 10 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 10/10 |
| tool_calls_total | count | 10 | 16.600 ± 4.200 | 17.000 | 22.000 | [13.900, 19.100] | 10/10 |
| write_calls | count | 10 | 1.800 ± 0.748 | 2.000 | 3.000 | [1.400, 2.300] | 10/10 |
| drift_empty_path | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_outside_target | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_ws1_mask_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_b1_drift_fires | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| atlas_tier5c_retries | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| wall_time_s | count | 10 | 210.039 ± 100.834 | 234.079 | 357.708 | [148.577, 271.586] | 10/10 |
