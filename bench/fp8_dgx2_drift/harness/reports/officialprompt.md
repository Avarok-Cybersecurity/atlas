# Harness aggregate — tier `officialprompt` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_officialprompt_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 0.600 ± 0.917 | 0.000 | 3.000 | [0.100, 1.200] | 4/10 |
| cargo_toml_valid | rate | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| cargo_toml_present | rate | 10 | 0.400 ± 0.490 | 0.000 | 1.000 | [0.100, 0.700] | 4/10 |
| tool_calls_total | count | 10 | 4.200 ± 2.676 | 4.000 | 10.000 | [2.600, 5.900] | 9/10 |
| write_calls | count | 10 | 0.900 ± 1.221 | 0.000 | 3.000 | [0.200, 1.700] | 4/10 |
| drift_empty_path | count | 10 | 0.500 ± 0.806 | 0.000 | 2.000 | [0.000, 1.000] | 3/10 |
| drift_path_outside_target | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.300 ± 0.458 | 0.000 | 1.000 | [0.000, 0.600] | 3/10 |
| atlas_ws1_mask_fires | count | 10 | 4.700 ± 3.523 | 4.000 | 11.000 | [2.600, 6.900] | 8/10 |
| atlas_b1_drift_fires | count | 10 | 1.300 ± 1.552 | 1.000 | 4.000 | [0.400, 2.300] | 5/10 |
| atlas_tier5c_retries | count | 10 | 0.300 ± 0.458 | 0.000 | 1.000 | [0.000, 0.600] | 3/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 3.900 ± 2.468 | 4.000 | 9.000 | [2.400, 5.500] | 9/10 |
| wall_time_s | count | 10 | 272.622 ± 112.776 | 358.468 | 360.082 | [198.050, 340.528] | 10/10 |
