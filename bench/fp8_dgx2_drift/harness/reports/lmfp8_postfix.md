# Harness aggregate — tier `lmfp8_postfix` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_lmfp8_postfix_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 854.000 ± 295.845 | 904.000 | 1170.000 | [647.200, 993.600] | 10/10 |
| cargo_toml_valid | rate | 10 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 10/10 |
| cargo_toml_present | rate | 10 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 10/10 |
| webserver_ok | rate | 10 | 1.000 ± 0.000 | 1.000 | 1.000 | [1.000, 1.000] | 10/10 |
| followed_directions | rate | 10 | 0.900 ± 0.300 | 1.000 | 1.000 | [0.700, 1.000] | 9/10 |
| fd_steps_completed | count | 10 | 5.600 ± 1.200 | 6.000 | 6.000 | [4.800, 6.000] | 10/10 |
| tool_calls_total | count | 10 | 13.900 ± 4.369 | 13.000 | 21.000 | [11.300, 16.600] | 10/10 |
| write_calls | count | 10 | 1.800 ± 0.400 | 2.000 | 2.000 | [1.500, 2.000] | 10/10 |
| drift_empty_path | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_outside_target | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_ws1_mask_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_b1_drift_fires | count | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| atlas_tier5c_retries | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| wall_time_s | count | 10 | 203.774 ± 103.029 | 226.661 | 360.044 | [141.955, 266.997] | 10/10 |
