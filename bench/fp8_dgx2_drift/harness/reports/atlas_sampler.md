# Harness aggregate — tier `atlas_sampler` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_atlas_sampler_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 5.400 ± 12.571 | 2.000 | 43.000 | [0.700, 14.000] | 7/10 |
| cargo_toml_valid | rate | 10 | 0.400 ± 0.490 | 0.000 | 1.000 | [0.100, 0.700] | 4/10 |
| cargo_toml_present | rate | 10 | 0.700 ± 0.458 | 1.000 | 1.000 | [0.400, 1.000] | 7/10 |
| tool_calls_total | count | 10 | 7.600 ± 5.553 | 7.000 | 21.000 | [4.500, 11.300] | 10/10 |
| write_calls | count | 10 | 2.000 ± 1.732 | 2.000 | 6.000 | [1.000, 3.100] | 7/10 |
| drift_empty_path | count | 10 | 0.200 ± 0.400 | 0.000 | 1.000 | [0.000, 0.500] | 2/10 |
| drift_path_outside_target | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 0.900 ± 1.044 | 1.000 | 3.000 | [0.300, 1.600] | 5/10 |
| atlas_ws1_mask_fires | count | 10 | 1.700 ± 1.487 | 2.000 | 5.000 | [0.800, 2.700] | 7/10 |
| atlas_b1_drift_fires | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| atlas_tier5c_retries | count | 10 | 0.300 ± 0.458 | 0.000 | 1.000 | [0.000, 0.600] | 3/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 7.200 ± 5.653 | 6.000 | 21.000 | [4.100, 11.100] | 10/10 |
| wall_time_s | count | 10 | 284.552 ± 115.516 | 360.075 | 360.110 | [207.923, 360.074] | 10/10 |
