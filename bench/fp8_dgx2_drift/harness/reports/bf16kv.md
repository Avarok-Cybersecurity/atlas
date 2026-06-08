# Harness aggregate — tier `bf16kv` (N=10)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_bf16kv_*.json`.
Runs: [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 10 | 0.300 ± 0.458 | 0.000 | 1.000 | [0.000, 0.600] | 3/10 |
| cargo_toml_valid | rate | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| cargo_toml_present | rate | 10 | 0.300 ± 0.458 | 0.000 | 1.000 | [0.000, 0.600] | 3/10 |
| tool_calls_total | count | 10 | 4.100 ± 4.230 | 6.000 | 10.000 | [1.600, 6.800] | 5/10 |
| write_calls | count | 10 | 1.100 ± 1.640 | 0.000 | 5.000 | [0.200, 2.200] | 4/10 |
| drift_empty_path | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_path_outside_target | count | 10 | 0.500 ± 1.500 | 0.000 | 5.000 | [0.000, 1.500] | 1/10 |
| drift_path_literal_space | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_lean_prefix | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_bash_as_content | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_xml_attr_leak | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| drift_toml_newlines_collapsed | count | 10 | 1.100 ± 1.640 | 0.000 | 5.000 | [0.200, 2.200] | 4/10 |
| atlas_ws1_mask_fires | count | 10 | 8.200 ± 8.424 | 14.000 | 21.000 | [3.200, 13.400] | 5/10 |
| atlas_b1_drift_fires | count | 10 | 1.400 ± 1.685 | 1.000 | 5.000 | [0.400, 2.500] | 5/10 |
| atlas_tier5c_retries | count | 10 | 0.100 ± 0.300 | 0.000 | 1.000 | [0.000, 0.300] | 1/10 |
| atlas_a2_fuzzy_fires | count | 10 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/10 |
| atlas_tool_call_lines | count | 10 | 4.000 ± 4.195 | 5.000 | 10.000 | [1.500, 6.600] | 5/10 |
| wall_time_s | count | 10 | 126.572 ± 129.326 | 174.913 | 360.048 | [48.757, 208.889] | 10/10 |
