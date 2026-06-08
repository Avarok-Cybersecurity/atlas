# Harness aggregate — tier `bf16moe` (N=6)

Generated from `bench/fp8_dgx2_drift/harness/runs/run_bf16moe_*.json`.
Runs: [1, 2, 3, 4, 5, 6]

| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |
|---|---|---|---|---|---|---|---|
| files_written | count | 6 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/6 |
| cargo_toml_valid | rate | 6 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/6 |
| cargo_toml_present | rate | 6 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/6 |
| tool_calls_total | count | 6 | 1.667 ± 1.795 | 2.000 | 4.000 | [0.333, 3.000] | 3/6 |
| write_calls | count | 6 | 0.167 ± 0.373 | 0.000 | 1.000 | [0.000, 0.500] | 1/6 |
| drift_empty_path | count | 6 | 0.167 ± 0.373 | 0.000 | 1.000 | [0.000, 0.500] | 1/6 |
| drift_path_outside_target | count | 6 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/6 |
| drift_path_literal_space | count | 6 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/6 |
| drift_lean_prefix | count | 6 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/6 |
| drift_bash_as_content | count | 6 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/6 |
| drift_xml_attr_leak | count | 6 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/6 |
| drift_toml_newlines_collapsed | count | 6 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/6 |
| atlas_ws1_mask_fires | count | 6 | 1.667 ± 1.700 | 3.000 | 4.000 | [0.500, 3.000] | 3/6 |
| atlas_b1_drift_fires | count | 6 | 2.000 ± 2.887 | 0.000 | 7.000 | [0.000, 4.333] | 2/6 |
| atlas_tier5c_retries | count | 6 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/6 |
| atlas_a2_fuzzy_fires | count | 6 | 0.000 ± 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 0/6 |
| atlas_tool_call_lines | count | 6 | 1.667 ± 1.795 | 2.000 | 4.000 | [0.333, 3.000] | 3/6 |
| wall_time_s | count | 6 | 336.549 ± 23.523 | 360.066 | 360.075 | [320.848, 352.241] | 6/6 |
