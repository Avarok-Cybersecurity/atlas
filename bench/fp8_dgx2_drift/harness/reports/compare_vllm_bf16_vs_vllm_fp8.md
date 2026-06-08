# Tier comparison — `vllm_bf16` (n=10) vs `vllm_fp8` (n=10)

alpha=0.05 (Bonferroni-adjusted across 18 metrics).
Significant rows are starred (*).

| metric | median(vllm_bf16) | median(vllm_fp8) | Δ | p | p_bonf | sig |
|---|---|---|---|---|---|---|
| files_written | 1159.500 | 1119.500 | -40.000 | 0.3443 | 1.0000 |  |
| cargo_toml_valid | 1.000 | 1.000 | +0.000 | 1.0000 | 1.0000 |  |
| cargo_toml_present | 1.000 | 1.000 | +0.000 | 1.0000 | 1.0000 |  |
| tool_calls_total | 16.000 | 15.500 | -0.500 | 0.9396 | 1.0000 |  |
| write_calls | 2.000 | 2.000 | +0.000 | 0.3566 | 1.0000 |  |
| drift_empty_path | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_path_outside_target | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_path_literal_space | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_lean_prefix | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_bash_as_content | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_xml_attr_leak | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_toml_newlines_collapsed | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_ws1_mask_fires | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_b1_drift_fires | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_tier5c_retries | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_a2_fuzzy_fires | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_tool_call_lines | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| wall_time_s | 254.567 | 214.753 | -39.814 | 0.3075 | 1.0000 |  |
