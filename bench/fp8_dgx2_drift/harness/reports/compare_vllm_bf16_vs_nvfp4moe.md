# Tier comparison — `vllm_bf16` (n=10) vs `nvfp4moe` (n=10)

alpha=0.05 (Bonferroni-adjusted across 18 metrics).
Significant rows are starred (*).

| metric | median(vllm_bf16) | median(nvfp4moe) | Δ | p | p_bonf | sig |
|---|---|---|---|---|---|---|
| atlas_tool_call_lines | 0.000 | 8.000 | +8.000 | 0.0001 | 0.0011 | **\*** |
| atlas_ws1_mask_fires | 0.000 | 14.000 | +14.000 | 0.0001 | 0.0011 | **\*** |
| atlas_b1_drift_fires | 0.000 | 1.000 | +1.000 | 0.0002 | 0.0028 | **\*** |
| files_written | 1159.500 | 1.000 | -1158.500 | 0.0002 | 0.0031 | **\*** |
| wall_time_s | 254.567 | 108.297 | -146.270 | 0.0013 | 0.0237 | **\*** |
| cargo_toml_valid | 1.000 | 0.000 | -1.000 | 0.0016 | 0.0291 | **\*** |
| tool_calls_total | 16.000 | 8.000 | -8.000 | 0.0028 | 0.0498 | **\*** |
| drift_toml_newlines_collapsed | 0.000 | 1.000 | +1.000 | 0.0059 | 0.1065 |  |
| cargo_toml_present | 1.000 | 0.500 | -0.500 | 0.0137 | 0.2457 |  |
| write_calls | 2.000 | 1.500 | -0.500 | 0.5974 | 1.0000 |  |
| drift_empty_path | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_path_outside_target | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_path_literal_space | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_lean_prefix | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_bash_as_content | 0.000 | 0.000 | +0.000 | 0.3681 | 1.0000 |  |
| drift_xml_attr_leak | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_tier5c_retries | 0.000 | 0.000 | +0.000 | 0.1681 | 1.0000 |  |
| atlas_a2_fuzzy_fires | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
