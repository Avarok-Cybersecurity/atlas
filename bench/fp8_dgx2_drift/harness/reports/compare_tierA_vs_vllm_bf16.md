# Tier comparison — `tierA` (n=10) vs `vllm_bf16` (n=10)

alpha=0.05 (Bonferroni-adjusted across 18 metrics).
Significant rows are starred (*).

| metric | median(tierA) | median(vllm_bf16) | Δ | p | p_bonf | sig |
|---|---|---|---|---|---|---|
| files_written | 0.500 | 1159.500 | +1159.000 | 0.0002 | 0.0029 | **\*** |
| atlas_tool_call_lines | 8.500 | 0.000 | -8.500 | 0.0002 | 0.0041 | **\*** |
| tool_calls_total | 8.500 | 16.000 | +7.500 | 0.0006 | 0.0100 | **\*** |
| cargo_toml_valid | 0.000 | 1.000 | +1.000 | 0.0016 | 0.0291 | **\*** |
| cargo_toml_present | 0.500 | 1.000 | +0.500 | 0.0137 | 0.2457 |  |
| drift_toml_newlines_collapsed | 0.500 | 0.000 | -0.500 | 0.0148 | 0.2664 |  |
| wall_time_s | 168.930 | 254.567 | +85.637 | 0.0173 | 0.3106 |  |
| write_calls | 1.500 | 2.000 | +0.500 | 0.6544 | 1.0000 |  |
| drift_empty_path | 0.000 | 0.000 | +0.000 | 0.1675 | 1.0000 |  |
| drift_path_outside_target | 0.000 | 0.000 | +0.000 | 0.1675 | 1.0000 |  |
| drift_path_literal_space | 0.000 | 0.000 | +0.000 | 0.3681 | 1.0000 |  |
| drift_lean_prefix | 0.000 | 0.000 | +0.000 | 0.3681 | 1.0000 |  |
| drift_bash_as_content | 0.000 | 0.000 | +0.000 | 0.3681 | 1.0000 |  |
| drift_xml_attr_leak | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_ws1_mask_fires | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_b1_drift_fires | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_tier5c_retries | 0.000 | 0.000 | +0.000 | 0.0776 | 1.0000 |  |
| atlas_a2_fuzzy_fires | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
