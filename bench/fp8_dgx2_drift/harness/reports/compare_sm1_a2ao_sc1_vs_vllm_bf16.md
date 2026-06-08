# Tier comparison — `sm1_a2ao_sc1` (n=10) vs `vllm_bf16` (n=10)

alpha=0.05 (Bonferroni-adjusted across 18 metrics).
Significant rows are starred (*).

| metric | median(sm1_a2ao_sc1) | median(vllm_bf16) | Δ | p | p_bonf | sig |
|---|---|---|---|---|---|---|
| atlas_ws1_mask_fires | 20.000 | 0.000 | -20.000 | 0.0001 | 0.0011 | **\*** |
| atlas_tool_call_lines | 10.000 | 0.000 | -10.000 | 0.0001 | 0.0011 | **\*** |
| files_written | 1.000 | 1159.500 | +1158.500 | 0.0002 | 0.0030 | **\*** |
| atlas_b1_drift_fires | 1.500 | 0.000 | -1.500 | 0.0007 | 0.0131 | **\*** |
| wall_time_s | 127.134 | 254.567 | +127.433 | 0.0013 | 0.0237 | **\*** |
| cargo_toml_valid | 0.000 | 1.000 | +1.000 | 0.0016 | 0.0291 | **\*** |
| tool_calls_total | 10.000 | 16.000 | +6.000 | 0.0039 | 0.0706 |  |
| cargo_toml_present | 1.000 | 1.000 | +0.000 | 0.0336 | 0.6046 |  |
| write_calls | 1.000 | 2.000 | +1.000 | 0.3581 | 1.0000 |  |
| drift_empty_path | 0.000 | 0.000 | +0.000 | 0.3681 | 1.0000 |  |
| drift_path_outside_target | 0.000 | 0.000 | +0.000 | 0.3681 | 1.0000 |  |
| drift_path_literal_space | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_lean_prefix | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_bash_as_content | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_xml_attr_leak | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_toml_newlines_collapsed | 0.000 | 0.000 | +0.000 | 0.0776 | 1.0000 |  |
| atlas_tier5c_retries | 0.000 | 0.000 | +0.000 | 0.1675 | 1.0000 |  |
| atlas_a2_fuzzy_fires | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
