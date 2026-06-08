# Tier comparison — `sm1_a2ao_sc1` (n=10) vs `sm1_a2ao_sc1_steps` (n=10)

alpha=0.05 (Bonferroni-adjusted across 18 metrics).
Significant rows are starred (*).

| metric | median(sm1_a2ao_sc1) | median(sm1_a2ao_sc1_steps) | Δ | p | p_bonf | sig |
|---|---|---|---|---|---|---|
| files_written | 1.000 | 1.000 | +0.000 | 0.6340 | 1.0000 |  |
| cargo_toml_valid | 0.000 | 0.000 | +0.000 | 0.6506 | 1.0000 |  |
| cargo_toml_present | 1.000 | 1.000 | +0.000 | 1.0000 | 1.0000 |  |
| tool_calls_total | 10.000 | 7.000 | -3.000 | 0.7041 | 1.0000 |  |
| write_calls | 1.000 | 1.500 | +0.500 | 0.6695 | 1.0000 |  |
| drift_empty_path | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_path_outside_target | 0.000 | 0.000 | +0.000 | 0.1497 | 1.0000 |  |
| drift_path_literal_space | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_lean_prefix | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_bash_as_content | 0.000 | 0.000 | +0.000 | 0.3681 | 1.0000 |  |
| drift_xml_attr_leak | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_toml_newlines_collapsed | 0.000 | 1.000 | +1.000 | 0.1439 | 1.0000 |  |
| atlas_ws1_mask_fires | 20.000 | 17.000 | -3.000 | 0.9396 | 1.0000 |  |
| atlas_b1_drift_fires | 1.500 | 2.000 | +0.500 | 0.9079 | 1.0000 |  |
| atlas_tier5c_retries | 0.000 | 0.000 | +0.000 | 0.9567 | 1.0000 |  |
| atlas_a2_fuzzy_fires | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_tool_call_lines | 10.000 | 6.500 | -3.500 | 0.7905 | 1.0000 |  |
| wall_time_s | 127.134 | 168.414 | +41.281 | 0.3075 | 1.0000 |  |
