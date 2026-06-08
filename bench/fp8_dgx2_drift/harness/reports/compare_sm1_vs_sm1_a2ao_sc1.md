# Tier comparison — `sm1` (n=10) vs `sm1_a2ao_sc1` (n=10)

alpha=0.05 (Bonferroni-adjusted across 18 metrics).
Significant rows are starred (*).

| metric | median(sm1) | median(sm1_a2ao_sc1) | Δ | p | p_bonf | sig |
|---|---|---|---|---|---|---|
| files_written | 1.000 | 1.000 | +0.000 | 0.3446 | 1.0000 |  |
| cargo_toml_valid | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| cargo_toml_present | 1.000 | 1.000 | +0.000 | 0.3662 | 1.0000 |  |
| tool_calls_total | 7.000 | 10.000 | +3.000 | 0.1381 | 1.0000 |  |
| write_calls | 2.000 | 1.000 | -1.000 | 0.6414 | 1.0000 |  |
| drift_empty_path | 0.000 | 0.000 | +0.000 | 0.2781 | 1.0000 |  |
| drift_path_outside_target | 0.000 | 0.000 | +0.000 | 0.6264 | 1.0000 |  |
| drift_path_literal_space | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_lean_prefix | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_bash_as_content | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_xml_attr_leak | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_toml_newlines_collapsed | 0.000 | 0.000 | +0.000 | 0.6867 | 1.0000 |  |
| atlas_ws1_mask_fires | 11.000 | 20.000 | +9.000 | 0.0581 | 1.0000 |  |
| atlas_b1_drift_fires | 2.500 | 1.500 | -1.000 | 0.7860 | 1.0000 |  |
| atlas_tier5c_retries | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_a2_fuzzy_fires | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_tool_call_lines | 6.500 | 10.000 | +3.500 | 0.1288 | 1.0000 |  |
| wall_time_s | 130.362 | 127.134 | -3.228 | 0.5205 | 1.0000 |  |
