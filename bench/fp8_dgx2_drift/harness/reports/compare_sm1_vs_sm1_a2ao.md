# Tier comparison — `sm1` (n=10) vs `sm1_a2ao` (n=10)

alpha=0.05 (Bonferroni-adjusted across 18 metrics).
Significant rows are starred (*).

| metric | median(sm1) | median(sm1_a2ao) | Δ | p | p_bonf | sig |
|---|---|---|---|---|---|---|
| files_written | 1.000 | 1.000 | +0.000 | 0.4439 | 1.0000 |  |
| cargo_toml_valid | 0.000 | 0.000 | +0.000 | 0.6506 | 1.0000 |  |
| cargo_toml_present | 1.000 | 1.000 | +0.000 | 0.6506 | 1.0000 |  |
| tool_calls_total | 7.000 | 7.000 | +0.000 | 0.7293 | 1.0000 |  |
| write_calls | 2.000 | 2.000 | +0.000 | 0.6406 | 1.0000 |  |
| drift_empty_path | 0.000 | 0.000 | +0.000 | 0.2781 | 1.0000 |  |
| drift_path_outside_target | 0.000 | 0.000 | +0.000 | 0.1681 | 1.0000 |  |
| drift_path_literal_space | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_lean_prefix | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_bash_as_content | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_xml_attr_leak | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_toml_newlines_collapsed | 0.000 | 1.000 | +1.000 | 0.2076 | 1.0000 |  |
| atlas_ws1_mask_fires | 11.000 | 15.000 | +4.000 | 0.4235 | 1.0000 |  |
| atlas_b1_drift_fires | 2.500 | 3.000 | +0.500 | 0.6739 | 1.0000 |  |
| atlas_tier5c_retries | 0.000 | 0.000 | +0.000 | 0.6701 | 1.0000 |  |
| atlas_a2_fuzzy_fires | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_tool_call_lines | 6.500 | 7.000 | +0.500 | 0.7006 | 1.0000 |  |
| wall_time_s | 130.362 | 160.674 | +30.312 | 0.2123 | 1.0000 |  |
