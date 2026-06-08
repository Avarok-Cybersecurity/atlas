# Tier comparison — `sm1_a2ao_sc1` (n=10) vs `mixedkv2` (n=10)

alpha=0.05 (Bonferroni-adjusted across 18 metrics).
Significant rows are starred (*).

| metric | median(sm1_a2ao_sc1) | median(mixedkv2) | Δ | p | p_bonf | sig |
|---|---|---|---|---|---|---|
| tool_calls_total | 10.000 | 6.500 | -3.500 | 0.0479 | 0.8625 |  |
| files_written | 1.000 | 0.000 | -1.000 | 0.3452 | 1.0000 |  |
| cargo_toml_valid | 0.000 | 0.000 | +0.000 | 0.3006 | 1.0000 |  |
| cargo_toml_present | 1.000 | 0.000 | -1.000 | 0.4076 | 1.0000 |  |
| write_calls | 1.000 | 1.500 | +0.500 | 1.0000 | 1.0000 |  |
| drift_empty_path | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_path_outside_target | 0.000 | 0.000 | +0.000 | 0.6701 | 1.0000 |  |
| drift_path_literal_space | 0.000 | 0.000 | +0.000 | 0.1681 | 1.0000 |  |
| drift_lean_prefix | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_bash_as_content | 0.000 | 0.000 | +0.000 | 0.3681 | 1.0000 |  |
| drift_xml_attr_leak | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_toml_newlines_collapsed | 0.000 | 0.000 | +0.000 | 0.5622 | 1.0000 |  |
| atlas_ws1_mask_fires | 20.000 | 13.000 | -7.000 | 0.1200 | 1.0000 |  |
| atlas_b1_drift_fires | 1.500 | 4.500 | +3.000 | 0.0667 | 1.0000 |  |
| atlas_tier5c_retries | 0.000 | 0.000 | +0.000 | 0.5828 | 1.0000 |  |
| atlas_a2_fuzzy_fires | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_tool_call_lines | 10.000 | 6.000 | -4.000 | 0.0622 | 1.0000 |  |
| wall_time_s | 127.134 | 170.623 | +43.489 | 0.1859 | 1.0000 |  |
