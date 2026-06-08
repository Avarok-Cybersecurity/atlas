# Tier comparison — `tierA` (n=10) vs `sm1` (n=10)

alpha=0.05 (Bonferroni-adjusted across 18 metrics).
Significant rows are starred (*).

| metric | median(tierA) | median(sm1) | Δ | p | p_bonf | sig |
|---|---|---|---|---|---|---|
| atlas_ws1_mask_fires | 0.000 | 11.000 | +11.000 | 0.0001 | 0.0011 | **\*** |
| atlas_b1_drift_fires | 0.000 | 2.500 | +2.500 | 0.0002 | 0.0038 | **\*** |
| files_written | 0.500 | 1.000 | +0.500 | 0.4811 | 1.0000 |  |
| cargo_toml_valid | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| cargo_toml_present | 0.500 | 1.000 | +0.500 | 0.1851 | 1.0000 |  |
| tool_calls_total | 8.500 | 7.000 | -1.500 | 0.2850 | 1.0000 |  |
| write_calls | 1.500 | 2.000 | +0.500 | 0.4403 | 1.0000 |  |
| drift_empty_path | 0.000 | 0.000 | +0.000 | 0.5823 | 1.0000 |  |
| drift_path_outside_target | 0.000 | 0.000 | +0.000 | 0.8711 | 1.0000 |  |
| drift_path_literal_space | 0.000 | 0.000 | +0.000 | 0.3681 | 1.0000 |  |
| drift_lean_prefix | 0.000 | 0.000 | +0.000 | 0.3681 | 1.0000 |  |
| drift_bash_as_content | 0.000 | 0.000 | +0.000 | 0.3681 | 1.0000 |  |
| drift_xml_attr_leak | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_toml_newlines_collapsed | 0.500 | 0.000 | -0.500 | 0.5871 | 1.0000 |  |
| atlas_tier5c_retries | 0.000 | 0.000 | +0.000 | 0.5823 | 1.0000 |  |
| atlas_a2_fuzzy_fires | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_tool_call_lines | 8.500 | 6.500 | -2.000 | 0.2532 | 1.0000 |  |
| wall_time_s | 168.930 | 130.362 | -38.568 | 0.5708 | 1.0000 |  |
