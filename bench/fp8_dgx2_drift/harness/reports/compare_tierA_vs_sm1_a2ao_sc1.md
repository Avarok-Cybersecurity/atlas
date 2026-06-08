# Tier comparison — `tierA` (n=10) vs `sm1_a2ao_sc1` (n=10)

alpha=0.05 (Bonferroni-adjusted across 18 metrics).
Significant rows are starred (*).

| metric | median(tierA) | median(sm1_a2ao_sc1) | Δ | p | p_bonf | sig |
|---|---|---|---|---|---|---|
| atlas_ws1_mask_fires | 0.000 | 20.000 | +20.000 | 0.0001 | 0.0011 | **\*** |
| atlas_b1_drift_fires | 0.000 | 1.500 | +1.500 | 0.0007 | 0.0131 | **\*** |
| files_written | 0.500 | 1.000 | +0.500 | 0.9681 | 1.0000 |  |
| cargo_toml_valid | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| cargo_toml_present | 0.500 | 1.000 | +0.500 | 0.6934 | 1.0000 |  |
| tool_calls_total | 8.500 | 10.000 | +1.500 | 0.4244 | 1.0000 |  |
| write_calls | 1.500 | 1.000 | -0.500 | 1.0000 | 1.0000 |  |
| drift_empty_path | 0.000 | 0.000 | +0.000 | 0.5828 | 1.0000 |  |
| drift_path_outside_target | 0.000 | 0.000 | +0.000 | 0.6701 | 1.0000 |  |
| drift_path_literal_space | 0.000 | 0.000 | +0.000 | 0.3681 | 1.0000 |  |
| drift_lean_prefix | 0.000 | 0.000 | +0.000 | 0.3681 | 1.0000 |  |
| drift_bash_as_content | 0.000 | 0.000 | +0.000 | 0.3681 | 1.0000 |  |
| drift_xml_attr_leak | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| drift_toml_newlines_collapsed | 0.500 | 0.000 | -0.500 | 0.3026 | 1.0000 |  |
| atlas_tier5c_retries | 0.000 | 0.000 | +0.000 | 0.5823 | 1.0000 |  |
| atlas_a2_fuzzy_fires | 0.000 | 0.000 | +0.000 | 1.0000 | 1.0000 |  |
| atlas_tool_call_lines | 8.500 | 10.000 | +1.500 | 0.3611 | 1.0000 |  |
| wall_time_s | 168.930 | 127.134 | -41.796 | 0.9698 | 1.0000 |  |
