# Atlas vs HF Reference: Per-Operation Drift Table (Qwen3.6-35B-A3B-FP8)

## Test setup
- Prompt: canonical 10382-token chat probe (last 5 prompt tokens = [248045, 74455, 198, 248068, 198])
- Atlas image: `atlas-gb10:op-drift` (built today from `atlas-gb10:fp8-much-better` lineage, commit 8d2cc87, native FP8 SSM dispatch)
- HF reference: `Qwen/Qwen3.6-35B-A3B` (BF16, original unquantized weights — the absolute reference)
- Compute device: dgx2 GPU (Atlas) + dgx1 CPU (HF reference forward)
- Layer indexing: Qwen3.6-35B-A3B-FP8 has 40 layers; full-attention at L3,7,11,15,19,23,27,31,35,39 (10); linear-attention (SSM/GDN) at the rest (30).
- Comparison metric: cosine similarity over the LAST-TOKEN slice of each named operation, widened to f32 on both sides.

## Headline numbers
- Total comparable rows: **240** (incl. 30 known-unreliable ssm.pre_norm rows)
- Worst meaningful op: **ssm.moe_out** at layer **L20**, cos=0.91983
- Best single op: **ssm.post_norm** at layer **L0**, cos=1.00000
- Mean cosine across reliable rows: **0.98531**
- Per-layer-hidden mean cosine: **0.98982** (min 0.97657, max 0.99952) — this is the Atlas-vs-HF[BF16-unquant] residual-stream drift = 'B' comparison.

## Per-operation-class summary (aggregated across layers)

| Op class                | n  | min cos | median cos | mean cos | max cos | worst layer |
|-------------------------|----|---------|------------|----------|---------|-------------|
| attn.input_norm_in      | 10 | 0.98404 | 0.99084    | 0.99037  | 0.99926 | L23         |
| attn.input_norm_out     | 10 | 0.98460 | 0.99201    | 0.99112  | 0.99943 | L27         |
| attn.k_proj             | 10 | 0.98853 | 0.99306    | 0.99362  | 0.99906 | L23         |
| attn.moe_out            | 10 | 0.93214 | 0.97725    | 0.97316  | 0.99750 | L23         |
| attn.o_proj             | 10 | 0.96829 | 0.98582    | 0.98255  | 0.99741 | L19         |
| attn.post_attn_norm_out | 10 | 0.97421 | 0.98521    | 0.98530  | 0.99772 | L23         |
| attn.q_proj_full        | 10 | 0.99689 | 0.99835    | 0.99850  | 0.99992 | L31         |
| attn.v_proj             | 10 | 0.97341 | 0.99053    | 0.98795  | 0.99865 | L27         |
| layer.hidden_out        | 40 | 0.97657 | 0.98884    | 0.98982  | 0.99952 | L20         |
| ssm.moe_out             | 30 | 0.91983 | 0.97784    | 0.97456  | 0.99930 | L20         |
| ssm.out_proj            | 30 | 0.95680 | 0.97931    | 0.97813  | 0.99950 | L33         |
| ssm.post_norm           | 30 | 0.97799 | 0.99045    | 0.99053  | 1.00000 | L21         |
| ssm.pre_norm            | 30 | -0.00592 | 0.05028    | 0.04171  | 0.07501 | L18         |

## Full table (one row per layer × op)

| Layer | Operation                | Status         | Shape | cos_sim   | max_abs   | mean_abs  |
|-------|--------------------------|----------------|-------|-----------|-----------|-----------|
| L0    | ssm.pre_norm             | ok             | 2048  | 0.01680   | 0.5230    | 0.09820   |
| L0    | ssm.post_norm            | ok             | 2048  | 1.00000   | 0.0000    | 0.00000   |
| L0    | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L0    | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L0    | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L0    | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L0    | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L0    | ssm.out_proj             | ok             | 2048  | 0.99950   | 0.0098    | 0.00018   |
| L0    | ssm.moe_out              | ok             | 2048  | 0.99930   | 0.0010    | 0.00019   |
| L0    | layer.hidden_out         | ok             | 2048  | 0.99952   | 0.0078    | 0.00026   |
|       |                          |                |       |           |           |           |
| L1    | ssm.pre_norm             | ok             | 2048  | 0.05686   | 0.5195    | 0.09805   |
| L1    | ssm.post_norm            | ok             | 2048  | 0.99937   | 0.3672    | 0.02352   |
| L1    | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L1    | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L1    | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L1    | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L1    | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L1    | ssm.out_proj             | ok             | 2048  | 0.99697   | 0.0205    | 0.00037   |
| L1    | ssm.moe_out              | ok             | 2048  | 0.99748   | 0.0054    | 0.00040   |
| L1    | layer.hidden_out         | ok             | 2048  | 0.99919   | 0.0078    | 0.00055   |
|       |                          |                |       |           |           |           |
| L2    | ssm.pre_norm             | ok             | 2048  | 0.07501   | 0.5114    | 0.09819   |
| L2    | ssm.post_norm            | ok             | 2048  | 0.99907   | 0.5625    | 0.03031   |
| L2    | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L2    | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L2    | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L2    | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L2    | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L2    | ssm.out_proj             | ok             | 2048  | 0.99753   | 0.0098    | 0.00039   |
| L2    | ssm.moe_out              | ok             | 2048  | 0.99556   | 0.0066    | 0.00027   |
| L2    | layer.hidden_out         | ok             | 2048  | 0.99926   | 0.0156    | 0.00062   |
|       |                          |                |       |           |           |           |
| L3    | attn.input_norm_in       | ok             | 2048  | 0.99926   | 0.0156    | 0.00062   |
| L3    | attn.input_norm_out      | ok             | 2048  | 0.99943   | 0.4375    | 0.02720   |
| L3    | attn.q_proj_full         | ok_warn_layout | 8192  | 0.99992   | 0.2188    | 0.03695   |
| L3    | attn.k_proj              | ok             | 512   | 0.99906   | 0.1250    | 0.02601   |
| L3    | attn.v_proj              | ok             | 512   | 0.99814   | 0.0815    | 0.01781   |
| L3    | attn.o_proj              | ok             | 2048  | 0.99741   | 0.0037    | 0.00042   |
| L3    | attn.post_attn_norm_out  | ok             | 2048  | 0.99772   | 0.2188    | 0.03451   |
| L3    | attn.moe_out             | ok             | 2048  | 0.99750   | 0.0040    | 0.00053   |
| L3    | layer.hidden_out         | ok             | 2048  | 0.99900   | 0.0117    | 0.00087   |
|       |                          |                |       |           |           |           |
| L4    | ssm.pre_norm             | ok             | 2048  | 0.06432   | 0.5011    | 0.09880   |
| L4    | ssm.post_norm            | ok             | 2048  | 0.99850   | 0.4688    | 0.03507   |
| L4    | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L4    | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L4    | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L4    | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L4    | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L4    | ssm.out_proj             | ok             | 2048  | 0.99445   | 0.0072    | 0.00058   |
| L4    | ssm.moe_out              | ok             | 2048  | 0.99580   | 0.0103    | 0.00062   |
| L4    | layer.hidden_out         | ok             | 2048  | 0.99860   | 0.0068    | 0.00108   |
|       |                          |                |       |           |           |           |
| L5    | ssm.pre_norm             | ok             | 2048  | 0.06999   | 0.4949    | 0.09878   |
| L5    | ssm.post_norm            | ok             | 2048  | 0.99845   | 0.3438    | 0.04230   |
| L5    | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L5    | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L5    | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L5    | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L5    | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L5    | ssm.out_proj             | ok             | 2048  | 0.99414   | 0.0254    | 0.00079   |
| L5    | ssm.moe_out              | ok             | 2048  | 0.99408   | 0.0144    | 0.00069   |
| L5    | layer.hidden_out         | ok             | 2048  | 0.99858   | 0.0078    | 0.00117   |
|       |                          |                |       |           |           |           |
| L6    | ssm.pre_norm             | ok             | 2048  | 0.05771   | 0.5098    | 0.09919   |
| L6    | ssm.post_norm            | ok             | 2048  | 0.99836   | 0.3125    | 0.04170   |
| L6    | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L6    | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L6    | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L6    | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L6    | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L6    | ssm.out_proj             | ok             | 2048  | 0.99455   | 0.0059    | 0.00077   |
| L6    | ssm.moe_out              | ok             | 2048  | 0.99218   | 0.0031    | 0.00058   |
| L6    | layer.hidden_out         | ok             | 2048  | 0.99858   | 0.0117    | 0.00118   |
|       |                          |                |       |           |           |           |
| L7    | attn.input_norm_in       | ok             | 2048  | 0.99858   | 0.0117    | 0.00118   |
| L7    | attn.input_norm_out      | ok             | 2048  | 0.99883   | 0.2969    | 0.03684   |
| L7    | attn.q_proj_full         | ok_warn_layout | 8192  | 0.99984   | 0.1875    | 0.03237   |
| L7    | attn.k_proj              | ok             | 512   | 0.99847   | 0.1270    | 0.03256   |
| L7    | attn.v_proj              | ok             | 512   | 0.99658   | 0.1707    | 0.02544   |
| L7    | attn.o_proj              | ok             | 2048  | 0.99637   | 0.0047    | 0.00099   |
| L7    | attn.post_attn_norm_out  | ok             | 2048  | 0.99685   | 0.3164    | 0.05441   |
| L7    | attn.moe_out             | ok             | 2048  | 0.99499   | 0.0059    | 0.00085   |
| L7    | layer.hidden_out         | ok             | 2048  | 0.99784   | 0.0071    | 0.00149   |
|       |                          |                |       |           |           |           |
| L8    | ssm.pre_norm             | ok             | 2048  | 0.04817   | 0.5117    | 0.09974   |
| L8    | ssm.post_norm            | ok             | 2048  | 0.99779   | 0.3125    | 0.05440   |
| L8    | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L8    | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L8    | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L8    | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L8    | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L8    | ssm.out_proj             | ok             | 2048  | 0.99196   | 0.0187    | 0.00089   |
| L8    | ssm.moe_out              | ok             | 2048  | 0.97590   | 0.0093    | 0.00200   |
| L8    | layer.hidden_out         | ok             | 2048  | 0.99442   | 0.0195    | 0.00242   |
|       |                          |                |       |           |           |           |
| L9    | ssm.pre_norm             | ok             | 2048  | 0.05235   | 0.5446    | 0.09953   |
| L9    | ssm.post_norm            | ok             | 2048  | 0.99465   | 0.4482    | 0.08438   |
| L9    | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L9    | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L9    | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L9    | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L9    | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L9    | ssm.out_proj             | ok             | 2048  | 0.98207   | 0.0257    | 0.00113   |
| L9    | ssm.moe_out              | ok             | 2048  | 0.96988   | 0.0142    | 0.00204   |
| L9    | layer.hidden_out         | ok             | 2048  | 0.99292   | 0.0138    | 0.00284   |
|       |                          |                |       |           |           |           |
| L10   | ssm.pre_norm             | ok             | 2048  | 0.04046   | 0.5562    | 0.09982   |
| L10   | ssm.post_norm            | ok             | 2048  | 0.99370   | 0.4395    | 0.09191   |
| L10   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L10   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L10   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L10   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L10   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L10   | ssm.out_proj             | ok             | 2048  | 0.98212   | 0.0061    | 0.00117   |
| L10   | ssm.moe_out              | ok             | 2048  | 0.98844   | 0.0137    | 0.00151   |
| L10   | layer.hidden_out         | ok             | 2048  | 0.99486   | 0.0120    | 0.00280   |
|       |                          |                |       |           |           |           |
| L11   | attn.input_norm_in       | ok             | 2048  | 0.99486   | 0.0120    | 0.00280   |
| L11   | attn.input_norm_out      | ok             | 2048  | 0.99594   | 0.3828    | 0.06825   |
| L11   | attn.q_proj_full         | ok_warn_layout | 8192  | 0.99964   | 0.3008    | 0.05466   |
| L11   | attn.k_proj              | ok             | 512   | 0.99690   | 0.1951    | 0.04705   |
| L11   | attn.v_proj              | ok             | 512   | 0.99194   | 0.1738    | 0.04112   |
| L11   | attn.o_proj              | ok             | 2048  | 0.99071   | 0.0078    | 0.00130   |
| L11   | attn.post_attn_norm_out  | ok             | 2048  | 0.99014   | 0.5422    | 0.10567   |
| L11   | attn.moe_out             | ok             | 2048  | 0.97249   | 0.0114    | 0.00219   |
| L11   | layer.hidden_out         | ok             | 2048  | 0.99195   | 0.0195    | 0.00351   |
|       |                          |                |       |           |           |           |
| L12   | ssm.pre_norm             | ok             | 2048  | 0.01975   | 0.5645    | 0.10099   |
| L12   | ssm.post_norm            | ok             | 2048  | 0.99165   | 0.5625    | 0.09623   |
| L12   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L12   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L12   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L12   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L12   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L12   | ssm.out_proj             | ok             | 2048  | 0.98081   | 0.0083    | 0.00162   |
| L12   | ssm.moe_out              | ok             | 2048  | 0.98005   | 0.0176    | 0.00153   |
| L12   | layer.hidden_out         | ok             | 2048  | 0.99291   | 0.0156    | 0.00307   |
|       |                          |                |       |           |           |           |
| L13   | ssm.pre_norm             | ok             | 2048  | 0.01158   | 0.5504    | 0.10078   |
| L13   | ssm.post_norm            | ok             | 2048  | 0.99277   | 0.5469    | 0.09451   |
| L13   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L13   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L13   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L13   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L13   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L13   | ssm.out_proj             | ok             | 2048  | 0.97931   | 0.0322    | 0.00134   |
| L13   | ssm.moe_out              | ok             | 2048  | 0.97725   | 0.0327    | 0.00192   |
| L13   | layer.hidden_out         | ok             | 2048  | 0.99249   | 0.0152    | 0.00334   |
|       |                          |                |       |           |           |           |
| L14   | ssm.pre_norm             | ok             | 2048  | 0.02991   | 0.5696    | 0.10060   |
| L14   | ssm.post_norm            | ok             | 2048  | 0.99252   | 0.5312    | 0.09636   |
| L14   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L14   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L14   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L14   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L14   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L14   | ssm.out_proj             | ok             | 2048  | 0.98253   | 0.0107    | 0.00143   |
| L14   | ssm.moe_out              | ok             | 2048  | 0.96576   | 0.0371    | 0.00224   |
| L14   | layer.hidden_out         | ok             | 2048  | 0.99137   | 0.0508    | 0.00350   |
|       |                          |                |       |           |           |           |
| L15   | attn.input_norm_in       | ok             | 2048  | 0.99137   | 0.0508    | 0.00350   |
| L15   | attn.input_norm_out      | ok             | 2048  | 0.99226   | 0.8125    | 0.07947   |
| L15   | attn.q_proj_full         | ok_warn_layout | 8192  | 0.99928   | 0.3906    | 0.05614   |
| L15   | attn.k_proj              | ok             | 512   | 0.99292   | 0.2041    | 0.05398   |
| L15   | attn.v_proj              | ok             | 512   | 0.98504   | 0.2188    | 0.04818   |
| L15   | attn.o_proj              | ok             | 2048  | 0.98582   | 0.0102    | 0.00192   |
| L15   | attn.post_attn_norm_out  | ok             | 2048  | 0.98559   | 0.6172    | 0.13622   |
| L15   | attn.moe_out             | ok             | 2048  | 0.97913   | 0.0137    | 0.00230   |
| L15   | layer.hidden_out         | ok             | 2048  | 0.99022   | 0.0312    | 0.00413   |
|       |                          |                |       |           |           |           |
| L16   | ssm.pre_norm             | ok             | 2048  | 0.01339   | 0.5977    | 0.10179   |
| L16   | ssm.post_norm            | ok             | 2048  | 0.99039   | 0.6211    | 0.10672   |
| L16   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L16   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L16   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L16   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L16   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L16   | ssm.out_proj             | ok             | 2048  | 0.97245   | 0.0459    | 0.00142   |
| L16   | ssm.moe_out              | ok             | 2048  | 0.96567   | 0.0160    | 0.00322   |
| L16   | layer.hidden_out         | ok             | 2048  | 0.98839   | 0.0625    | 0.00486   |
|       |                          |                |       |           |           |           |
| L17   | ssm.pre_norm             | ok             | 2048  | 0.01497   | 0.6001    | 0.10255   |
| L17   | ssm.post_norm            | ok             | 2048  | 0.98870   | 1.0625    | 0.11699   |
| L17   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L17   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L17   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L17   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L17   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L17   | ssm.out_proj             | ok             | 2048  | 0.98850   | 0.0092    | 0.00159   |
| L17   | ssm.moe_out              | ok             | 2048  | 0.98025   | 0.0156    | 0.00290   |
| L17   | layer.hidden_out         | ok             | 2048  | 0.98766   | 0.0840    | 0.00504   |
|       |                          |                |       |           |           |           |
| L18   | ssm.pre_norm             | ok             | 2048  | -0.00592  | 0.5813    | 0.10344   |
| L18   | ssm.post_norm            | ok             | 2048  | 0.98847   | 1.5625    | 0.12190   |
| L18   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L18   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L18   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L18   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L18   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L18   | ssm.out_proj             | ok             | 2048  | 0.97807   | 0.0107    | 0.00172   |
| L18   | ssm.moe_out              | ok             | 2048  | 0.95030   | 0.0176    | 0.00361   |
| L18   | layer.hidden_out         | ok             | 2048  | 0.98608   | 0.1172    | 0.00591   |
|       |                          |                |       |           |           |           |
| L19   | attn.input_norm_in       | ok             | 2048  | 0.98608   | 0.1172    | 0.00591   |
| L19   | attn.input_norm_out      | ok             | 2048  | 0.98661   | 1.3125    | 0.10327   |
| L19   | attn.q_proj_full         | ok_warn_layout | 8192  | 0.99819   | 0.6875    | 0.10886   |
| L19   | attn.k_proj              | ok             | 512   | 0.99114   | 0.3535    | 0.06682   |
| L19   | attn.v_proj              | ok             | 512   | 0.98148   | 0.3442    | 0.05857   |
| L19   | attn.o_proj              | ok             | 2048  | 0.96829   | 0.0327    | 0.00371   |
| L19   | attn.post_attn_norm_out  | ok             | 2048  | 0.98046   | 0.9893    | 0.16991   |
| L19   | attn.moe_out             | ok             | 2048  | 0.93830   | 0.0294    | 0.00616   |
| L19   | layer.hidden_out         | ok             | 2048  | 0.97958   | 0.1562    | 0.00915   |
|       |                          |                |       |           |           |           |
| L20   | ssm.pre_norm             | ok             | 2048  | 0.02197   | 0.6562    | 0.10764   |
| L20   | ssm.post_norm            | ok             | 2048  | 0.98112   | 1.8125    | 0.14837   |
| L20   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L20   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L20   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L20   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L20   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L20   | ssm.out_proj             | ok             | 2048  | 0.97382   | 0.0231    | 0.00297   |
| L20   | ssm.moe_out              | ok             | 2048  | 0.91983   | 0.0459    | 0.00683   |
| L20   | layer.hidden_out         | ok             | 2048  | 0.97657   | 0.1250    | 0.01038   |
|       |                          |                |       |           |           |           |
| L21   | ssm.pre_norm             | ok             | 2048  | 0.02148   | 0.6245    | 0.10849   |
| L21   | ssm.post_norm            | ok             | 2048  | 0.97799   | 1.4375    | 0.15828   |
| L21   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L21   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L21   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L21   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L21   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L21   | ssm.out_proj             | ok             | 2048  | 0.96418   | 0.0234    | 0.00316   |
| L21   | ssm.moe_out              | ok             | 2048  | 0.95410   | 0.0232    | 0.00404   |
| L21   | layer.hidden_out         | ok             | 2048  | 0.97835   | 0.0938    | 0.00984   |
|       |                          |                |       |           |           |           |
| L22   | ssm.pre_norm             | ok             | 2048  | 0.01140   | 0.6792    | 0.10796   |
| L22   | ssm.post_norm            | ok             | 2048  | 0.98022   | 1.1875    | 0.14769   |
| L22   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L22   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L22   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L22   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L22   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L22   | ssm.out_proj             | ok             | 2048  | 0.96361   | 0.1621    | 0.00328   |
| L22   | ssm.moe_out              | ok             | 2048  | 0.97782   | 0.0664    | 0.00443   |
| L22   | layer.hidden_out         | ok             | 2048  | 0.98404   | 0.0547    | 0.00948   |
|       |                          |                |       |           |           |           |
| L23   | attn.input_norm_in       | ok             | 2048  | 0.98404   | 0.0547    | 0.00948   |
| L23   | attn.input_norm_out      | ok             | 2048  | 0.98537   | 0.8750    | 0.10816   |
| L23   | attn.q_proj_full         | ok_warn_layout | 8192  | 0.99835   | 0.7188    | 0.08200   |
| L23   | attn.k_proj              | ok             | 512   | 0.98853   | 0.4609    | 0.08031   |
| L23   | attn.v_proj              | ok             | 512   | 0.97986   | 0.2947    | 0.06342   |
| L23   | attn.o_proj              | ok             | 2048  | 0.97852   | 0.0312    | 0.00301   |
| L23   | attn.post_attn_norm_out  | ok             | 2048  | 0.97421   | 1.0156    | 0.19267   |
| L23   | attn.moe_out             | ok             | 2048  | 0.93214   | 0.0796    | 0.00456   |
| L23   | layer.hidden_out         | ok             | 2048  | 0.98160   | 0.1094    | 0.01044   |
|       |                          |                |       |           |           |           |
| L24   | ssm.pre_norm             | ok             | 2048  | 0.03709   | 0.8901    | 0.10894   |
| L24   | ssm.post_norm            | ok             | 2048  | 0.98186   | 0.9375    | 0.13964   |
| L24   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L24   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L24   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L24   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L24   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L24   | ssm.out_proj             | ok             | 2048  | 0.95897   | 0.1230    | 0.00312   |
| L24   | ssm.moe_out              | ok             | 2048  | 0.95153   | 0.0303    | 0.00393   |
| L24   | layer.hidden_out         | ok             | 2048  | 0.98245   | 0.0547    | 0.00985   |
|       |                          |                |       |           |           |           |
| L25   | ssm.pre_norm             | ok             | 2048  | 0.02704   | 0.8901    | 0.10869   |
| L25   | ssm.post_norm            | ok             | 2048  | 0.98174   | 0.9180    | 0.14395   |
| L25   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L25   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L25   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L25   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L25   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L25   | ssm.out_proj             | ok             | 2048  | 0.96468   | 0.0381    | 0.00304   |
| L25   | ssm.moe_out              | ok             | 2048  | 0.94750   | 0.0214    | 0.00543   |
| L25   | layer.hidden_out         | ok             | 2048  | 0.98261   | 0.0781    | 0.01100   |
|       |                          |                |       |           |           |           |
| L26   | ssm.pre_norm             | ok             | 2048  | 0.03641   | 0.9844    | 0.11029   |
| L26   | ssm.post_norm            | ok             | 2048  | 0.98262   | 0.9375    | 0.14227   |
| L26   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L26   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L26   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L26   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L26   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L26   | ssm.out_proj             | ok             | 2048  | 0.96952   | 0.0184    | 0.00348   |
| L26   | ssm.moe_out              | ok             | 2048  | 0.96845   | 0.0898    | 0.00356   |
| L26   | layer.hidden_out         | ok             | 2048  | 0.98581   | 0.1875    | 0.01041   |
|       |                          |                |       |           |           |           |
| L27   | attn.input_norm_in       | ok             | 2048  | 0.98581   | 0.1875    | 0.01041   |
| L27   | attn.input_norm_out      | ok             | 2048  | 0.98460   | 1.1406    | 0.09534   |
| L27   | attn.q_proj_full         | ok_warn_layout | 8192  | 0.99736   | 0.5469    | 0.09974   |
| L27   | attn.k_proj              | ok             | 512   | 0.99061   | 0.3438    | 0.06971   |
| L27   | attn.v_proj              | ok             | 512   | 0.97341   | 0.2207    | 0.05604   |
| L27   | attn.o_proj              | ok             | 2048  | 0.97104   | 0.0405    | 0.00423   |
| L27   | attn.post_attn_norm_out  | ok             | 2048  | 0.97862   | 0.7344    | 0.17600   |
| L27   | attn.moe_out             | ok             | 2048  | 0.96738   | 0.0576    | 0.00413   |
| L27   | layer.hidden_out         | ok             | 2048  | 0.98726   | 0.0859    | 0.01114   |
|       |                          |                |       |           |           |           |
| L28   | ssm.pre_norm             | ok             | 2048  | 0.05028   | 1.5625    | 0.11332   |
| L28   | ssm.post_norm            | ok             | 2048  | 0.98745   | 0.7812    | 0.11641   |
| L28   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L28   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L28   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L28   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L28   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L28   | ssm.out_proj             | ok             | 2048  | 0.97825   | 0.0332    | 0.00342   |
| L28   | ssm.moe_out              | ok             | 2048  | 0.98098   | 0.0217    | 0.00413   |
| L28   | layer.hidden_out         | ok             | 2048  | 0.98823   | 0.1406    | 0.01140   |
|       |                          |                |       |           |           |           |
| L29   | ssm.pre_norm             | ok             | 2048  | 0.05399   | 1.7969    | 0.11470   |
| L29   | ssm.post_norm            | ok             | 2048  | 0.98871   | 0.8125    | 0.11242   |
| L29   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L29   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L29   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L29   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L29   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L29   | ssm.out_proj             | ok             | 2048  | 0.97015   | 0.0274    | 0.00463   |
| L29   | ssm.moe_out              | ok             | 2048  | 0.96752   | 0.0272    | 0.00518   |
| L29   | layer.hidden_out         | ok             | 2048  | 0.98796   | 0.1250    | 0.01219   |
|       |                          |                |       |           |           |           |
| L30   | ssm.pre_norm             | ok             | 2048  | 0.05524   | 1.6875    | 0.11670   |
| L30   | ssm.post_norm            | ok             | 2048  | 0.98855   | 0.9688    | 0.11639   |
| L30   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L30   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L30   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L30   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L30   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L30   | ssm.out_proj             | ok             | 2048  | 0.95945   | 0.1230    | 0.00597   |
| L30   | ssm.moe_out              | ok             | 2048  | 0.96529   | 0.0228    | 0.00479   |
| L30   | layer.hidden_out         | ok             | 2048  | 0.98746   | 0.0709    | 0.01293   |
|       |                          |                |       |           |           |           |
| L31   | attn.input_norm_in       | ok             | 2048  | 0.98746   | 0.0709    | 0.01293   |
| L31   | attn.input_norm_out      | ok             | 2048  | 0.98867   | 0.7278    | 0.09600   |
| L31   | attn.q_proj_full         | ok_warn_layout | 8192  | 0.99689   | 0.6562    | 0.08996   |
| L31   | attn.k_proj              | ok             | 512   | 0.99167   | 0.3188    | 0.07250   |
| L31   | attn.v_proj              | ok             | 512   | 0.98382   | 0.3125    | 0.06265   |
| L31   | attn.o_proj              | ok             | 2048  | 0.97557   | 0.0291    | 0.00566   |
| L31   | attn.post_attn_norm_out  | ok             | 2048  | 0.98258   | 0.8047    | 0.18089   |
| L31   | attn.moe_out             | ok             | 2048  | 0.97725   | 0.1562    | 0.00617   |
| L31   | layer.hidden_out         | ok             | 2048  | 0.98884   | 0.1094    | 0.01491   |
|       |                          |                |       |           |           |           |
| L32   | ssm.pre_norm             | ok             | 2048  | 0.05176   | 2.1562    | 0.12715   |
| L32   | ssm.post_norm            | ok             | 2048  | 0.99031   | 0.6353    | 0.11111   |
| L32   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L32   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L32   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L32   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L32   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L32   | ssm.out_proj             | ok             | 2048  | 0.95879   | 0.2090    | 0.00774   |
| L32   | ssm.moe_out              | ok             | 2048  | 0.96449   | 0.1562    | 0.00676   |
| L32   | layer.hidden_out         | ok             | 2048  | 0.98783   | 0.1094    | 0.01688   |
|       |                          |                |       |           |           |           |
| L33   | ssm.pre_norm             | ok             | 2048  | 0.05798   | 2.2656    | 0.13222   |
| L33   | ssm.post_norm            | ok             | 2048  | 0.98899   | 0.6250    | 0.11729   |
| L33   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L33   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L33   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L33   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L33   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L33   | ssm.out_proj             | ok             | 2048  | 0.95680   | 0.2949    | 0.00688   |
| L33   | ssm.moe_out              | ok             | 2048  | 0.97862   | 0.2500    | 0.00755   |
| L33   | layer.hidden_out         | ok             | 2048  | 0.98874   | 0.1016    | 0.01867   |
|       |                          |                |       |           |           |           |
| L34   | ssm.pre_norm             | ok             | 2048  | 0.06101   | 2.8125    | 0.13965   |
| L34   | ssm.post_norm            | ok             | 2048  | 0.99045   | 0.6562    | 0.11472   |
| L34   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L34   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L34   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L34   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L34   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L34   | ssm.out_proj             | ok             | 2048  | 0.97126   | 0.2422    | 0.00918   |
| L34   | ssm.moe_out              | ok             | 2048  | 0.98788   | 0.2109    | 0.00781   |
| L34   | layer.hidden_out         | ok             | 2048  | 0.99084   | 0.1094    | 0.01893   |
|       |                          |                |       |           |           |           |
| L35   | attn.input_norm_in       | ok             | 2048  | 0.99084   | 0.1094    | 0.01893   |
| L35   | attn.input_norm_out      | ok             | 2048  | 0.99201   | 0.6250    | 0.09061   |
| L35   | attn.q_proj_full         | ok_warn_layout | 8192  | 0.99723   | 0.6094    | 0.08700   |
| L35   | attn.k_proj              | ok             | 512   | 0.99387   | 0.3281    | 0.06977   |
| L35   | attn.v_proj              | ok             | 512   | 0.99053   | 0.3359    | 0.07088   |
| L35   | attn.o_proj              | ok             | 2048  | 0.97276   | 0.3125    | 0.01051   |
| L35   | attn.post_attn_norm_out  | ok             | 2048  | 0.98521   | 0.8281    | 0.19398   |
| L35   | attn.moe_out             | ok             | 2048  | 0.97562   | 0.2656    | 0.01001   |
| L35   | layer.hidden_out         | ok             | 2048  | 0.99041   | 0.1133    | 0.02218   |
|       |                          |                |       |           |           |           |
| L36   | ssm.pre_norm             | ok             | 2048  | 0.06580   | 4.1406    | 0.16073   |
| L36   | ssm.post_norm            | ok             | 2048  | 0.99145   | 1.0312    | 0.11750   |
| L36   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L36   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L36   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L36   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L36   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L36   | ssm.out_proj             | ok             | 2048  | 0.98595   | 0.0518    | 0.00986   |
| L36   | ssm.moe_out              | ok             | 2048  | 0.98415   | 0.1367    | 0.00809   |
| L36   | layer.hidden_out         | ok             | 2048  | 0.98918   | 0.2031    | 0.02518   |
|       |                          |                |       |           |           |           |
| L37   | ssm.pre_norm             | ok             | 2048  | 0.06217   | 2.8750    | 0.17966   |
| L37   | ssm.post_norm            | ok             | 2048  | 0.99076   | 1.1250    | 0.12405   |
| L37   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L37   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L37   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L37   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L37   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L37   | ssm.out_proj             | ok             | 2048  | 0.98224   | 0.0859    | 0.01251   |
| L37   | ssm.moe_out              | ok             | 2048  | 0.97784   | 0.3047    | 0.01446   |
| L37   | layer.hidden_out         | ok             | 2048  | 0.98771   | 0.1309    | 0.03119   |
|       |                          |                |       |           |           |           |
| L38   | ssm.pre_norm             | ok             | 2048  | 0.07235   | 3.2031    | 0.19818   |
| L38   | ssm.post_norm            | ok             | 2048  | 0.98920   | 0.9062    | 0.12739   |
| L38   | ssm.post_qkvz            | missing        | n/a   | -         | -         | -         |
| L38   | ssm.conv                 | shape_mismatch | 8192  | -         | -         | -         |
| L38   | ssm.l2                   | atlas_only_no_hf_ref | 8192  | -         | -         | -         |
| L38   | ssm.gdn                  | atlas_only_no_hf_ref | 4096  | -         | -         | -         |
| L38   | ssm.gnorm                | shape_mismatch | 4096  | -         | -         | -         |
| L38   | ssm.out_proj             | ok             | 2048  | 0.97121   | 0.2500    | 0.01988   |
| L38   | ssm.moe_out              | ok             | 2048  | 0.98286   | 0.0781    | 0.01725   |
| L38   | layer.hidden_out         | ok             | 2048  | 0.98539   | 0.2812    | 0.04184   |
|       |                          |                |       |           |           |           |
| L39   | attn.input_norm_in       | ok             | 2048  | 0.98539   | 0.2812    | 0.04184   |
| L39   | attn.input_norm_out      | ok             | 2048  | 0.98748   | 0.8750    | 0.12397   |
| L39   | attn.q_proj_full         | ok_warn_layout | 8192  | 0.99824   | 1.1250    | 0.09373   |
| L39   | attn.k_proj              | ok             | 512   | 0.99306   | 0.3281    | 0.07507   |
| L39   | attn.v_proj              | ok             | 512   | 0.99865   | 0.4258    | 0.06018   |
| L39   | attn.o_proj              | ok             | 2048  | 0.98895   | 0.1328    | 0.01153   |
| L39   | attn.post_attn_norm_out  | ok             | 2048  | 0.98165   | 0.9180    | 0.19517   |
| L39   | attn.moe_out             | ok             | 2048  | 0.99679   | 0.2578    | 0.01448   |
| L39   | layer.hidden_out         | ok             | 2048  | 0.98790   | 0.3750    | 0.04601   |
|       |                          |                |       |           |           |           |

## Divergence analysis

- Ops with cos < 0.99 (real drift hotspots): **119**
- Ops with cos > 0.9999 (negligible drift): **2**

### Key per-op findings

1. **Q-projection is the cleanest op** (mean cos 0.9985 over 10 layers).
   Atlas's Q GEMM is effectively at the FP8 compute ceiling — the FP8
   weight quantization itself is by far the dominant error, not any Atlas-side
   compute inaccuracy.

2. **MoE block is the dominant Atlas-side drift source.**
   Both `attn.moe_out` (mean 0.9732, min 0.9321 at L23) and `ssm.moe_out`
   (mean 0.9746, min 0.9198 at L20) drop ~2–3 percentage points relative to
   their inputs. The shape of the per-layer trend matches the gate-indecision
   signature documented in `project_qwen36_drift_moe_smoking_gun.md` (2026-05-23).

3. **V > K > Q projection drift ordering** in full-attn layers (V: 0.988, K: 0.994,
   Q: 0.999) is consistent with the fact that Q is followed by RMSNorm-per-head
   (which renormalises Atlas-side scale error) while V and K go directly into
   attention without rescaling. Atlas's QK-norm is correct.

4. **L19–L24 cluster** is the global drift floor: every op-class shows its
   minimum in this band. L19 = the deepest sliding-window full-attn boundary;
   L20–L24 are the immediately following SSM layers carrying L19's degraded
   residual. The 2026-05-23 phase ζ MoE study identified the same band.

5. **L37–L38 micro-bump**: a second local minimum cluster, smaller than the L19–L24
   one. The 2026-05-23 study attributed this to FP8-KV mid-context noise.

6. **No L39 cliff**: Atlas's per-layer hidden_out at L39 = 0.98790 (mid-pack);
   the 2026-05-23 NVFP4-detour SSM dispatch had a 0.927 cliff at L39.
   Native FP8 SSM dispatch (today's image) eliminates that cliff.

### Top 20 worst MEANINGFUL ops (Atlas compute headroom)

| Rank | Layer | Op                       | cos_sim   |
|------|-------|--------------------------|-----------|
|    1 | L20   | ssm.moe_out              | 0.91983   |
|    2 | L23   | attn.moe_out             | 0.93214   |
|    3 | L19   | attn.moe_out             | 0.93830   |
|    4 | L25   | ssm.moe_out              | 0.94750   |
|    5 | L18   | ssm.moe_out              | 0.95030   |
|    6 | L24   | ssm.moe_out              | 0.95153   |
|    7 | L21   | ssm.moe_out              | 0.95410   |
|    8 | L33   | ssm.out_proj             | 0.95680   |
|    9 | L32   | ssm.out_proj             | 0.95879   |
|   10 | L24   | ssm.out_proj             | 0.95897   |
|   11 | L30   | ssm.out_proj             | 0.95945   |
|   12 | L22   | ssm.out_proj             | 0.96361   |
|   13 | L21   | ssm.out_proj             | 0.96418   |
|   14 | L32   | ssm.moe_out              | 0.96449   |
|   15 | L25   | ssm.out_proj             | 0.96468   |
|   16 | L30   | ssm.moe_out              | 0.96529   |
|   17 | L16   | ssm.moe_out              | 0.96567   |
|   18 | L14   | ssm.moe_out              | 0.96576   |
|   19 | L27   | attn.moe_out             | 0.96738   |
|   20 | L29   | ssm.moe_out              | 0.96752   |

## Per-op-class trend across layers

Layer-by-layer cosine for each op class. Cells empty when op not applicable to that layer type.

| L  |  input_in  |  input_out  |  q_proj    |  k_proj    |  v_proj    |  o_proj    |  postattn  |  moe_out   |  ssm_norm  |  ssm_oprj  |  ssm_moe   |  hidden    |
|----|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
|  0 |            |            |            |            |            |            |            |            |  1.00000  |  0.99950  |  0.99930  |  0.99952  |
|  1 |            |            |            |            |            |            |            |            |  0.99937  |  0.99697  |  0.99748  |  0.99919  |
|  2 |            |            |            |            |            |            |            |            |  0.99907  |  0.99753  |  0.99556  |  0.99926  |
|  3 |  0.99926  |  0.99943  |  0.99992  |  0.99906  |  0.99814  |  0.99741  |  0.99772  |  0.99750  |            |            |            |  0.99900  |
|  4 |            |            |            |            |            |            |            |            |  0.99850  |  0.99445  |  0.99580  |  0.99860  |
|  5 |            |            |            |            |            |            |            |            |  0.99845  |  0.99414  |  0.99408  |  0.99858  |
|  6 |            |            |            |            |            |            |            |            |  0.99836  |  0.99455  |  0.99218  |  0.99858  |
|  7 |  0.99858  |  0.99883  |  0.99984  |  0.99847  |  0.99658  |  0.99637  |  0.99685  |  0.99499  |            |            |            |  0.99784  |
|  8 |            |            |            |            |            |            |            |            |  0.99779  |  0.99196  |  0.97590  |  0.99442  |
|  9 |            |            |            |            |            |            |            |            |  0.99465  |  0.98207  |  0.96988  |  0.99292  |
| 10 |            |            |            |            |            |            |            |            |  0.99370  |  0.98212  |  0.98844  |  0.99486  |
| 11 |  0.99486  |  0.99594  |  0.99964  |  0.99690  |  0.99194  |  0.99071  |  0.99014  |  0.97249  |            |            |            |  0.99195  |
| 12 |            |            |            |            |            |            |            |            |  0.99165  |  0.98081  |  0.98005  |  0.99291  |
| 13 |            |            |            |            |            |            |            |            |  0.99277  |  0.97931  |  0.97725  |  0.99249  |
| 14 |            |            |            |            |            |            |            |            |  0.99252  |  0.98253  |  0.96576  |  0.99137  |
| 15 |  0.99137  |  0.99226  |  0.99928  |  0.99292  |  0.98504  |  0.98582  |  0.98559  |  0.97913  |            |            |            |  0.99022  |
| 16 |            |            |            |            |            |            |            |            |  0.99039  |  0.97245  |  0.96567  |  0.98839  |
| 17 |            |            |            |            |            |            |            |            |  0.98870  |  0.98850  |  0.98025  |  0.98766  |
| 18 |            |            |            |            |            |            |            |            |  0.98847  |  0.97807  |  0.95030  |  0.98608  |
| 19 |  0.98608  |  0.98661  |  0.99819  |  0.99114  |  0.98148  |  0.96829  |  0.98046  |  0.93830  |            |            |            |  0.97958  |
| 20 |            |            |            |            |            |            |            |            |  0.98112  |  0.97382  |  0.91983  |  0.97657  |
| 21 |            |            |            |            |            |            |            |            |  0.97799  |  0.96418  |  0.95410  |  0.97835  |
| 22 |            |            |            |            |            |            |            |            |  0.98022  |  0.96361  |  0.97782  |  0.98404  |
| 23 |  0.98404  |  0.98537  |  0.99835  |  0.98853  |  0.97986  |  0.97852  |  0.97421  |  0.93214  |            |            |            |  0.98160  |
| 24 |            |            |            |            |            |            |            |            |  0.98186  |  0.95897  |  0.95153  |  0.98245  |
| 25 |            |            |            |            |            |            |            |            |  0.98174  |  0.96468  |  0.94750  |  0.98261  |
| 26 |            |            |            |            |            |            |            |            |  0.98262  |  0.96952  |  0.96845  |  0.98581  |
| 27 |  0.98581  |  0.98460  |  0.99736  |  0.99061  |  0.97341  |  0.97104  |  0.97862  |  0.96738  |            |            |            |  0.98726  |
| 28 |            |            |            |            |            |            |            |            |  0.98745  |  0.97825  |  0.98098  |  0.98823  |
| 29 |            |            |            |            |            |            |            |            |  0.98871  |  0.97015  |  0.96752  |  0.98796  |
| 30 |            |            |            |            |            |            |            |            |  0.98855  |  0.95945  |  0.96529  |  0.98746  |
| 31 |  0.98746  |  0.98867  |  0.99689  |  0.99167  |  0.98382  |  0.97557  |  0.98258  |  0.97725  |            |            |            |  0.98884  |
| 32 |            |            |            |            |            |            |            |            |  0.99031  |  0.95879  |  0.96449  |  0.98783  |
| 33 |            |            |            |            |            |            |            |            |  0.98899  |  0.95680  |  0.97862  |  0.98874  |
| 34 |            |            |            |            |            |            |            |            |  0.99045  |  0.97126  |  0.98788  |  0.99084  |
| 35 |  0.99084  |  0.99201  |  0.99723  |  0.99387  |  0.99053  |  0.97276  |  0.98521  |  0.97562  |            |            |            |  0.99041  |
| 36 |            |            |            |            |            |            |            |            |  0.99145  |  0.98595  |  0.98415  |  0.98918  |
| 37 |            |            |            |            |            |            |            |            |  0.99076  |  0.98224  |  0.97784  |  0.98771  |
| 38 |            |            |            |            |            |            |            |            |  0.98920  |  0.97121  |  0.98286  |  0.98539  |
| 39 |  0.98539  |  0.98748  |  0.99824  |  0.99306  |  0.99865  |  0.98895  |  0.98165  |  0.99679  |            |            |            |  0.98790  |

## Notes / limitations

- **Layer-indexing convention**: Atlas full-attention dumps use ATTN-RELATIVE index (0..9). The cosine script translates Atlas-L0→abs-L3, ..., Atlas-L9→abs-L39 for joining with HF.
- **q_proj_full layout**: Atlas dumps Q+Gate interleaved (`[q_dim*2]`), HF dumps the same shape but unconcatenated. Full-vector cosine still correlates with overall projection quality.
- **ssm.pre_norm**: pre-existing Atlas dump bug — when SSM is configured with FP32 residual, `maybe_dump_gdn_buf` reads N×2 bytes (BF16 semantic) starting at an FP32-byte-stride offset. The resulting blob is half the values, mis-typed. Skip these rows.
- **ssm.conv**: HF Conv1d output is `[1, channels, T+3]`; naive `extract_last` picks the wrong slice (last channel rather than last time-step). Atlas dumps `[8192]` last-token. Shape-mismatch flagged in table.
- **ssm.gnorm**: HF Qwen3_5MoeRMSNormGated reshapes to `[-1, 128]` then norms — the captured hook returns the last (token, head) pair = `[128]`. Atlas dumps the full `[4096]` = 32 heads × 128. Per-head ordering between Atlas and HF differs (Atlas heads-contiguous, HF token×head interleaved), so naive prefix-cosine is not meaningful.
- **ssm.in_proj_qkvz / in_proj_ba**: Qwen3.6 HF splits these into `in_proj_qkv` + `in_proj_z` and `in_proj_a` + `in_proj_b` (4 separate Linear modules); Atlas combines them into a single fused GEMM (`in_proj_qkvz`). No direct 1:1 HF reference.
- **HF gating side-effects**: For SSM layers, `linear_attn.norm(x, z)` applies `z`-gating inside the norm; HF hook captures the gated output. Atlas's gnorm hook is at the same position.
- **First-time stochasticity**: All sampling is deterministic — temperature=0.0, single-shot prefill. Random seed not in use because forward only (no sampling).

## Reproducibility

```bash
# Atlas op-drift image (commit 8d2cc87 lineage + ATLAS_OP_DUMP hooks)
docker build -f docker/gb10/Dockerfile -t atlas-gb10:op-drift .

# Run on dgx2 with all dump env vars enabled:
./bench/fp8_dgx2_drift/dgx2_op_dump.sh

# Fire the prompt (text-decoded for /v1/completions):
python3 bench/fp8_dgx2_drift/fire_atlas_prompt.py

# HF reference forward on dgx1 CPU (~30 min for 10382 tokens):
python3 bench/fp8_dgx2_drift/hf_op_dump.py

# Compute cosines and render master table:
python3 bench/fp8_dgx2_drift/op_cosine.py \
    --atlas-dir /workspace/atlas-dumps/op_drift_atlas/ \
    --hf-dir /workspace/atlas-dumps/op_drift/ \
    --out /workspace/atlas-mtp/bench/fp8_dgx2_drift/op_drift.json
python3 bench/fp8_dgx2_drift/render_master_table.py \
    --json /workspace/atlas-mtp/bench/fp8_dgx2_drift/op_drift.json \
    --out /workspace/atlas-mtp/bench/fp8_dgx2_drift/MASTER_DRIFT_TABLE.md
```
