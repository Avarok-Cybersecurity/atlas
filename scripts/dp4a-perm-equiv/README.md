# DP4A v_perm codebook expansion — equivalence proof

Standalone gfx1151 test proving the branchless `__builtin_amdgcn_perm` codebook
expansion (grabbed from rocmfp4-llama, adapted to Atlas's consecutive-pair layout
+ `{0,1,2,3,4,6,8,12}` grid) is **byte-exact** vs the portable per-element loop, for
all inputs. This locks the two codebook encodings (perm constants vs `DP4A_CODEBOOK`)
together.

## Run (on a gfx1151 / Strix Halo box)
```bash
export LD_LIBRARY_PATH=/opt/rocm/core-7.13/lib:/opt/rocm/lib
hipcc -x hip --offload-arch=gfx1151 -O2 perm_equiv_test.cu -o perm_equiv_test
./perm_equiv_test
# expect: mismatches = 0  -> PASS (perm == reference, bit-exact)
```

Verified 2026-06-21 on Radeon 8060S (gfx1151): `mismatches = 0`.
