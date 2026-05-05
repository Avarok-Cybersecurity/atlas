# Copilot instructions for xgrammar-rs

What this repo is
- Rust bindings for the C++ XGrammar engine vendored in `external/xgrammar`.
- The build compiles C++ with CMake (static `xgrammar`), then generates Rust FFI with `autocxx`/`cxx`.

Repo layout
- Rust crate: `src/**`, build script `build.rs`, tests in `tests/`.
- Vendored C++: `external/xgrammar/{include,cpp,3rdparty,..}` (keep pristine).
- C++ shims used by FFI live in `src/cxx_utils/**` and are included via `src/cxx_utils.hpp`.
- FFI declarations are in `src/lib.rs` with `include_cpp! { generate!(...) }`.

Build and test (macOS, zsh)
- Prereqs: CMake and a C++23 toolchain (macOS: `xcode-select --install`; optional `brew install cmake`).
- Build: `cargo build`. Test: `cargo test`. For paths to generated code: `cargo build -vv`.

Non-obvious build details
- `build.rs` sets CMake flags: C++17, disables upstream Python/C++ tests.
- C++ standard library linking is handled automatically by the `cxx`/`autocxx` crates (via `link-cplusplus`).
- Apple targets: configures `CMAKE_OSX_ARCHITECTURES` and iOS `CMAKE_OSX_SYSROOT`; honors `IPHONEOS_DEPLOYMENT_TARGET`.
- Copies headers into `OUT_DIR/autocxx-build-dir/rs` to satisfy generated `include!(...)` paths.

Public Rust API (today)
- `pub use TokenizerInfo;`, `pub use VocabType;` from `xgrammar`.
- `TokenizerInfo` wraps C++ `xgrammar::TokenizerInfo`:
  - `new(&vocab, VocabType::RAW|BYTE_FALLBACK, &stop_ids, add_prefix_space)`; use `new_with_vocab_size(.., Some(padded))` for padded model vocabs.
  - `decoded_vocab()` returns bytes; JSON via `serialize_json()` and `deserialize_json(..) -> Option<Self>`.
- See `tests/tokenizer_info_serialization.rs` and `tests/json_generation.rs` for usage.

Extending bindings (pattern)
1) Add `generate!("xgrammar::TypeOrFunc")` in `src/lib.rs`.
2) If STL-heavy or complex, add a tiny helper in `src/cxx_utils/*.hpp` and include it in `src/cxx_utils.hpp`.
3) Add a safe Rust wrapper module and re-export from `lib.rs`.
4) `cargo build` to regenerate the bridge.

Conventions and gotchas
- Always use full, explicit names for variables and function arguments; avoid abbreviations and single-letter names (except trivial loop indices).
- Prefer shims over editing upstream C++; construct STL types via `cxx_utils` helpers (e.g., `new_string_vector`, `string_vec_push_bytes`).
- Deserialization errors return `None` by design (mirrors C++ `SerializationError`).
- iOS builds are supported by `build.rs`; ensure `TARGET` is an iOS triple when cross-compiling.

Open questions for maintainers
- Prioritize exposing `GrammarCompiler`/`GrammarMatcher` in the Rust API?
- Any CI/formatting expectations beyond optional `rustfmt` on generated code?
