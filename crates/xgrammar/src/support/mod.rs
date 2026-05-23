// SPDX-License-Identifier: AGPL-3.0-only
//
// Support utilities — port of `cpp/support/*` (the subset that needs a
// hand-written Rust implementation).
//
// Headers replaced by crates instead of being ported:
//   dynamic_bitset.h  -> bitvec
//   thread_pool.h     -> rayon
//   thread_safe_cache.h -> dashmap
//   json_serializer.h / reflection.h -> serde
//   logging.h / recursion_guard.h / cpptrace.h / memory_size.h /
//   container.h -> standard library / `thiserror`
//
// Ported modules:
//   encoding         <- encoding.h   (UTF-8 / hex / Latin-1)
//   escape           <- encoding.h   (grammar-literal escape handling)
//   int_set          <- int_set.h
//   compact_2d_array <- compact_2d_array.h
//   union_find       <- union_find_set.h
//   hash             <- utils.h      (hash-combine helpers only)

pub mod compact_2d_array;
pub mod encoding;
pub mod escape;
pub mod hash;
pub mod int_set;
pub mod union_find;

pub use compact_2d_array::Compact2DArray;
pub use encoding::{
    TCodepoint, byte_to_latin1, char_handling_error, char_to_utf8, handle_utf8_first_byte,
    hex_char_to_int, latin1_to_bytes, parse_next_utf8, parse_utf8,
};
pub use escape::{
    parse_next_escaped, parse_next_utf8_or_escaped, print_as_escaped, print_byte_as_escaped,
    print_str_as_escaped, unescape_string,
};
pub use hash::{hash_combine, hash_combine_binary};
pub use int_set::{intset_intersection, intset_union};
pub use union_find::UnionFindSet;
