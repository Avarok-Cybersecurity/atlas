// SPDX-License-Identifier: AGPL-3.0-only

//! The per-request `control_vector` field, as a THREE-state directive.
//!
//! # Why `Option<String>` was not enough
//!
//! It collapses two different requests into the same value:
//!
//! ```text
//! {}                            -> None
//! {"control_vector": null}      -> None
//! ```
//!
//! Today both mean "no steering" and nothing is lost. The moment a server
//! default exists they must diverge — omitting the field should accept the
//! deployment's default, while sending `null` is a caller explicitly declining
//! it — and `Option<String>` has no room left to express that.
//!
//! So the distinction is introduced BEFORE the default is, while every reading
//! still produces today's behaviour. Adding it later would be a silent change
//! in meaning for every request that omits the field, which is the kind of
//! break that cannot be done quietly.
//!
//! ```text
//! omitted           -> ServerDefault   (no default configured => no steering)
//! null | false | "" -> Off             (explicitly none)
//! "name"            -> Named("name")
//! true              -> rejected
//! ```
//!
//! `true` is refused rather than guessed at. A serve can hold many vectors and
//! "on" does not say which; picking one for the caller would be inventing
//! intent, and picking the only one when there happens to be exactly one would
//! make a request's meaning depend on the server's inventory.

use serde::{Deserialize, Deserializer};

/// What a request said about steering.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CvecDirective {
    /// The field was absent: take whatever the deployment decided.
    #[default]
    ServerDefault,
    /// `null`, `false` or `""`: this caller wants no steering, whatever the
    /// deployment default says.
    Off,
    /// An explicit selection.
    Named(String),
}

/// The shapes the field accepts, before interpretation.
#[derive(Deserialize)]
#[serde(untagged)]
enum Wire {
    Name(String),
    Toggle(bool),
}

/// Deserialize the field.
///
/// Used as `#[serde(default, deserialize_with = "deserialize_cvec_directive")]`,
/// and the pairing is what makes the three states representable:
/// `deserialize_with` runs ONLY when the key is present, so a missing key falls
/// to `Default` (`ServerDefault`) without ever reaching here, and a `null`
/// arrives as `None` from the inner `Option`. A plain `Option<String>` field
/// cannot express this — serde maps both to `None`.
pub fn deserialize_cvec_directive<'de, D>(d: D) -> Result<CvecDirective, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error;
    Ok(match Option::<Wire>::deserialize(d)? {
        // Key present with `null`.
        None => CvecDirective::Off,
        Some(Wire::Toggle(false)) => CvecDirective::Off,
        Some(Wire::Toggle(true)) => {
            return Err(D::Error::custom(
                "control_vector: `true` does not say WHICH vector. Send the name \
                 as a string, omit the field to take the server default, or send \
                 null/false for no steering.",
            ));
        }
        Some(Wire::Name(s)) if s.trim().is_empty() => CvecDirective::Off,
        Some(Wire::Name(s)) => CvecDirective::Named(s.trim().to_string()),
    })
}

#[cfg(test)]
#[path = "control_vector_directive_tests.rs"]
mod control_vector_directive_tests;
