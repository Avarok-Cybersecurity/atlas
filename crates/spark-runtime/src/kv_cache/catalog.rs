// SPDX-License-Identifier: AGPL-3.0-only

//! The exhaustive [`KvCacheDtype`] catalogue: every variant and its canonical
//! CLI spelling, in one place the compiler defends.
//!
//! `--kv-cache-dtype` is a plain `String` in clap, so anything that wants to
//! LIST the valid values — the CLI validator's error text, the dashboard's
//! option picker — has historically had to copy them by hand, and a copied
//! list drifts: it offers a value the server refuses, or hides one that
//! works. Both consumers now read [`KvCacheDtype::ALL`], and the `name` match
//! below is what keeps `ALL` honest — it has no wildcard arm, so adding a
//! variant fails THIS build until the new name is written here, next to the
//! two-line comment telling you to extend `ALL` with it.

use super::KvCacheDtype;

impl KvCacheDtype {
    /// Every variant, in the order the enum declares them.
    ///
    /// Extend this together with [`KvCacheDtype::name`] below — the
    /// non-exhaustive-match error a new variant raises there points here.
    pub const ALL: [KvCacheDtype; 16] = [
        KvCacheDtype::Bf16,
        KvCacheDtype::Fp8,
        KvCacheDtype::Nvfp4,
        KvCacheDtype::Turbo4,
        KvCacheDtype::Turbo3,
        KvCacheDtype::Turbo2,
        KvCacheDtype::Turbo8,
        KvCacheDtype::Turbo4KTurbo3V,
        KvCacheDtype::Turbo4KTurbo8V,
        KvCacheDtype::Turbo3KTurbo8V,
        KvCacheDtype::Bf16KTurbo4V,
        KvCacheDtype::Bf16KTurbo3V,
        KvCacheDtype::Fp8KTurbo4V,
        KvCacheDtype::Fp8KTurbo3V,
        KvCacheDtype::Bf16KTurbo2V,
        KvCacheDtype::Fp8KTurbo2V,
    ];

    /// The canonical `--kv-cache-dtype` spelling. `Display` delegates here,
    /// so the string a picker offers is byte-for-byte the string the flag
    /// parser reads back — the round trip the tests below pin.
    pub const fn name(self) -> &'static str {
        match self {
            KvCacheDtype::Bf16 => "bf16",
            KvCacheDtype::Fp8 => "fp8",
            KvCacheDtype::Nvfp4 => "nvfp4",
            KvCacheDtype::Turbo4 => "turbo4",
            KvCacheDtype::Turbo3 => "turbo3",
            KvCacheDtype::Turbo2 => "turbo2",
            KvCacheDtype::Turbo8 => "turbo8",
            KvCacheDtype::Turbo4KTurbo3V => "turbo4k_turbo3v",
            KvCacheDtype::Turbo4KTurbo8V => "turbo4k_turbo8v",
            KvCacheDtype::Turbo3KTurbo8V => "turbo3k_turbo8v",
            KvCacheDtype::Bf16KTurbo4V => "bf16k_turbo4v",
            KvCacheDtype::Bf16KTurbo3V => "bf16k_turbo3v",
            KvCacheDtype::Fp8KTurbo4V => "fp8k_turbo4v",
            KvCacheDtype::Fp8KTurbo3V => "fp8k_turbo3v",
            KvCacheDtype::Bf16KTurbo2V => "bf16k_turbo2v",
            KvCacheDtype::Fp8KTurbo2V => "fp8k_turbo2v",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::KvCacheDtype;

    #[test]
    fn every_listed_name_parses_back_to_the_variant_that_produced_it() {
        // The catalogue's whole promise: a value copied out of `ALL` is a
        // value `FromStr` accepts, and it means the same dtype. A name that
        // fails either half is an option a picker would offer and the server
        // would refuse.
        for dtype in KvCacheDtype::ALL {
            let parsed: KvCacheDtype = dtype
                .name()
                .parse()
                .unwrap_or_else(|e| panic!("{} does not parse: {e:#}", dtype.name()));
            assert_eq!(
                parsed,
                dtype,
                "{} parses to a different dtype",
                dtype.name()
            );
            assert_eq!(
                dtype.to_string(),
                dtype.name(),
                "Display and the catalogue disagree"
            );
        }
    }

    #[test]
    fn the_catalogue_has_no_duplicates() {
        // A duplicated entry is a picker row that looks like a choice and is
        // not one; with sixteen hand-ordered entries it is an easy slip.
        for (i, a) in KvCacheDtype::ALL.iter().enumerate() {
            for b in &KvCacheDtype::ALL[i + 1..] {
                assert_ne!(a, b, "{} is listed twice", a.name());
            }
        }
    }
}
