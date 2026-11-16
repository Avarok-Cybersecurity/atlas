// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the serve options of the load-time BF16 -> affine routes (lane L23a): the code
//! width, the group size and the clip-search depth the GPU quantizer uses, and whether the prompt GEMM
//! takes the W4A8 int8 path.
//!
//! Resolution, per option: the serve flag (`--w4-bits`, `--w4-group`, `--w4-clip-search`, published
//! once by `spark-server`'s `port_apply` through [`apply`]), else the environment (`AVAROK_W4_BITS`,
//! `AVAROK_W4_GROUP`, `AVAROK_W4_CLIP_SEARCH`), else the default (4 bits, group 64, no clip search:
//! plain MLX min-max). `AVAROK_W4_GDN_GROUP` sets the group of the GDN projections alone, whose
//! quantization error feeds a recurrent state (a group of 32 is proposed there); without it
//! the GDN route uses the common group. The int8 prompt GEMM is the `affine4_prefill_int8` row, and
//! only the 4-bit kernels have it.
//!
//! The options change nothing by themselves: they apply only to a layer type whose route row is on
//! (`dense_affine4_from_bf16`, `gdn_affine4_from_bf16`, `attn_affine4_from_bf16`, all off on every
//! target), so a serve without those rows is byte-identical with or without the flags.
//!
//! Served combinations (the kernels that exist): 4 bits at group 32, 64 or 128 with at most 10
//! clip-search candidates (lane L17a, `port::affine`, `MAX_CLIP_SEARCH`); 2, 3, 5, 6 or 8 bits at group
//! 64 with at most 8 (lane L17c, `port::affine::widths`, `affine_n_quantize_ada`). [`validate`] refuses
//! anything else at startup, so the `OnceLock` in [`opts_for`] never sees an unservable value.

use std::sync::OnceLock;

use anyhow::{Context, Result, bail};

use crate::port::w4::W4Opts;

pub const ENV_BITS: &str = "AVAROK_W4_BITS";
pub const ENV_GROUP: &str = "AVAROK_W4_GROUP";
pub const ENV_GDN_GROUP: &str = "AVAROK_W4_GDN_GROUP";
pub const ENV_CLIP: &str = "AVAROK_W4_CLIP_SEARCH";

/// The 4-bit lane path's group sizes (`port::affine::weight::GROUPS`).
pub const GROUPS_4BIT: [u16; 3] = [32, 64, 128];
/// The other widths (`port::affine::widths`), all at [`WIDTHS_GROUP`].
pub const WIDTHS: [u8; 5] = [2, 3, 5, 6, 8];
pub const WIDTHS_GROUP: u16 = 64;
/// Clip-search candidates the 4-bit quantizer takes (`port::affine::weight::MAX_CLIP_SEARCH`).
pub const MAX_CLIP_4BIT: u8 = crate::port::affine::weight::MAX_CLIP_SEARCH;
/// Clip-search candidates `affine_n_quantize_ada` takes ("at most 8", its header).
pub const MAX_CLIP_WIDTHS: u8 = 8;

/// The layer types with an affine route (lane L23b's `W4Role`; the LM head has none, it is packed by
/// lane L24 from a GGUF checkpoint).
pub use crate::port::w4_stats::W4Role;

/// The roles an affine route serves, in boot-line order. Only the group can differ between them, and
/// only for GDN.
pub const AFFINE_ROLES: [W4Role; 3] = [W4Role::Ffn, W4Role::Gdn, W4Role::Attn];

/// What the quantizer and the prompt GEMM of one route use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineOpts {
    pub bits: u8,
    pub group: u16,
    pub clip_search: u8,
    /// The W4A8 int8 prompt GEMM (lane L17b); only ever true at 4 bits.
    pub int8_prefill: bool,
}

impl AffineOpts {
    /// 4 bits, group 64, plain MLX min-max, BF16 prompt GEMM.
    pub const DEFAULT: AffineOpts = AffineOpts {
        bits: 4,
        group: 64,
        clip_search: 0,
        int8_prefill: false,
    };

    /// The 3.7.2 constructor options (`port::affine::quantize_from_bf16`).
    pub fn w4(&self) -> W4Opts {
        W4Opts {
            bits: self.bits,
            group: self.group,
            clip_search: self.clip_search,
            int8_prefill: self.int8_prefill,
        }
    }

    /// `bits=4 group=64 clip=0 int8_prefill=off`, for the route's boot line.
    pub fn describe(&self) -> String {
        format!(
            "bits={} group={} clip={} int8_prefill={}",
            self.bits,
            self.group,
            self.clip_search,
            if self.int8_prefill { "on" } else { "off" }
        )
    }
}

/// The three serve flags, as `port_apply` hands them over (`None` = flag absent).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AffineCli {
    pub bits: Option<u8>,
    pub group: Option<u16>,
    pub clip_search: Option<u8>,
}

impl AffineCli {
    pub fn any(&self) -> bool {
        self.bits.is_some() || self.group.is_some() || self.clip_search.is_some()
    }
}

/// Whether the kernels serve `bits` at `group` with `clip` candidates; the error names the rule.
pub fn supported(bits: u8, group: u16, clip: u8) -> Result<()> {
    if bits == 4 {
        if !GROUPS_4BIT.contains(&group) {
            bail!("affine 4-bit takes a group of 32, 64 or 128, not {group}");
        }
        if clip > MAX_CLIP_4BIT {
            bail!("affine 4-bit takes at most {MAX_CLIP_4BIT} clip-search candidates, not {clip}");
        }
    } else if WIDTHS.contains(&bits) {
        if group != WIDTHS_GROUP {
            bail!("affine {bits}-bit is built for group {WIDTHS_GROUP} only, not {group}");
        }
        if clip > MAX_CLIP_WIDTHS {
            bail!(
                "affine {bits}-bit takes at most {MAX_CLIP_WIDTHS} clip-search candidates, not {clip}"
            );
        }
    } else {
        bail!("affine codes are 2, 3, 4, 5, 6 or 8 bits wide, not {bits}");
    }
    Ok(())
}

/// One environment value: absent or blank is `None`, anything that does not parse is an error naming
/// the variable (an operator's typo must not read as agreement with the default).
pub fn parse_env<T: std::str::FromStr>(name: &str, raw: Option<&str>) -> Result<Option<T>> {
    match raw.map(str::trim) {
        None | Some("") => Ok(None),
        Some(v) => v
            .parse::<T>()
            .map(Some)
            .map_err(|_| anyhow::anyhow!("{name}={v:?} is not a valid value")),
    }
}

/// Resolve the options of `role`, as a pure function of the flags, a variable lookup and the
/// `affine4_prefill_int8` row, so a CPU test can pin every precedence and refusal.
pub fn resolve(
    role: W4Role,
    cli: AffineCli,
    mut var: impl FnMut(&str) -> Option<String>,
    int8_row: bool,
) -> Result<AffineOpts> {
    let bits = match cli.bits {
        Some(b) => b,
        None => {
            parse_env::<u8>(ENV_BITS, var(ENV_BITS).as_deref())?.unwrap_or(AffineOpts::DEFAULT.bits)
        }
    };
    let common_group = match cli.group {
        Some(g) => g,
        None => parse_env::<u16>(ENV_GROUP, var(ENV_GROUP).as_deref())?
            .unwrap_or(AffineOpts::DEFAULT.group),
    };
    let group = if role == W4Role::Gdn {
        parse_env::<u16>(ENV_GDN_GROUP, var(ENV_GDN_GROUP).as_deref())?.unwrap_or(common_group)
    } else {
        common_group
    };
    let clip_search = match cli.clip_search {
        Some(c) => c,
        None => parse_env::<u8>(ENV_CLIP, var(ENV_CLIP).as_deref())?
            .unwrap_or(AffineOpts::DEFAULT.clip_search),
    };
    supported(bits, group, clip_search).with_context(|| {
        format!(
            "the {} affine route's options (--w4-bits / {ENV_BITS}, --w4-group / {ENV_GROUP}{}, \
             --w4-clip-search / {ENV_CLIP})",
            role.label(),
            if role == W4Role::Gdn {
                format!(", {ENV_GDN_GROUP}")
            } else {
                String::new()
            }
        )
    })?;
    Ok(AffineOpts {
        bits,
        group,
        clip_search,
        int8_prefill: int8_row && bits == 4,
    })
}

static CLI: OnceLock<AffineCli> = OnceLock::new();

/// The flags `port_apply` published, or none (examples, tests, a non-serve entry point).
pub fn cli() -> AffineCli {
    CLI.get().copied().unwrap_or_default()
}

/// Startup check for `port_apply::validate`: every role's options must resolve to a served
/// combination under the flags and the environment. Runs whether or not a route row is on, so a bad
/// value is reported where it was typed rather than when a row is flipped.
pub fn validate(cli: AffineCli) -> Result<()> {
    for role in AFFINE_ROLES {
        resolve(role, cli, |n| std::env::var(n).ok(), false)?;
    }
    Ok(())
}

/// Publish the flags (`port_apply::apply`). The first call wins; a model hot swap does not re-run it.
/// Flags given while no affine row is on are reported once, since they then change nothing.
pub fn apply(cli: AffineCli) {
    if CLI.set(cli).is_err() {
        tracing::debug!("affine W4 options already published; keeping the first set");
        return;
    }
    if cli.any() && !any_route_on() {
        tracing::info!(
            "[avarok] --w4-bits / --w4-group / --w4-clip-search given, but no affine route row is on \
             (AVAROK_DENSE_AFFINE4_FROM_BF16, AVAROK_GDN_AFFINE4_FROM_BF16, \
             AVAROK_ATTN_AFFINE4_FROM_BF16): they change nothing on this serve."
        );
    }
}

/// True when any of the three affine route rows resolves on. Evaluated from the baked declaration and
/// the environment directly, NOT through `resolved_port()`: `apply` runs from `port_apply`, and reading
/// the `OnceLock` there would seal the port table before every lane's `apply` line has run.
pub fn any_route_on() -> bool {
    use crate::layers::ops::port_rows::{
        attn_affine4_from_bf16, dense_affine4_from_bf16, gdn_affine4_from_bf16,
    };
    let d = &avarok_kernels::TARGET_DEFAULTS.port;
    let var = |name: &str| std::env::var(name).ok();
    dense_affine4_from_bf16::resolve(
        d.dense_affine4_from_bf16,
        var(dense_affine4_from_bf16::ENV).as_deref(),
    )
    .value
        || gdn_affine4_from_bf16::resolve(
            d.gdn_affine4_from_bf16,
            var(gdn_affine4_from_bf16::ENV).as_deref(),
        )
        .value
        || attn_affine4_from_bf16::resolve(
            d.attn_affine4_from_bf16,
            var(attn_affine4_from_bf16::ENV).as_deref(),
        )
        .value
}

/// The process-wide options of `role`, resolved once per role from [`cli`], the environment and the
/// `affine4_prefill_int8` row. [`validate`] has refused an unservable value at startup; on a path that
/// skipped it (an example) an error falls back to [`AffineOpts::DEFAULT`] with a warning rather than
/// panicking inside the `OnceLock`. The LM head has no affine route; asked for it, this answers the
/// common (FFN) options.
pub fn opts_for(role: W4Role) -> AffineOpts {
    static OPTS: [OnceLock<AffineOpts>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    let (slot, role) = match role {
        W4Role::Gdn => (&OPTS[1], W4Role::Gdn),
        W4Role::Attn => (&OPTS[2], W4Role::Attn),
        _ => (&OPTS[0], W4Role::Ffn),
    };
    *slot.get_or_init(|| {
        let int8 = crate::layers::ops::port_rows::affine4_prefill_int8::row();
        resolve(role, cli(), |n| std::env::var(n).ok(), int8).unwrap_or_else(|e| {
            tracing::warn!(
                "[avarok] {e:#}; the {} affine route uses the defaults",
                role.label()
            );
            AffineOpts {
                int8_prefill: int8,
                ..AffineOpts::DEFAULT
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl FnMut(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |n| m.get(n).cloned()
    }

    fn cli(bits: Option<u8>, group: Option<u16>, clip: Option<u8>) -> AffineCli {
        AffineCli {
            bits,
            group,
            clip_search: clip,
        }
    }

    #[test]
    fn nothing_given_resolves_to_the_default() {
        for role in AFFINE_ROLES {
            assert_eq!(
                resolve(role, AffineCli::default(), env(&[]), false).unwrap(),
                AffineOpts::DEFAULT
            );
        }
        assert_eq!(
            AffineOpts::DEFAULT.describe(),
            "bits=4 group=64 clip=0 int8_prefill=off"
        );
    }

    #[test]
    fn the_flag_beats_the_environment_which_beats_the_default() {
        let e = [(ENV_BITS, "8"), (ENV_GROUP, "128"), (ENV_CLIP, "3")];
        let from_env = resolve(
            W4Role::Ffn,
            AffineCli::default(),
            env(&[(ENV_GROUP, "128"), (ENV_CLIP, "3")]),
            false,
        )
        .unwrap();
        assert_eq!(
            (from_env.bits, from_env.group, from_env.clip_search),
            (4, 128, 3)
        );
        let flags = resolve(W4Role::Ffn, cli(Some(4), Some(32), Some(0)), env(&e), false).unwrap();
        assert_eq!((flags.bits, flags.group, flags.clip_search), (4, 32, 0));
    }

    #[test]
    fn the_gdn_group_override_reaches_the_gdn_route_only() {
        let e = [(ENV_GDN_GROUP, "32")];
        assert_eq!(
            resolve(W4Role::Gdn, AffineCli::default(), env(&e), false)
                .unwrap()
                .group,
            32
        );
        assert_eq!(
            resolve(W4Role::Ffn, AffineCli::default(), env(&e), false)
                .unwrap()
                .group,
            64
        );
        assert_eq!(
            resolve(W4Role::Attn, AffineCli::default(), env(&e), false)
                .unwrap()
                .group,
            64
        );
        // Without the override the GDN route follows the common group, flag included.
        assert_eq!(
            resolve(W4Role::Gdn, cli(None, Some(128), None), env(&[]), false)
                .unwrap()
                .group,
            128
        );
    }

    #[test]
    fn int8_prefill_follows_the_row_at_four_bits_only() {
        assert!(
            resolve(W4Role::Ffn, AffineCli::default(), env(&[]), true)
                .unwrap()
                .int8_prefill
        );
        assert!(
            !resolve(W4Role::Ffn, AffineCli::default(), env(&[]), false)
                .unwrap()
                .int8_prefill
        );
        let three = resolve(W4Role::Ffn, cli(Some(3), None, None), env(&[]), true).unwrap();
        assert!(
            !three.int8_prefill,
            "the W4A8 prompt GEMM is a 4-bit kernel"
        );
        let w = three.w4();
        assert_eq!(
            (w.bits, w.group, w.clip_search, w.int8_prefill),
            (3, 64, 0, false)
        );
    }

    #[test]
    fn every_served_combination_resolves() {
        for g in GROUPS_4BIT {
            for c in [0, MAX_CLIP_4BIT] {
                supported(4, g, c).unwrap_or_else(|e| panic!("4/{g}/{c}: {e}"));
            }
        }
        for b in WIDTHS {
            for c in [0, MAX_CLIP_WIDTHS] {
                supported(b, WIDTHS_GROUP, c).unwrap_or_else(|e| panic!("{b}/64/{c}: {e}"));
            }
        }
    }

    #[test]
    fn an_unservable_combination_is_refused_with_its_rule() {
        for (bits, group, clip, why) in [
            (4, 16, 0, "group of 32, 64 or 128"),
            (4, 64, MAX_CLIP_4BIT + 1, "clip-search"),
            (3, 128, 0, "group 64 only"),
            (5, 32, 0, "group 64 only"),
            (8, 64, MAX_CLIP_WIDTHS + 1, "clip-search"),
            (7, 64, 0, "2, 3, 4, 5, 6 or 8"),
            (1, 64, 0, "2, 3, 4, 5, 6 or 8"),
            (16, 64, 0, "2, 3, 4, 5, 6 or 8"),
        ] {
            let e = supported(bits, group, clip).expect_err(why).to_string();
            assert!(e.contains(why), "{bits}/{group}/{clip}: {e}");
        }
        // Through resolve, the error names the flags and variables to fix.
        let e = resolve(
            W4Role::Gdn,
            cli(Some(3), None, None),
            env(&[(ENV_GDN_GROUP, "32")]),
            false,
        )
        .expect_err("3-bit at group 32");
        let text = format!("{e:#}");
        assert!(
            text.contains("--w4-bits") && text.contains(ENV_GDN_GROUP),
            "{text}"
        );
    }

    #[test]
    fn a_malformed_variable_is_an_error_not_the_default() {
        for (name, raw) in [
            (ENV_BITS, "four"),
            (ENV_GROUP, "-64"),
            (ENV_CLIP, "300"),
            (ENV_GDN_GROUP, "x"),
        ] {
            let e = resolve(
                W4Role::Gdn,
                AffineCli::default(),
                env(&[(name, raw)]),
                false,
            )
            .expect_err(name)
            .to_string();
            assert!(e.contains(name), "{e}");
        }
        // A blank variable counts as absent.
        assert_eq!(
            resolve(
                W4Role::Ffn,
                AffineCli::default(),
                env(&[(ENV_BITS, "  ")]),
                false
            )
            .unwrap(),
            AffineOpts::DEFAULT
        );
    }

    #[test]
    fn the_affine_roles_and_the_flag_presence() {
        let names: Vec<_> = AFFINE_ROLES.iter().map(|r| r.label()).collect();
        assert_eq!(names, ["ffn", "gdn", "attn"]);
        assert!(!AffineCli::default().any());
        assert!(cli(None, None, Some(2)).any());
    }
}
