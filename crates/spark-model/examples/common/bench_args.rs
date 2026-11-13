// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the command-line helpers shared by the L07 examples.
//!
//! The examples take `--name value` or `--name=value` options and bare
//! `--flag`s, and share the benchmark knobs `--rounds`, `--warmups`, `--calls`
//! and `--profile-measured`. No argument parser crate is a dependency of
//! `spark-model`, and the grammar is small enough not to need one.
//!
//! An example that includes this file also declares
//! `#[path = "common/cold_l2.rs"] mod cold_l2;` and the profiler module.
#![allow(dead_code)]

use crate::cold_l2::BenchCfg;
use anyhow::{Result, anyhow};
use std::str::FromStr;

/// The value of `--name value` or `--name=value` in `args`.
pub fn arg_value_in(args: &[String], name: &str) -> Option<String> {
    let eq = format!("{name}=");
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            return it.next().cloned();
        }
        if let Some(v) = a.strip_prefix(&eq) {
            return Some(v.to_string());
        }
    }
    None
}

/// Every value of a repeatable option (`--name a --name=b`), in order.
pub fn all_values_in(args: &[String], name: &str) -> Vec<String> {
    let eq = format!("{name}=");
    let mut out = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            if let Some(v) = it.next() {
                out.push(v.clone());
            }
        } else if let Some(v) = a.strip_prefix(&eq) {
            out.push(v.to_string());
        }
    }
    out
}

/// Every value of a repeatable option on the process command line.
pub fn all_values(name: &str) -> Vec<String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    all_values_in(&args, name)
}

/// The value of `--name` on the process command line.
pub fn arg_value(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    arg_value_in(&args, name)
}

/// Whether the bare flag `name` is on the process command line.
pub fn has_flag(name: &str) -> bool {
    std::env::args().skip(1).any(|a| a == name)
}

/// Parse `--name` into `T`, or return `default` when it is absent.
pub fn arg_parse<T: FromStr>(name: &str, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match arg_value(name) {
        None => Ok(default),
        Some(s) => s
            .parse::<T>()
            .map_err(|e| anyhow!("{name}: cannot parse {s:?}: {e}")),
    }
}

/// A comma-separated list option, e.g. `--families bf16,fp8`.
pub fn list_value_in(args: &[String], name: &str) -> Option<Vec<String>> {
    arg_value_in(args, name).map(|v| {
        v.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    })
}

/// The benchmark configuration: the 5 / 3 / 20 defaults, overridden by
/// `--rounds`, `--warmups`, `--calls`, and `--profile-measured`.
pub fn bench_cfg(stream: u64) -> Result<BenchCfg> {
    let d = BenchCfg::default();
    Ok(BenchCfg {
        rounds: arg_parse("--rounds", d.rounds)?,
        warmups: arg_parse("--warmups", d.warmups)?,
        calls: arg_parse("--calls", d.calls)?,
        stream,
        profile: has_flag("--profile-measured"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn value_forms() {
        let a = v(&["--rounds", "7", "--calls=9", "--hot"]);
        assert_eq!(arg_value_in(&a, "--rounds").as_deref(), Some("7"));
        assert_eq!(arg_value_in(&a, "--calls").as_deref(), Some("9"));
        assert_eq!(arg_value_in(&a, "--warmups"), None);
        // A bare flag has no value.
        assert_eq!(arg_value_in(&a, "--hot"), None);
    }

    #[test]
    fn repeatable_options_collect_in_order() {
        let a = v(&[
            "--route-kernel",
            "a=x:y:w4-gemv",
            "--hot",
            "--route-kernel=b=p:q:fp8-gemv",
        ]);
        assert_eq!(
            all_values_in(&a, "--route-kernel"),
            vec!["a=x:y:w4-gemv", "b=p:q:fp8-gemv"]
        );
        assert!(all_values_in(&a, "--other").is_empty());
    }

    #[test]
    fn a_value_option_at_the_end_has_no_value() {
        assert_eq!(arg_value_in(&v(&["--rounds"]), "--rounds"), None);
    }

    #[test]
    fn lists_split_and_trim() {
        let a = v(&["--families", "bf16, fp8,,nvfp4"]);
        assert_eq!(
            list_value_in(&a, "--families").unwrap(),
            vec!["bf16", "fp8", "nvfp4"]
        );
        assert!(list_value_in(&a, "--routes").is_none());
    }

    #[test]
    fn absent_options_take_the_default() {
        // The test binary's own arguments never contain these.
        assert_eq!(arg_parse("--no-such-option", 5usize).unwrap(), 5);
        assert!(!has_flag("--no-such-flag"));
    }
}
