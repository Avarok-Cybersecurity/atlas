// SPDX-License-Identifier: AGPL-3.0-only
//! The L04 routing-gate study, CPU only: `route_gate build|collect|grade|summary|rows|goldens|tokens|fit`.
//! Library code lives in `avarok_spark_bench::route_gate`; this file only forwards the arguments.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(avarok_spark_bench::route_gate::cli::run(&args));
}
