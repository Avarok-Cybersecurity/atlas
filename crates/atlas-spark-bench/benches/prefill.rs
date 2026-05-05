// SPDX-License-Identifier: AGPL-3.0-only

use std::time::Duration;

use atlas_spark_bench::{
    long_prompt, medium_prompt, require_server, send_streaming, short_prompt, very_long_prompt,
};
use criterion::{Criterion, criterion_group, criterion_main};

fn prefill_ttft(c: &mut Criterion) {
    let url = require_server();
    let mut group = c.benchmark_group("prefill_ttft");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let prompts: Vec<(&str, String)> = vec![
        ("short_20tok", short_prompt().to_string()),
        ("medium_100tok", medium_prompt()),
        ("long_400tok", long_prompt()),
        ("very_long_800tok", very_long_prompt()),
    ];

    for (name, prompt) in &prompts {
        group.bench_function(*name, |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let r = send_streaming(&url, prompt, 10).expect("streaming request failed");
                    total += r.ttft;
                }
                total
            });
        });
    }
    group.finish();
}

criterion_group!(benches, prefill_ttft);
criterion_main!(benches);
