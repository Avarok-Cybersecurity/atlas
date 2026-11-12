// SPDX-License-Identifier: AGPL-3.0-only

//! `GET /props`: llama.cpp's server-properties document.
//!
//! Field set and nesting follow `get_res_props` in `tools/server/server-context.cpp`
//! (tag b11583). Clients read `default_generation_settings.n_ctx` to size a
//! prompt, `total_slots` to size a worker pool, and `chat_template` to decide
//! whether to call `/apply-template`.

use serde_json::{Value, json};

/// The server-wide sampling defaults a request that omits a field gets.
/// They come from the model's `generation_config.json` and the CLI, not from
/// llama.cpp's own defaults (`temperature 0.8`, `top_k 40`, ...), so `/props`
/// and `/completion` report and apply the same numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SamplingDefaults {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_n_sigma: f32,
    pub repetition_penalty: f32,
}

/// Everything `/props` reports, gathered so the document is a pure function.
pub(crate) struct PropsInputs<'a> {
    pub model_name: &'a str,
    pub n_ctx: usize,
    pub total_slots: usize,
    pub chat_template: &'a str,
    pub bos_token: &'a str,
    pub eos_token: &'a str,
    pub build_info: &'a str,
    pub vision: bool,
    pub defaults: SamplingDefaults,
}

/// The `/props` body.
pub(crate) fn props_json(i: &PropsInputs<'_>) -> Value {
    let d = i.defaults;
    // Same keys as llama.cpp's `params`. A sampler Atlas does not have is
    // reported at its disabled value, which is what a request that leaves it
    // at the default gets (a non-default is refused with a 400).
    let params = json!({
        "n_predict": -1,
        "seed": -1,
        "temperature": d.temperature,
        "dynatemp_range": 0.0,
        "dynatemp_exponent": 1.0,
        "top_k": d.top_k,
        "top_p": d.top_p,
        "min_p": d.min_p,
        "top_n_sigma": if d.top_n_sigma > 0.0 { json!(d.top_n_sigma) } else { json!(-1.0) },
        "xtc_probability": 0.0,
        "xtc_threshold": 0.1,
        "typical_p": 1.0,
        "repeat_last_n": 64,
        "repeat_penalty": d.repetition_penalty,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "dry_multiplier": 0.0,
        "dry_base": 1.75,
        "dry_allowed_length": 2,
        "dry_penalty_last_n": 64,
        "mirostat": 0,
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1,
        "stop": [],
        "max_tokens": -1,
        "n_keep": 0,
        "n_discard": 0,
        "ignore_eos": false,
        "stream": false,
        "n_probs": 0,
        "min_keep": 0,
        "grammar": "",
        "samplers": ["dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"],
        "timings_per_token": false,
        "post_sampling_probs": false,
    });
    json!({
        "default_generation_settings": {
            "params": params,
            "n_ctx": i.n_ctx,
        },
        "total_slots": i.total_slots,
        "model_alias": i.model_name,
        // Atlas knows its model by the path or repository id it was started
        // with; there is no separate file path to report.
        "model_path": i.model_name,
        "modalities": {"vision": i.vision, "video": false, "audio": false},
        "endpoint_slots": false,
        "endpoint_props": false,
        "endpoint_metrics": true,
        "chat_template": i.chat_template,
        "bos_token": i.bos_token,
        "eos_token": i.eos_token,
        "build_info": i.build_info,
        "is_sleeping": false,
    })
}

/// The build string: llama.cpp reports `b<number>-<hash>`; Atlas reports its
/// crate version.
pub(crate) fn build_info() -> String {
    format!("atlas-{}", env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> PropsInputs<'static> {
        PropsInputs {
            model_name: "Qwen/Qwen3-8B",
            n_ctx: 32768,
            total_slots: 16,
            chat_template: "{{ messages }}",
            bos_token: "",
            eos_token: "<|im_end|>",
            build_info: "atlas-test",
            vision: false,
            defaults: SamplingDefaults {
                temperature: 0.7,
                top_k: 20,
                top_p: 0.8,
                min_p: 0.0,
                top_n_sigma: 0.0,
                repetition_penalty: 1.0,
            },
        }
    }

    #[test]
    fn carries_the_fields_a_llama_cpp_client_reads() {
        let v = props_json(&inputs());
        assert_eq!(v["default_generation_settings"]["n_ctx"], 32768);
        assert_eq!(v["total_slots"], 16);
        assert_eq!(v["chat_template"], "{{ messages }}");
        assert_eq!(v["model_path"], "Qwen/Qwen3-8B");
        assert_eq!(v["model_alias"], "Qwen/Qwen3-8B");
        assert_eq!(v["build_info"], "atlas-test");
        assert_eq!(v["eos_token"], "<|im_end|>");
        assert_eq!(v["modalities"]["vision"], false);
        assert_eq!(v["is_sleeping"], false);
    }

    #[test]
    fn params_report_the_server_defaults_not_llama_cpps() {
        let p = &props_json(&inputs())["default_generation_settings"]["params"];
        assert_eq!(p["temperature"].as_f64().unwrap() as f32, 0.7);
        assert_eq!(p["top_k"], 20);
        assert_eq!(p["top_p"].as_f64().unwrap() as f32, 0.8);
        assert_eq!(p["min_p"].as_f64().unwrap() as f32, 0.0);
        assert_eq!(p["n_predict"], -1);
    }

    #[test]
    fn top_n_sigma_off_is_reported_the_llama_cpp_way() {
        // llama.cpp disables top-n-sigma with -1; Atlas uses 0.0.
        let p = &props_json(&inputs())["default_generation_settings"]["params"];
        assert_eq!(p["top_n_sigma"].as_f64().unwrap(), -1.0);
        let mut on = inputs();
        on.defaults.top_n_sigma = 1.5;
        let p = &props_json(&on)["default_generation_settings"]["params"];
        assert_eq!(p["top_n_sigma"].as_f64().unwrap() as f32, 1.5);
    }

    #[test]
    fn vision_models_say_so() {
        let mut i = inputs();
        i.vision = true;
        assert_eq!(props_json(&i)["modalities"]["vision"], true);
    }
}
