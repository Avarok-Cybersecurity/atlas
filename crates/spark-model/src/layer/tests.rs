// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn test_empty_layer_state_downcast() {
    let state: Box<dyn LayerState> = Box::new(EmptyLayerState);
    assert!(state.as_any().downcast_ref::<EmptyLayerState>().is_some());
    assert!(state.as_any().downcast_ref::<SsmLayerState>().is_none());
}

#[test]
fn test_ssm_layer_state_downcast() {
    let state: Box<dyn LayerState> = Box::new(SsmLayerState {
        h_state: DevicePtr(0x1000),
        conv_state: DevicePtr(0x2000),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        conv_state_intermediates: Vec::new(),
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    });
    let ssm = state.as_any().downcast_ref::<SsmLayerState>().unwrap();
    assert_eq!(ssm.h_state.0, 0x1000);
    assert_eq!(ssm.conv_state.0, 0x2000);
}

#[test]
fn test_ssm_layer_state_mut() {
    let mut state: Box<dyn LayerState> = Box::new(SsmLayerState {
        h_state: DevicePtr(0x1000),
        conv_state: DevicePtr(0x2000),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        conv_state_intermediates: Vec::new(),
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    });
    let ssm = state.as_any_mut().downcast_mut::<SsmLayerState>().unwrap();
    ssm.h_state = DevicePtr(0x3000);
    assert_eq!(ssm.h_state.0, 0x3000);
}

#[test]
fn test_forward_context_lifetime() {
    use spark_runtime::gpu::mock::MockGpuBackend;

    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let buffers = BufferArena::new(&config, 1, 4096, 16, 1, &gpu).unwrap();

    // A test can now state the dispatch it wants instead of mutating the
    // process environment — the point of carrying it rather than caching it.
    let dispatch = crate::layers::ops::GemmDispatch::defaults();
    let derived = crate::layers::ops::DerivedWeights::new();
    let levers = crate::layers::ops::ModelLevers::defaults();
    let stats = crate::layers::ops::ModelStats::new();
    let ctx = ForwardContext {
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers: &buffers,
        hc_row_offset: 0,
        gpu: &gpu,
        config: &config,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Fold,
    };

    assert_eq!(ctx.config.hidden_size, 2048);
    assert_eq!(ctx.config.num_hidden_layers, 48);
}
