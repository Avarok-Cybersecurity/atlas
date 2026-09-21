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
