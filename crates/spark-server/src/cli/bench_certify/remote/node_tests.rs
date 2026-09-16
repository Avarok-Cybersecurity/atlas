// SPDX-License-Identifier: AGPL-3.0-only
use super::super::avarokctl::NodeRow;
use super::super::avarokctl::avarokctl_tests::NODE_ROW;
use super::*;

fn row() -> NodeRow {
    serde_json::from_str(NODE_ROW).unwrap()
}

fn wanted<'a>(signers: &'a [String]) -> Wanted<'a> {
    Wanted {
        hardware: "gb10",
        committed_signers: signers,
        anchor: "1a0dc88a8c9083bb956bd84cafa2cccbdb8e6e18",
        min_free_fraction: 0.85,
    }
}

#[test]
fn a_clean_node_is_admitted_with_its_fingerprint_and_build_state() {
    let signers = vec!["a27dbc8ed2fc2a31".to_string()];
    let n = admit(&row(), &wanted(&signers)).expect("admitted");
    assert_eq!(n.addr, "10.10.10.2");
    assert_eq!(n.name, "spark-43fa");
    assert_eq!(n.signer, "a27dbc8ed2fc2a31");
    assert!(n.built && !n.local);
    assert_eq!(n.free_fraction, Some(0.91));
    assert_eq!(n.hardware.gpu, "NVIDIA GB10");
    assert_eq!(n.hardware.driver_major, Some(580));
    assert_eq!(n.hardware.sm_clock_max_mhz, Some(3003.0));
    assert_eq!(n.hardware.mem_total_kb, Some(127_601_452));
    assert_eq!(n.hardware.thermal_alert, Some(false));
    assert_eq!(n.hardware.hottest_chassis_c, Some(65.0));
    assert_eq!(n.hardware.postcheck_valid, None);
    // Another anchor: not built there.
    let mut w = wanted(&signers);
    w.anchor = "0000000000000000000000000000000000000000";
    assert!(!admit(&row(), &w).unwrap().built);
}

/// Every rule refuses on its own, names its reason, and the reasons add up.
#[test]
fn each_admission_rule_refuses_with_its_reason() {
    let signers = vec!["a27dbc8ed2fc2a31".to_string()];
    type Break = Box<dyn Fn(&mut NodeRow)>;
    let cases: Vec<(&str, Break)> = vec![
        (
            "bench is off",
            Box::new(|r| {
                let i = r.info.as_mut().unwrap();
                i.bench_enabled = false;
                i.disabled_reason = Some("no bench.yaml".into());
            }),
        ),
        (
            "box class is h100",
            Box::new(|r| r.info.as_mut().unwrap().hardware_class = Some("h100".into())),
        ),
        (
            "reports no box class",
            Box::new(|r| r.info.as_mut().unwrap().hardware_class = None),
        ),
        (
            "not committed",
            Box::new(|r| r.info.as_mut().unwrap().signer_fp = Some("deadbeef".into())),
        ),
        (
            "no signing identity",
            Box::new(|r| r.info.as_mut().unwrap().signer_fp = None),
        ),
        (
            "it is busy",
            Box::new(|r| {
                let i = r.info.as_mut().unwrap();
                i.busy = true;
                i.busy_reason = Some("spark pid 1".into());
            }),
        ),
        (
            "job(s) queued",
            Box::new(|r| r.info.as_mut().unwrap().queued = 1),
        ),
        (
            "of host memory is free",
            Box::new(|r| {
                r.info.as_mut().unwrap().host_free_fraction =
                    super::super::wire::Metric::Reading { value: 0.5 }
            }),
        ),
        (
            "cannot report free memory",
            Box::new(|r| {
                r.info.as_mut().unwrap().host_free_fraction =
                    super::super::wire::Metric::Unsupported
            }),
        ),
        (
            "free on its bench cache",
            Box::new(|r| {
                r.info.as_mut().unwrap().disk_free_bytes =
                    super::super::wire::Metric::Reading { value: 1.0 }
            }),
        ),
        (
            "reports no GPU",
            Box::new(|r| r.info.as_mut().unwrap().gpu = None),
        ),
    ];
    for (needle, brk) in cases {
        let mut r = row();
        brk(&mut r);
        let e = admit(&r, &wanted(&signers)).expect_err(needle);
        assert!(e.why.contains(needle), "{needle}: {}", e.why);
        assert_eq!(e.addr, "10.10.10.2");
    }
    // Two broken facts: both named.
    let mut r = row();
    let i = r.info.as_mut().unwrap();
    i.busy = true;
    i.hardware_class = Some("h100".into());
    let e = admit(&r, &wanted(&signers)).unwrap_err();
    assert!(
        e.why.contains("busy") && e.why.contains("h100"),
        "{}",
        e.why
    );
    // No info at all: the error avarokctl gave.
    let e = admit(
        &NodeRow {
            node: "10.10.10.9".into(),
            ok: false,
            info: None,
            error: Some(super::super::avarokctl::ErrorObj {
                code: "not_paired".into(),
                message: "peer x is not paired".into(),
                ..Default::default()
            }),
        },
        &wanted(&signers),
    )
    .unwrap_err();
    assert!(e.why.contains("not_paired"), "{}", e.why);
}

#[test]
fn a_node_without_thermal_facts_has_an_undecidable_fingerprint() {
    let mut r = row();
    r.info.as_mut().unwrap().thermal = None;
    let fp = fingerprint_of(r.info.as_ref().unwrap());
    assert_eq!(fp.gpu, "NVIDIA GB10");
    assert_eq!(fp.sm_clock_max_mhz, None);
    assert_eq!(fp.hottest_chassis_c, None);
    assert_eq!(fp.thermal_alert, None);
    // Such a node is still admitted (it can run correctness gates); the
    // scheduler will refuse to spread Speed onto it.
    let signers = vec!["a27dbc8ed2fc2a31".to_string()];
    assert!(admit(&r, &wanted(&signers)).is_ok());
}

#[test]
fn the_local_node_reads_its_own_state() {
    let hw = Hardware {
        gpu: "NVIDIA GB10".into(),
        driver: "580.126.09".into(),
        sm_clock_mhz: None,
        gpu_count: None,
        source: "test".into(),
    };
    let mut st = HardwareState::default();
    st.machine.hostname = Some("dgx1".into());
    st.mem_available_kb = Some(90);
    st.mem_total_kb = Some(100);
    let n = local("sig", &hw, &st);
    assert!(n.local && n.built);
    assert_eq!(n.addr, "local");
    assert_eq!(n.name, "dgx1");
    assert_eq!(n.free_fraction, Some(0.9));
    assert_eq!(n.hardware.mem_total_kb, Some(100));
}
