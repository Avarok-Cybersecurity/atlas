// SPDX-License-Identifier: AGPL-3.0-only

//! A panicking forward must become an error. The GPU backend lives outside
//! this catch, so its allocations stay mapped.

use std::panic::UnwindSafe;

use anyhow::{Result, bail};

pub fn retain_weights_on_panic<T>(forward: impl FnOnce() -> Result<T> + UnwindSafe) -> Result<T> {
    match std::panic::catch_unwind(forward) {
        Ok(result) => result,
        Err(_) => bail!("K3 forward panicked; GPU weights retained"),
    }
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;

    use avarok_core::kimi_k3::ops::matvec;

    use super::*;

    #[test]
    fn bad_matvec_does_not_drop_the_weight_owner() {
        let live = vec![0xA11C_u64];
        let result = retain_weights_on_panic(AssertUnwindSafe(|| {
            let _ = matvec(&[0.0; 3_211_264], &[0.0; 896], 3584, 7168);
            Ok(())
        }));
        let err = result.expect_err("panic becomes Result");
        assert!(format!("{err:#}").contains("GPU weights retained"));
        assert_eq!(live, vec![0xA11C_u64]);
    }
}
