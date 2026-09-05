// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded synchronous work dispatch for selected grammar masks.

pub(super) fn for_each_bounded<T: Sync>(
    items: &[T],
    max_threads: usize,
    work: &(impl Fn(&T) + Sync),
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    assert!(max_threads > 0);
    let workers = max_threads.min(items.len());
    if workers <= 1 {
        items.iter().for_each(work);
        return;
    }
    let next = AtomicUsize::new(0);
    let run = || {
        while let Some(item) = items.get(next.fetch_add(1, Ordering::Relaxed)) {
            work(item);
        }
    };
    std::thread::scope(|scope| {
        for _ in 1..workers {
            // Under host thread pressure, finish the same work with fewer
            // workers. The caller also drains the queue; no job is dropped.
            if std::thread::Builder::new()
                .spawn_scoped(scope, run)
                .is_err()
            {
                break;
            }
        }
        run();
    }); // All selected masks are complete before the caller can sample.
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex};
    use std::time::Duration;

    #[test]
    fn bounded_workers_overlap_and_complete_each_job_once() {
        let seen: Vec<_> = (0..16).map(|_| AtomicUsize::new(0)).collect();
        let entered = (Mutex::new(0usize), Condvar::new());
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        for_each_bounded(&(0..16).collect::<Vec<_>>(), 4, &|&index| {
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            let mut count = entered.0.lock().unwrap();
            *count += 1;
            entered.1.notify_all();
            // Only the first wave waits. A serial implementation times out
            // and fails the overlap oracle rather than hanging the suite.
            if *count < 4 {
                let _ = entered
                    .1
                    .wait_timeout_while(count, Duration::from_secs(1), |n| *n < 4)
                    .unwrap();
            }
            seen[index].fetch_add(1, Ordering::SeqCst);
            active.fetch_sub(1, Ordering::SeqCst);
        });
        assert!(
            peak.load(Ordering::SeqCst) > 1,
            "selected masks are still computed serially"
        );
        assert!(peak.load(Ordering::SeqCst) <= 4, "worker bound exceeded");
        assert_eq!(
            active.load(Ordering::SeqCst),
            0,
            "returned before joining workers"
        );
        assert!(seen.iter().all(|n| n.load(Ordering::SeqCst) == 1));
    }

    #[test]
    fn one_worker_keeps_serial_order_and_empty_work_does_nothing() {
        let seen = Mutex::new(Vec::new());
        for_each_bounded(&[3, 2, 1], 1, &|n| seen.lock().unwrap().push(*n));
        assert_eq!(*seen.lock().unwrap(), [3, 2, 1]);
        for_each_bounded::<usize>(&[], 4, &|_| panic!("empty work executed"));
    }
}
