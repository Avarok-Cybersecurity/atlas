// SPDX-License-Identifier: AGPL-3.0-only

//! `spark sync-recipes` — fill the local recipe index without a terminal.
//!
//! The index is what `benchmark run` resolves a recipe id against. Nothing but
//! the TUI Library wrote it, so a headless box was told to "open the TUI
//! Library once to populate it" — advice a CI runner, a container, or a machine
//! reached over ssh cannot act on.

use anyhow::{Context, Result, bail};
use std::sync::atomic::AtomicBool;

/// What a finished refresh means, as a decision separate from the I/O.
///
/// Pure so it can be tested: the interesting behaviour is which outcomes count
/// as a successful sync, and the network call was the only thing standing
/// between that judgement and a test.
///
/// # Errors
/// When the refresh did not actually reach the repository, or reached it and
/// found nothing.
pub fn verdict(
    offline: Option<&str>,
    incomplete: Option<&str>,
    recipes: usize,
    before: usize,
) -> Result<(), String> {
    // `refresh` serves the CACHE when the network fails, annotated with why.
    // Reporting that as a successful sync is how a CI job ends up believing it
    // has a fresh index and then failing to resolve a recipe added yesterday —
    // far from here, with an error that names neither the network nor this
    // command.
    if let Some(why) = offline {
        return Err(format!(
            "could not reach the recipe repository: {why}\n\
             The cache is unchanged ({before} recipe(s)). This command reports \
             failure rather than success-with-stale-data, because a stale index \
             fails later, somewhere less obvious."
        ));
    }
    if recipes == 0 {
        // A reachable repository that returned nothing is not a success either.
        return Err(
            "the repository returned no recipes; refusing to call that a synced index".to_owned(),
        );
    }
    // Reached the repository, got something back, and still could not put it on
    // disk — some files did not arrive, or the write failed. The fetch layer
    // leaves the cache ALONE in both cases, so what is on disk is still the
    // complete thing it was; it is simply not what was just asked for.
    //
    // This must be an error rather than a note. `sync-recipes` runs before the
    // tracing subscriber exists, so anything reported at `warn!` on this path is
    // discarded — the command would print "recipe index written to …" over a
    // file it did not write, which is the success-with-stale-data this command
    // exists to prevent.
    if let Some(why) = incomplete {
        return Err(format!(
            "the recipe index was not written: {why}\n\
             The cache is unchanged ({before} recipe(s)) — nothing was replaced."
        ));
    }
    Ok(())
}

/// Fetch the recipe index and report what landed.
///
/// # Errors
/// If the artifact store cannot be located, or the fetch did not reach the
/// network.
pub fn run() -> Result<()> {
    let store = atlas_plugin::ArtifactStore::discover()
        .context("locating the artifact store that holds the recipe cache")?;
    let root = store.root();

    let before = crate::recipe::fetch::cached(root).recipes.len();
    // Not cancellable: there is no UI to cancel from, and the fetch has its own
    // timeout.
    let index = crate::recipe::fetch::refresh(root, &AtomicBool::new(false));

    // `refresh` serves the CACHE when the network fails, annotated with why.
    // Reporting that as a successful sync is how a CI job ends up believing it
    // has a fresh index and then failing to resolve a recipe added yesterday.
    if let Err(why) = verdict(
        index.offline.as_deref(),
        index.incomplete.as_deref(),
        index.recipes.len(),
        before,
    ) {
        bail!("{why}");
    }

    println!(
        "recipe index written to {}/atlas-recipes/index.json",
        root.display()
    );
    println!(
        "  {} recipe(s), tree {}",
        index.recipes.len(),
        index.tree_sha
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::verdict;

    #[test]
    fn a_real_fetch_with_recipes_is_a_sync() {
        assert!(verdict(None, None, 30, 0).is_ok());
    }

    /// The case this command exists to refuse.
    #[test]
    fn falling_back_to_a_warm_cache_is_not_a_sync() {
        // `refresh` returns the cached index annotated with why the network
        // failed, so the recipe count looks healthy. A command that reported
        // success here would leave CI believing it had fresh data.
        let e = verdict(Some("dns failure"), None, 30, 30).expect_err("must refuse");
        assert!(e.contains("could not reach"), "{e}");
        assert!(
            e.contains("30 recipe(s)"),
            "must say what is actually there: {e}"
        );
        assert!(
            e.contains("unchanged"),
            "must not imply anything was written: {e}"
        );
    }

    #[test]
    fn falling_back_with_no_cache_at_all_is_also_refused() {
        assert!(verdict(Some("timed out"), None, 0, 0).is_err());
    }

    /// A reachable repository that returns nothing is not success either:
    /// writing an empty index over a good one is worse than failing.
    #[test]
    fn an_empty_index_is_refused_even_when_the_network_worked() {
        let e = verdict(None, None, 0, 30).expect_err("must refuse");
        assert!(e.contains("no recipes"), "{e}");
    }

    /// The failure an operator actually hits on a locked-down box: the fetch
    /// worked, the write did not. It used to be a `warn!` that nothing was
    /// listening for, so the command printed a success line naming a file it
    /// had never written.
    #[test]
    fn a_fetch_that_could_not_be_cached_is_not_a_sync() {
        let e = verdict(
            None,
            Some("the index could not be cached: permission denied"),
            30,
            12,
        )
        .expect_err("an uncached fetch is not a synced index");
        assert!(e.contains("permission denied"), "{e}");
        assert!(
            e.contains("unchanged (12 recipe(s))"),
            "the operator needs to know what is still on disk: {e}"
        );
    }

    /// A partial fetch is the dangerous one: it looks like a success, and it is
    /// the case where a complete cache would have been silently shrunk.
    #[test]
    fn a_partial_fetch_is_not_a_sync_even_though_recipes_arrived() {
        let e = verdict(
            None,
            Some("27 of 30 recipe file(s) could not be fetched"),
            3,
            30,
        )
        .expect_err("3 of 30 recipes is not a synced index");
        assert!(e.contains("27 of 30"), "{e}");
    }

    /// Offline is reported ahead of everything else: it is the more basic
    /// failure, and naming a write problem for a fetch that never happened
    /// would send the operator after the wrong thing.
    #[test]
    fn being_offline_is_reported_before_any_write_problem() {
        let e = verdict(Some("dns failure"), Some("could not be cached"), 30, 30)
            .expect_err("must refuse");
        assert!(e.contains("dns failure"), "{e}");
        assert!(!e.contains("could not be cached"), "{e}");
    }
}
