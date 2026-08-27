// SPDX-License-Identifier: AGPL-3.0-only

//! Plugin provenance — who wrote this, where it came from, where to report it.
//!
//! This lives on [`crate::Plugin`], not on the benchmark, because it answers a
//! question about the *plugin* rather than about a run: once third-party
//! plugins exist, "who am I about to run and is it official?" is the first
//! thing anyone will want, and a benchmark-only field would have to be
//! retrofitted onto every other plugin kind.
//!
//! Everything is `&'static str` so a plugin's identity is fixed at compile
//! time — a plugin cannot present one author on the list screen and another in
//! its detail pane.

/// Authorship and support links for one plugin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PluginMetadata {
    /// One line. What the plugin is, for the detail pane.
    pub description: &'static str,
    /// Semver. First-party plugins track the crate version.
    pub version: &'static str,
    pub author: &'static str,
    /// The author's home on the web. Empty when there is none.
    pub author_url: &'static str,
    pub email: &'static str,
    pub repository: &'static str,
    /// Documentation for this plugin specifically.
    pub help_url: &'static str,
    pub bug_report_url: &'static str,
    pub license: &'static str,
    /// True only for plugins shipped inside Atlas itself.
    ///
    /// The badge this drives is a trust signal, so it is deliberately not
    /// something a plugin sets to `true` by writing a nice-looking string —
    /// first-party plugins get it from [`PluginMetadata::atlas`], and anything
    /// built with [`PluginMetadata::third_party`] cannot set it at all.
    pub official: bool,
}

impl PluginMetadata {
    /// A first-party Atlas plugin. Every field except the description is the
    /// same for all of them, so this is the one place they are written.
    pub const fn atlas(description: &'static str) -> Self {
        Self {
            description,
            version: env!("CARGO_PKG_VERSION"),
            author: "Avarok Cybersecurity",
            author_url: "https://atlasinference.io",
            email: "support@avarok.net",
            repository: "https://github.com/Avarok-Cybersecurity/atlas",
            help_url: "https://docs.atlasinference.io/benchmarks",
            bug_report_url: "https://github.com/Avarok-Cybersecurity/atlas/issues/new",
            license: "AGPL-3.0-only",
            official: true,
        }
    }

    /// A plugin from outside the Atlas tree. `official` is forced false.
    #[allow(clippy::too_many_arguments)]
    pub const fn third_party(
        description: &'static str,
        version: &'static str,
        author: &'static str,
        author_url: &'static str,
        email: &'static str,
        repository: &'static str,
        help_url: &'static str,
        bug_report_url: &'static str,
        license: &'static str,
    ) -> Self {
        Self {
            description,
            version,
            author,
            author_url,
            email,
            repository,
            help_url,
            bug_report_url,
            license,
            official: false,
        }
    }

    /// Label/value pairs for the detail pane, skipping the empty ones so a
    /// plugin that has no email does not render a blank row.
    pub fn rows(&self) -> Vec<(&'static str, &'static str)> {
        [
            ("Author", self.author),
            ("Website", self.author_url),
            ("Contact", self.email),
            ("Repository", self.repository),
            ("Docs", self.help_url),
            ("Report a bug", self.bug_report_url),
            ("License", self.license),
        ]
        .into_iter()
        .filter(|(_, v)| !v.is_empty())
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_party_metadata_tracks_the_crate_version() {
        let m = PluginMetadata::atlas("a benchmark");
        assert_eq!(
            m,
            PluginMetadata {
                description: "a benchmark",
                version: env!("CARGO_PKG_VERSION"),
                author: "Avarok Cybersecurity",
                author_url: "https://atlasinference.io",
                email: "support@avarok.net",
                repository: "https://github.com/Avarok-Cybersecurity/atlas",
                help_url: "https://docs.atlasinference.io/benchmarks",
                bug_report_url: "https://github.com/Avarok-Cybersecurity/atlas/issues/new",
                license: "AGPL-3.0-only",
                official: true,
            }
        );
    }

    #[test]
    fn third_party_cannot_claim_to_be_official() {
        let m = PluginMetadata::third_party(
            "community sweep",
            "0.2.1",
            "Someone",
            "",
            "",
            "https://example.invalid/repo",
            "",
            "",
            "MIT",
        );
        assert!(!m.official);
    }

    #[test]
    fn empty_fields_are_not_rendered_as_blank_rows() {
        let m = PluginMetadata::third_party("d", "1", "A", "", "", "r", "", "", "MIT");
        assert_eq!(
            m.rows(),
            [("Author", "A"), ("Repository", "r"), ("License", "MIT")]
        );
    }
}
