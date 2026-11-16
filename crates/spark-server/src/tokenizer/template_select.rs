// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the operator's chat-template choice (`--chat-template`,
//! `AVAROK_CHAT_TEMPLATE`). The per-request `terse` knob the Sharp template
//! reads lives in `terse_render.rs`.
//!
//! Default `model`: nothing here runs and the prompt bytes are exactly what
//! they were before this module existed. A non-default choice replaces the
//! template of the one `ChatTokenizer` built from the model directory:
//!
//! * `sharp`: the embedded Sharp chat template (see
//!   `SHARP-CHAT-TEMPLATE-ATTRIBUTION.md`), terse system block on;
//! * `sharp-no-terse`: the same template with `terse` pinned off (retention of
//!   history reasoning and the tool-format fixes stay; the terseness system
//!   prompt goes), so an A/B can separate the two effects;
//! * any other value: a Jinja file read from that path.
//!
//! Precedence, highest first: this choice, then `jinja-templates/{model_type}.jinja`,
//! then the model's own template. A selected template also drops the OpenAI
//! variant environment (`jinja-templates/openai/*.jinja`), which would otherwise
//! shadow it in the request path.
//!
//! This file is compiled into the library crate too, so it must not name
//! bin-only modules (`crate::cli`, `crate::ir`, `crate::api`). The strict
//! parse is exposed as plain functions the bin-side `port_apply` calls.

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};

use super::{ChatEncoding, ChatTokenizer, jinja_helpers};

/// The environment variable that mirrors `--chat-template`.
pub const ENV_KEY: &str = "AVAROK_CHAT_TEMPLATE";
/// The flag name, for error messages.
pub const FLAG_NAME: &str = "--chat-template";
/// `template_version` the embedded Sharp template declares on its first line.
pub const SHARP_VERSION: &str = "qwen3.8-froggeric-v22.5.0";

/// The embedded template, byte for byte the upstream file plus one leading
/// Jinja comment. Embedded so the choice never depends on the working directory.
const SHARP_SOURCE: &str = include_str!("../../assets/chat_templates/qwen-sharp-v22.5.0.jinja");
/// Prepended for `sharp-no-terse`: the template reads `terse if terse is defined
/// else true`, and a template-level `set` shadows the render context, so the
/// per-request `terse` kwarg cannot turn it back on for this choice.
const NO_TERSE_PREFIX: &str = "{%- set terse = false %}\n";

/// Which chat template the server renders prompts with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplateChoice {
    /// The model's own template (or the `jinja-templates/` override): today's behavior.
    Model,
    /// The embedded Sharp template, terse system block on by default.
    Sharp,
    /// The embedded Sharp template with `terse` pinned off.
    SharpNoTerse,
    /// A Jinja file read from this path.
    File(PathBuf),
}

impl ChatTemplateChoice {
    /// Parse one flag or environment value. `None` for an empty (or all
    /// whitespace) value, which means "not set". Any word that is not a keyword
    /// is a path; [`resolve`] then insists the path is a readable file, so a
    /// typo such as `shrap` fails startup instead of being ignored.
    pub fn parse(raw: &str) -> Option<Self> {
        let raw = raw.trim();
        match raw {
            "" => None,
            "model" => Some(Self::Model),
            "sharp" => Some(Self::Sharp),
            "sharp-no-terse" => Some(Self::SharpNoTerse),
            path => Some(Self::File(PathBuf::from(path))),
        }
    }

    /// One short phrase for the boot log.
    pub fn describe(&self) -> String {
        match self {
            Self::Model => "model".to_string(),
            Self::Sharp => format!("sharp ({SHARP_VERSION})"),
            Self::SharpNoTerse => format!("sharp-no-terse ({SHARP_VERSION})"),
            Self::File(path) => format!("file {}", path.display()),
        }
    }

    fn is_sharp(&self) -> bool {
        matches!(self, Self::Sharp | Self::SharpNoTerse)
    }
}

/// Strict resolution of the flag and the environment value. The flag wins over
/// the environment; an empty value counts as unset; a value that is neither a
/// keyword nor a readable, non-empty file is an error naming where it came from.
pub fn resolve(flag: Option<&str>, env_value: Option<&str>) -> Result<Option<ChatTemplateChoice>> {
    fn given(v: Option<&str>) -> Option<&str> {
        v.map(str::trim).filter(|v| !v.is_empty())
    }
    let (origin, raw) = match (given(flag), given(env_value)) {
        (Some(raw), _) => (FLAG_NAME, raw),
        (None, Some(raw)) => (ENV_KEY, raw),
        (None, None) => return Ok(None),
    };
    let choice = ChatTemplateChoice::parse(raw).expect("a non-empty value always parses");
    if let ChatTemplateChoice::File(path) = &choice {
        let text = std::fs::read_to_string(path).map_err(|e| {
            anyhow::anyhow!(
                "{origin} {raw:?}: not one of sharp, sharp-no-terse, model, and not a readable \
                 template file ({e})"
            )
        })?;
        if text.trim().is_empty() {
            bail!("{origin} {raw:?}: the template file is empty");
        }
    }
    Ok(Some(choice))
}

/// [`resolve`] with the environment read from [`ENV_KEY`]. A value that is not
/// valid Unicode is an error rather than "unset".
pub fn resolve_from_env(flag: Option<&str>) -> Result<Option<ChatTemplateChoice>> {
    let env_value = match std::env::var(ENV_KEY) {
        Ok(v) => Some(v),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => bail!("{ENV_KEY} is not valid Unicode"),
    };
    resolve(flag, env_value.as_deref())
}

/// Startup validation for `port_apply::validate`: fail on a bad value before
/// any model is built.
pub fn validate(flag: Option<&str>) -> Result<()> {
    resolve_from_env(flag).map(|_| ())
}

/// Startup publication for `port_apply::apply`: resolve and store the choice
/// that [`apply_selected`] reads once the tokenizer exists.
pub fn apply_cli(flag: Option<&str>) -> Result<()> {
    if let Some(choice) = resolve_from_env(flag)? {
        set_choice(choice);
    }
    Ok(())
}

static CHOICE: Mutex<Option<ChatTemplateChoice>> = Mutex::new(None);

/// Store the process-wide choice. A later call replaces it.
pub fn set_choice(choice: ChatTemplateChoice) {
    *CHOICE.lock().unwrap_or_else(|p| p.into_inner()) = Some(choice);
}

/// The stored choice, if one was set.
pub fn selected_choice() -> Option<ChatTemplateChoice> {
    CHOICE.lock().unwrap_or_else(|p| p.into_inner()).clone()
}

/// Apply the stored choice to the tokenizer built from the model directory.
/// With no choice stored (or `model`) the tokenizer comes back untouched.
pub fn apply_selected(tokenizer: ChatTokenizer) -> Result<ChatTokenizer> {
    match selected_choice() {
        Some(choice) => apply_choice(tokenizer, &choice),
        None => Ok(tokenizer),
    }
}

/// The Jinja source `choice` renders with, before the Python-to-minijinja text
/// rewrites. `None` for `Model`, which keeps whatever the tokenizer loaded.
pub(crate) fn template_source(choice: &ChatTemplateChoice) -> Result<Option<String>> {
    Ok(Some(match choice {
        ChatTemplateChoice::Model => return Ok(None),
        ChatTemplateChoice::Sharp => SHARP_SOURCE.to_string(),
        ChatTemplateChoice::SharpNoTerse => format!("{NO_TERSE_PREFIX}{SHARP_SOURCE}"),
        ChatTemplateChoice::File(path) => std::fs::read_to_string(path)
            .with_context(|| format!("{FLAG_NAME}: cannot read {}", path.display()))?,
    }))
}

/// Replace the tokenizer's template with `choice`'s. `Model` is a no-op.
///
/// Sets `native_qwen_tool_template` when the new template declares `tools` and
/// renders the Qwen XML tool format itself (always true for Sharp, and for a
/// file that contains `<function=`), so the parser's own tool prompt is not
/// injected a second time. Models whose chat is not Jinja-encoded (DeepSeek
/// V4, Kimi K3) cannot take a template and fail startup instead of silently
/// ignoring the flag.
pub fn apply_choice(
    mut tokenizer: ChatTokenizer,
    choice: &ChatTemplateChoice,
) -> Result<ChatTokenizer> {
    let Some(source) = template_source(choice)? else {
        return Ok(tokenizer);
    };
    if tokenizer.chat_encoding != ChatEncoding::Jinja {
        bail!(
            "{FLAG_NAME} {}: this model uses a checkpoint-native chat encoding, which a Jinja \
             template cannot replace",
            choice.describe()
        );
    }
    let converted = jinja_helpers::convert_python_jinja_to_minijinja(&source);
    let env = jinja_helpers::build_jinja_env(&converted).with_context(|| {
        format!(
            "{FLAG_NAME} {}: the template does not compile",
            choice.describe()
        )
    })?;
    let declares_tools = env
        .get_template("chat")?
        .undeclared_variables(false)
        .contains("tools");
    tokenizer.native_qwen_tool_template =
        declares_tools && (choice.is_sharp() || converted.contains("<function="));
    tokenizer.chat_template = converted;
    tokenizer.jinja_env = env;
    tokenizer.openai_jinja_env = None;
    tracing::info!("chat template: {}", choice.describe());
    Ok(tokenizer)
}

impl ChatTokenizer {
    /// [`ChatTokenizer::from_model_dir`] followed by [`apply_choice`]; the
    /// serve path stores the choice and calls [`apply_selected`] instead.
    #[allow(clippy::too_many_arguments)]
    pub fn from_model_dir_with_template(
        model_dir: &std::path::Path,
        eos_token_id: u32,
        supports_thinking: bool,
        model_type: &str,
        repo_root: Option<&std::path::Path>,
        disable_template_overrides: bool,
        choice: &ChatTemplateChoice,
    ) -> Result<Self> {
        let tokenizer = Self::from_model_dir(
            model_dir,
            eos_token_id,
            supports_thinking,
            model_type,
            repo_root,
            disable_template_overrides,
        )?;
        apply_choice(tokenizer, choice)
    }
}
