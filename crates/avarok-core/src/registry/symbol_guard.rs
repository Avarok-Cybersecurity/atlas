// SPDX-License-Identifier: AGPL-3.0-only

//! The registry's binary-module symbol guard.
//!
//! One home for the question "can this lookup possibly succeed?", asked of the
//! code object rather than of the driver. [`BinaryKernels`] is built during
//! [`crate::registry::AvarokRegistry`] init, one entry per binary module whose
//! symbol table could be read, and consulted on every function lookup.

use std::collections::{HashMap, HashSet};

use crate::elf_symbols::defined_function_symbols;
use crate::error::{AvarokError, Result};

/// For every binary (code-object) module whose symbol table could be read,
/// the kernel names that object actually DEFINES.
///
/// The driver is not a reliable oracle for this: SCALE answers
/// `cuModuleGetFunction` with SUCCESS for a name the object does not define,
/// and the launch through that handle is the first thing that notices. See the
/// module docs on [`crate::elf_symbols`]. So the registry asks the object, not
/// the driver, and refuses the lookup itself.
///
/// A module is ABSENT from this table when it is text (PTX, the unchanged
/// path) or when its bytes could not be parsed, and an absent module is not
/// guarded: the driver's answer stands, exactly as before.
pub(super) struct BinaryKernels {
    defined: HashMap<&'static str, HashSet<String>>,
    /// Binary modules seen, parsed or not. Drives the one summary line.
    scanned: usize,
    /// Modules that parsed to an empty kernel set, for that same line.
    empty: Vec<&'static str>,
}

impl BinaryKernels {
    pub(super) fn new() -> Self {
        BinaryKernels {
            defined: HashMap::new(),
            scanned: 0,
            empty: Vec::new(),
        }
    }

    /// Read one binary module's symbol table, while the blob is in hand.
    /// Text (PTX) modules are not offered here: the NVIDIA path never had
    /// this problem.
    pub(super) fn scan(&mut self, name: &'static str, blob: &[u8]) {
        self.scanned += 1;
        match defined_function_symbols(blob) {
            Some(defined) => {
                if defined.is_empty() {
                    self.empty.push(name);
                }
                self.defined.insert(name, defined);
            }
            None => {
                // Not an ELF64 LE image (a clang offload bundle, say).
                // Nothing to guard with, so nothing is guarded: the driver
                // answers as it always did.
                eprintln!(
                    "avarok: WARN: module '{name}': code object carries no readable ELF \
                     symbol table; kernel lookups fall through to the driver"
                );
            }
        }
    }

    /// The one summary line for the whole scan. No-op when the run loaded no
    /// binary modules at all, which is the NVIDIA path.
    pub(super) fn report(&mut self) {
        if self.scanned == 0 {
            return;
        }
        // avarok-core carries no `tracing` (see the note on the registry's
        // `Drop`), so this goes to stderr like every other registry report.
        self.empty.sort_unstable();
        let empty = self
            .empty
            .iter()
            .map(|m| {
                format!(
                    "; module '{m}' defines no kernels on this target \
                     (optional module compiled out)"
                )
            })
            .collect::<String>();
        eprintln!(
            "avarok: {} binary kernel module(s), {} with a readable symbol table{}",
            self.scanned,
            self.defined.len(),
            empty
        );
    }

    /// Refuse a lookup the module's own code object says cannot succeed.
    ///
    /// Only binary modules with a parsed symbol table are checked. Everything
    /// else (PTX, an unparsable object, a module that is not loaded at all)
    /// returns `Ok(())` and the driver decides, as it always did.
    ///
    /// The caller in `spark-model` is `layers::kernel_probe::try_kernel`,
    /// which matches on ANY `Err` and returns `KernelHandle(0)`, so an
    /// optional kernel guarded here degrades to the same "not present" it
    /// degrades to on NVIDIA.
    pub(super) fn reject(&self, module_name: &str, func_name: &str) -> Result<()> {
        if is_undefined(self.defined.get(module_name), func_name) {
            return Err(AvarokError::ModuleLoad(undefined_symbol_message(
                module_name,
                func_name,
            )));
        }
        Ok(())
    }

    /// Drop the parsed symbol tables. They describe modules that are about to
    /// be unloaded.
    pub(super) fn clear(&mut self) {
        self.defined.drain().for_each(drop);
        self.empty.clear();
        self.scanned = 0;
    }
}

/// Whether a lookup is provably doomed: the module is binary, its symbol
/// table parsed, and `func_name` is not in it.
///
/// `None` (a PTX module, an unparsable object, a module that is not loaded)
/// is never a refusal. Split out from `BinaryKernels::reject` so the decision
/// itself is testable on a host with no CUDA context.
pub fn is_undefined(defined: Option<&HashSet<String>>, func_name: &str) -> bool {
    defined.is_some_and(|set| !set.contains(func_name))
}

/// The text of a lookup refused because the module's code object does not
/// define the name. Pure, so the wording is unit-testable with no GPU.
pub fn undefined_symbol_message(module_name: &str, func_name: &str) -> String {
    format!(
        "{module_name}::{func_name}: not defined in this target's code object \
         (optional module compiled out?)"
    )
}
