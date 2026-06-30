//! Loading SynthDefs from `.scsyndef` files (`--load-dir`).
//!
//! SuperCollider's `sclang` compiles SynthDefs to the binary SCgf format an `.scsyndef` file holds
//! and a client ships with `/d_recv`; [`plyphon::synthdef::read::parse`] reads those same bytes. This
//! is the file-system counterpart, adapting scsynth's `-D` (load the default SynthDef library).

use std::path::Path;

use plyphon::Controller;
use plyphon::synthdef::read::parse;

/// Read every `.scsyndef` file in `dir`, parse each as one or more SynthDefs, and register them on
/// `controller`. Returns the number of defs loaded.
pub fn load_dir(controller: &mut Controller, dir: &Path) -> Result<usize, String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("reading {}: {e}", dir.display()))?;
    let mut count = 0;
    for entry in entries {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("scsyndef") {
            continue;
        }
        let bytes = std::fs::read(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        let defs = parse(&bytes).map_err(|e| format!("parsing {}: {e}", path.display()))?;
        for (def, reblock, resample) in defs {
            controller.add_synthdef_rate(def, reblock, resample);
            count += 1;
        }
    }
    Ok(count)
}
