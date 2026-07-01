//! Convert parsed SCgf definitions (from the [`scgf`] crate) into plyphon [`SynthDef`]s.
//!
//! SC models a SynthDef's named parameters as `Control`-family UGens whose outputs feed the rest of
//! the graph; plyphon handles parameters directly (see [`crate::controller::Controller::set_control`]), so
//! this converter folds those control UGens into [`Param`]s and rewrites inputs that referenced them
//! into [`InputRef::Param`]. The remaining UGens are renumbered and emitted as a plyphon
//! [`SynthDef`], carrying their calculation rate (so audio- and control-rate UGens both load).

use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::{HashMap, HashSet};

use thiserror::Error;

use crate::synthdef::{InputRef, Param, SynthDef, UnitSpec};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;

/// An error loading SynthDefs from SCgf bytes.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ReadError {
    /// The bytes failed to parse as SCgf.
    #[error("invalid SCgf")]
    Scgf(#[from] scgf::Error),
    /// An input references a UGen or constant index that does not exist.
    #[error("input reference out of range")]
    BadInputRef,
}

/// Parse SCgf bytes into plyphon [`SynthDef`]s (a file may contain several), each with its
/// `(reblock, resample)` graph-rate overrides (scsynth's `Reblock`/`Resample`, version-3 fields).
/// The authored `SynthDef` has no rate field, so the overrides ride alongside, for the caller to
/// hand to [`Controller::add_synthdef_rate`](crate::Controller::add_synthdef_rate).
pub fn parse(data: &[u8]) -> Result<Vec<(SynthDef, Option<usize>, usize)>, ReadError> {
    let file = scgf::parse(data)?;
    file.defs
        .iter()
        .map(|def| Ok((convert(def)?, reblock_of(def), resample_of(def))))
        .collect()
}

/// The reblock block size a parsed def requests (scsynth's `Reblock`): `0` -> none, `N > 0` -> a fixed
/// block. A control-driven block size (`-1`) is unsupported - plyphon bakes the graph block at compile
/// - so it falls back to none.
fn reblock_of(def: &scgf::SynthDef) -> Option<usize> {
    usize::try_from(def.block_size).ok().filter(|&b| b > 0)
}

/// The oversample factor a parsed def requests (scsynth's `Resample`): `<= 1.0` -> 1 (none), `> 1.0`
/// -> the rounded factor. A control-driven factor (`-1.0`) likewise falls back to 1.
fn resample_of(def: &scgf::SynthDef) -> usize {
    if def.resample_factor > 1.0 {
        // `round()` is not available in `no_std` (the wasm target); round-half-up with `floor`
        // (the factor is `> 1.0` here).
        math::floor(def.resample_factor + 0.5) as usize
    } else {
        1
    }
}

/// Whether `name` is a parameter UGen that plyphon folds into [`Param`]s. `AudioControl` produces
/// audio-rate parameters; the rest are control-rate.
fn is_control(name: &str) -> bool {
    matches!(
        name,
        "Control" | "TrigControl" | "LagControl" | "AudioControl"
    )
}

fn rate(rate: scgf::Rate) -> Rate {
    match rate {
        scgf::Rate::Scalar => Rate::Scalar,
        scgf::Rate::Control => Rate::Control,
        scgf::Rate::Audio => Rate::Audio,
        scgf::Rate::Demand => Rate::Demand,
    }
}

fn convert(def: &scgf::SynthDef) -> Result<SynthDef, ReadError> {
    // Map each control UGen output to its parameter index (and capture the output's rate, so an
    // `AudioControl`'s audio-rate outputs become audio-rate params), and renumber the surviving UGens.
    let mut param_of: HashMap<(u32, u32), u32> = HashMap::new();
    let mut param_rate: HashMap<u32, Rate> = HashMap::new();
    let mut param_trig: HashSet<u32> = HashSet::new();
    let mut param_lag: HashMap<u32, f32> = HashMap::new();
    let mut remap: Vec<Option<u32>> = vec![None; def.ugens.len()];
    let mut next = 0u32;
    for (i, ugen) in def.ugens.iter().enumerate() {
        if is_control(&ugen.name) {
            for (output, &out_rate) in ugen.outputs.iter().enumerate() {
                let param = ugen.special_index as i64 + output as i64;
                if param >= 0 {
                    param_of.insert((i as u32, output as u32), param as u32);
                    param_rate.insert(param as u32, rate(out_rate));
                    if ugen.name == "TrigControl" {
                        param_trig.insert(param as u32);
                    }
                    // A `LagControl`'s per-output lag time is the corresponding (constant) UGen input.
                    if ugen.name == "LagControl"
                        && let Some(scgf::Input::Constant { index }) = ugen.inputs.get(output)
                        && let Some(&lag) = def.constants.get(*index as usize)
                    {
                        param_lag.insert(param as u32, lag);
                    }
                }
            }
        } else {
            remap[i] = Some(next);
            next += 1;
        }
    }

    // Parameters: defaults from the value array, names attached by index, rate from the folded UGen.
    let mut params: Vec<Param> = def
        .param_values
        .iter()
        .enumerate()
        .map(|(i, &default)| Param {
            name: String::new(),
            default,
            rate: param_rate
                .get(&(i as u32))
                .copied()
                .unwrap_or(Rate::Control),
            is_trig: param_trig.contains(&(i as u32)),
            lag: param_lag.get(&(i as u32)).copied(),
        })
        .collect();
    for named in &def.param_names {
        if let Some(param) = params.get_mut(named.index as usize) {
            param.name = named.name.clone();
        }
    }
    for (i, param) in params.iter_mut().enumerate() {
        if param.name.is_empty() {
            param.name = format!("param{i}");
        }
    }

    let mut units = Vec::with_capacity(next as usize);
    for ugen in &def.ugens {
        if is_control(&ugen.name) {
            continue;
        }
        let mut inputs = Vec::with_capacity(ugen.inputs.len());
        for input in &ugen.inputs {
            let input = match *input {
                scgf::Input::Constant { index } => {
                    let value = *def
                        .constants
                        .get(index as usize)
                        .ok_or(ReadError::BadInputRef)?;
                    InputRef::Constant(value)
                }
                scgf::Input::Ugen { ugen, output } => {
                    if let Some(&param) = param_of.get(&(ugen, output)) {
                        InputRef::Param(param)
                    } else {
                        let unit = remap
                            .get(ugen as usize)
                            .copied()
                            .flatten()
                            .ok_or(ReadError::BadInputRef)?;
                        InputRef::Unit { unit, output }
                    }
                }
            };
            inputs.push(input);
        }
        units.push(UnitSpec {
            name: ugen.name.clone(),
            rate: rate(ugen.rate),
            inputs,
            num_outputs: ugen.outputs.len(),
            special_index: ugen.special_index,
        });
    }

    Ok(SynthDef {
        name: def.name.clone(),
        params,
        units,
    })
}
