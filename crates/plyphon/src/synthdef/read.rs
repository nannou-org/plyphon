//! Convert parsed SCgf definitions (from the [`scgf`] crate) into plyphon [`SynthDef`]s.
//!
//! SC models a SynthDef's named parameters as `Control`-family UGens whose outputs feed the rest of
//! the graph; plyphon handles parameters directly (see [`crate::controller::Controller::set_control`]), so
//! this converter folds those control UGens into [`Param`]s and rewrites inputs that referenced them
//! into [`InputRef::Param`]. The remaining UGens are renumbered and emitted as a plyphon
//! [`SynthDef`], carrying their calculation rate (so audio- and control-rate UGens both load).

use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::HashMap;

use thiserror::Error;

use crate::synthdef::{InputRef, Param, SynthDef, UnitSpec};
use plyphon_dsp::rate::Rate;

/// An error loading SynthDefs from SCgf bytes.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ReadError {
    /// The bytes failed to parse as SCgf.
    #[error("invalid SCgf")]
    Scgf(#[from] scgf::Error),
    /// A rate plyphon does not yet support (e.g. demand rate) was used.
    #[error("unsupported calculation rate")]
    UnsupportedRate,
    /// An input references a UGen or constant index that does not exist.
    #[error("input reference out of range")]
    BadInputRef,
}

/// Parse SCgf bytes into plyphon [`SynthDef`]s (a file may contain several).
pub fn parse(data: &[u8]) -> Result<Vec<SynthDef>, ReadError> {
    let file = scgf::parse(data)?;
    file.defs.iter().map(convert).collect()
}

/// Whether `name` is a control-rate parameter UGen that plyphon folds into [`Param`]s.
fn is_control(name: &str) -> bool {
    matches!(name, "Control" | "TrigControl" | "LagControl")
}

fn rate(rate: scgf::Rate) -> Result<Rate, ReadError> {
    match rate {
        scgf::Rate::Scalar => Ok(Rate::Scalar),
        scgf::Rate::Control => Ok(Rate::Control),
        scgf::Rate::Audio => Ok(Rate::Audio),
        scgf::Rate::Demand => Err(ReadError::UnsupportedRate),
    }
}

fn convert(def: &scgf::SynthDef) -> Result<SynthDef, ReadError> {
    // Map each control UGen output to its parameter index, and renumber the surviving UGens.
    let mut param_of: HashMap<(u32, u32), u32> = HashMap::new();
    let mut remap: Vec<Option<u32>> = vec![None; def.ugens.len()];
    let mut next = 0u32;
    for (i, ugen) in def.ugens.iter().enumerate() {
        if is_control(&ugen.name) {
            for output in 0..ugen.outputs.len() {
                let param = ugen.special_index as i64 + output as i64;
                if param >= 0 {
                    param_of.insert((i as u32, output as u32), param as u32);
                }
            }
        } else {
            remap[i] = Some(next);
            next += 1;
        }
    }

    // Parameters: defaults from the value array, names attached by index.
    let mut params: Vec<Param> = def
        .param_values
        .iter()
        .map(|&default| Param {
            name: String::new(),
            default,
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
            rate: rate(ugen.rate)?,
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
