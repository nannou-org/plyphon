//! Encoding a [`SynthDefFile`] back into SCgf bytes (always format version 2).

use alloc::vec::Vec;

use crate::{Error, Input, SynthDef, SynthDefFile, Ugen};

/// Encode a file as SCgf (version 2).
pub fn encode(file: &SynthDefFile) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    out.extend_from_slice(b"SCgf");
    out.extend_from_slice(&2i32.to_be_bytes());
    out.extend_from_slice(&(file.defs.len() as i16).to_be_bytes());
    for def in &file.defs {
        encode_def(&mut out, def)?;
    }
    Ok(out)
}

fn encode_def(out: &mut Vec<u8>, def: &SynthDef) -> Result<(), Error> {
    write_pstring(out, &def.name)?;

    out.extend_from_slice(&(def.constants.len() as i32).to_be_bytes());
    for &c in &def.constants {
        out.extend_from_slice(&c.to_be_bytes());
    }

    out.extend_from_slice(&(def.param_values.len() as i32).to_be_bytes());
    for &v in &def.param_values {
        out.extend_from_slice(&v.to_be_bytes());
    }

    out.extend_from_slice(&(def.param_names.len() as i32).to_be_bytes());
    for param in &def.param_names {
        write_pstring(out, &param.name)?;
        out.extend_from_slice(&(param.index as i32).to_be_bytes());
    }

    out.extend_from_slice(&(def.ugens.len() as i32).to_be_bytes());
    for ugen in &def.ugens {
        encode_ugen(out, ugen)?;
    }

    out.extend_from_slice(&(def.variants.len() as i16).to_be_bytes());
    for variant in &def.variants {
        write_pstring(out, &variant.name)?;
        for &v in &variant.values {
            out.extend_from_slice(&v.to_be_bytes());
        }
    }
    Ok(())
}

fn encode_ugen(out: &mut Vec<u8>, ugen: &Ugen) -> Result<(), Error> {
    write_pstring(out, &ugen.name)?;
    out.push(ugen.rate.code() as u8);
    out.extend_from_slice(&(ugen.inputs.len() as i32).to_be_bytes());
    out.extend_from_slice(&(ugen.outputs.len() as i32).to_be_bytes());
    out.extend_from_slice(&ugen.special_index.to_be_bytes());
    for input in &ugen.inputs {
        let (from_ugen, from_output) = match *input {
            Input::Constant { index } => (-1i32, index as i32),
            Input::Ugen { ugen, output } => (ugen as i32, output as i32),
        };
        out.extend_from_slice(&from_ugen.to_be_bytes());
        out.extend_from_slice(&from_output.to_be_bytes());
    }
    for rate in &ugen.outputs {
        out.push(rate.code() as u8);
    }
    Ok(())
}

fn write_pstring(out: &mut Vec<u8>, s: &str) -> Result<(), Error> {
    let len = u8::try_from(s.len()).map_err(|_| Error::NameTooLong)?;
    out.push(len);
    out.extend_from_slice(s.as_bytes());
    Ok(())
}
