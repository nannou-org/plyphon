//! Encoding a [`SynthDefFile`] back into SCgf bytes - version 2, or version 3 when a def carries
//! reblock/resample settings.

use alloc::vec::Vec;

use crate::{Error, Input, SynthDef, SynthDefFile, Ugen};

/// Encode a file as SCgf. Writes version 2 for ordinary defs (byte-compatible with before); if any def
/// has a non-default reblock/resample setting, writes version 3 (each def prefixed by its int32 size,
/// the reblock/resample fields appended after the variants).
pub fn encode(file: &SynthDefFile) -> Result<Vec<u8>, Error> {
    let v3 = file.defs.iter().any(needs_v3);
    let mut out = Vec::new();
    out.extend_from_slice(b"SCgf");
    out.extend_from_slice(&(if v3 { 3i32 } else { 2 }).to_be_bytes());
    out.extend_from_slice(&(file.defs.len() as i16).to_be_bytes());
    for def in &file.defs {
        if v3 {
            // Encode the def body (with the v3 tail), then prefix it with its total size (body + the
            // 4-byte size field), as scsynth's v3 framing requires.
            let mut body = Vec::new();
            encode_def(&mut body, def, true)?;
            out.extend_from_slice(&((body.len() + 4) as i32).to_be_bytes());
            out.extend_from_slice(&body);
        } else {
            encode_def(&mut out, def, false)?;
        }
    }
    Ok(out)
}

/// Whether a def carries any non-default reblock/resample setting (so the file must be version 3).
fn needs_v3(def: &SynthDef) -> bool {
    def.block_size != 0
        || def.block_size_index != 0
        || def.resample_factor.to_bits() != 1.0f32.to_bits()
        || def.resample_index != 0
}

fn encode_def(out: &mut Vec<u8>, def: &SynthDef, v3: bool) -> Result<(), Error> {
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
    if v3 {
        out.extend_from_slice(&def.block_size.to_be_bytes());
        out.extend_from_slice(&def.block_size_index.to_be_bytes());
        out.extend_from_slice(&def.resample_factor.to_be_bytes());
        out.extend_from_slice(&def.resample_index.to_be_bytes());
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
