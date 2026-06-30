//! Parsing SCgf bytes into a [`SynthDefFile`].

use alloc::string::String;
use alloc::vec::Vec;

use crate::{Error, Input, ParamName, Rate, SynthDef, SynthDefFile, Ugen, Variant};

/// Parse an SCgf buffer.
pub fn parse(data: &[u8]) -> Result<SynthDefFile, Error> {
    let mut reader = Reader::new(data);
    if reader.take(4)? != b"SCgf" {
        return Err(Error::BadMagic);
    }
    let version = reader.i32()?;
    if !(1..=3).contains(&version) {
        return Err(Error::UnsupportedVersion(version));
    }
    let num_defs = reader.i16()? as usize;
    let mut defs = Vec::with_capacity(num_defs);
    for _ in 0..num_defs {
        if version >= 3 {
            // Version 3 prefixes each def with an int32 byte size (including the 4-byte size field
            // itself); the reader advances by it, tolerating any trailing bytes the def may carry.
            let start = reader.pos;
            let size = reader.i32()?;
            let def = parse_def(&mut reader, version)?;
            let end = start
                .checked_add(usize::try_from(size).map_err(|_| Error::Truncated)?)
                .ok_or(Error::Truncated)?;
            reader.seek(end)?;
            defs.push(def);
        } else {
            defs.push(parse_def(&mut reader, version)?);
        }
    }
    Ok(SynthDefFile { version, defs })
}

fn parse_def(reader: &mut Reader<'_>, version: i32) -> Result<SynthDef, Error> {
    let name = reader.pstring()?;

    let num_constants = reader.count(version)?;
    let mut constants = Vec::with_capacity(num_constants);
    for _ in 0..num_constants {
        constants.push(reader.f32()?);
    }

    let num_params = reader.count(version)?;
    let mut param_values = Vec::with_capacity(num_params);
    for _ in 0..num_params {
        param_values.push(reader.f32()?);
    }

    let num_param_names = reader.count(version)?;
    let mut param_names = Vec::with_capacity(num_param_names);
    for _ in 0..num_param_names {
        let name = reader.pstring()?;
        let index = reader.count(version)? as u32;
        param_names.push(ParamName { name, index });
    }

    let num_ugens = reader.count(version)?;
    let mut ugens = Vec::with_capacity(num_ugens);
    for _ in 0..num_ugens {
        ugens.push(parse_ugen(reader, version)?);
    }

    let num_variants = reader.i16()? as usize;
    let mut variants = Vec::with_capacity(num_variants);
    for _ in 0..num_variants {
        let name = reader.pstring()?;
        let mut values = Vec::with_capacity(num_params);
        for _ in 0..num_params {
            values.push(reader.f32()?);
        }
        variants.push(Variant { name, values });
    }

    // Version 3 appends the reblock/resample fields; older versions default to "no reblock, no
    // oversample". The index fields are stored unsigned (scsynth reads them with a signed int32 read).
    let (block_size, block_size_index, resample_factor, resample_index) = if version >= 3 {
        let block_size = reader.i32()?;
        let block_size_index = reader.i32()? as u32;
        let resample_factor = reader.f32()?;
        let resample_index = reader.i32()? as u32;
        (
            block_size,
            block_size_index,
            resample_factor,
            resample_index,
        )
    } else {
        (0, 0, 1.0, 0)
    };

    Ok(SynthDef {
        name,
        constants,
        param_values,
        param_names,
        ugens,
        variants,
        block_size,
        block_size_index,
        resample_factor,
        resample_index,
    })
}

fn parse_ugen(reader: &mut Reader<'_>, version: i32) -> Result<Ugen, Error> {
    let name = reader.pstring()?;
    let rate = reader.rate()?;
    let num_inputs = reader.count(version)?;
    let num_outputs = reader.count(version)?;
    let special_index = reader.i16()?;

    let mut inputs = Vec::with_capacity(num_inputs);
    for _ in 0..num_inputs {
        let from_ugen = reader.index(version)?;
        let from_output = reader.index(version)?;
        inputs.push(if from_ugen < 0 {
            Input::Constant {
                index: from_output as u32,
            }
        } else {
            Input::Ugen {
                ugen: from_ugen as u32,
                output: from_output as u32,
            }
        });
    }

    let mut outputs = Vec::with_capacity(num_outputs);
    for _ in 0..num_outputs {
        outputs.push(reader.rate()?);
    }

    Ok(Ugen {
        name,
        rate,
        special_index,
        inputs,
        outputs,
    })
}

/// A big-endian cursor over an SCgf buffer.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).ok_or(Error::Truncated)?;
        let slice = self.data.get(self.pos..end).ok_or(Error::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    /// Advance the cursor to `pos` (used to skip to a v3 def's declared end). Errors if `pos` is
    /// before the current cursor - which means the def overran its declared size - or past the buffer.
    fn seek(&mut self, pos: usize) -> Result<(), Error> {
        if pos < self.pos || pos > self.data.len() {
            return Err(Error::Truncated);
        }
        self.pos = pos;
        Ok(())
    }

    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }

    fn i16(&mut self) -> Result<i16, Error> {
        let b = self.take(2)?;
        Ok(i16::from_be_bytes([b[0], b[1]]))
    }

    fn i32(&mut self) -> Result<i32, Error> {
        let b = self.take(4)?;
        Ok(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn f32(&mut self) -> Result<f32, Error> {
        let b = self.take(4)?;
        Ok(f32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn pstring(&mut self) -> Result<String, Error> {
        let len = self.u8()? as usize;
        let bytes = self.take(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn rate(&mut self) -> Result<Rate, Error> {
        let code = self.u8()? as i8;
        Rate::from_code(code).ok_or(Error::BadRate(code))
    }

    /// A non-negative count: `int16` in v1, `int32` in v2.
    fn count(&mut self, version: i32) -> Result<usize, Error> {
        usize::try_from(self.index(version)?).map_err(|_| Error::Truncated)
    }

    /// A possibly-negative index: `int16` in v1, `int32` in v2.
    fn index(&mut self, version: i32) -> Result<i32, Error> {
        if version >= 2 {
            self.i32()
        } else {
            Ok(self.i16()? as i32)
        }
    }
}
