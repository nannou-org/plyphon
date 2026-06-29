//! Pure OSC argument extraction - the controller-free, dispatcher-free half of decoding a command.
//!
//! These helpers turn a single [`OscType`] (or a message's trailing blob) into the typed value a
//! command handler wants, with the same [`OscError`] the dispatcher reports. They
//! hold no state and touch neither the engine nor the dispatcher, so a host can reuse them to
//! pre-validate or hand-decode messages without constructing an [`OscDispatcher`](crate::OscDispatcher).

use rosc::OscType;

use crate::OscError;

/// An `int` argument.
pub fn int_arg(arg: &OscType) -> Result<i32, OscError> {
    match arg {
        OscType::Int(i) => Ok(*i),
        _ => Err(OscError::BadArguments("expected an int")),
    }
}

/// A non-negative bus index (`/c_set`, `/c_setn`).
pub fn bus_index(arg: &OscType) -> Result<u32, OscError> {
    u32::try_from(int_arg(arg)?).map_err(|_| OscError::BadArguments("negative bus index"))
}

/// A bus index for `/n_map`/`/n_mapn`, where a negative index means "unmap".
pub fn map_bus(arg: &OscType) -> Result<Option<u32>, OscError> {
    let bus = int_arg(arg)?;
    Ok(u32::try_from(bus).ok())
}

/// A non-negative count argument (`/c_setn`, `/n_mapn`).
pub fn count_arg(arg: Option<&OscType>) -> Result<usize, OscError> {
    let arg = arg.ok_or(OscError::BadArguments("expected a count"))?;
    usize::try_from(int_arg(arg)?).map_err(|_| OscError::BadArguments("negative count"))
}

/// A non-negative `usize` index argument (`/b_set`, `/b_setn`, `/b_fill`).
pub fn index_arg(arg: &OscType) -> Result<usize, OscError> {
    usize::try_from(int_arg(arg)?).map_err(|_| OscError::BadArguments("negative index"))
}

/// A number argument (`float`/`int`/`double` all coerce to `f32`).
pub fn float_arg(arg: &OscType) -> Result<f32, OscError> {
    match arg {
        OscType::Float(f) => Ok(*f),
        OscType::Int(i) => Ok(*i as f32),
        OscType::Double(d) => Ok(*d as f32),
        _ => Err(OscError::BadArguments("expected a number")),
    }
}

/// A string argument.
pub fn str_arg(arg: &OscType) -> Result<&str, OscError> {
    match arg {
        OscType::String(s) => Ok(s.as_str()),
        _ => Err(OscError::BadArguments("expected a string")),
    }
}

/// The trailing OSC completion blob of an async command, if the last argument is one.
pub fn last_blob(args: &[OscType]) -> Option<&[u8]> {
    match args.last() {
        Some(OscType::Blob(bytes)) => Some(bytes),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn numbers_coerce_to_f32() {
        assert_eq!(float_arg(&OscType::Int(3)).unwrap(), 3.0);
        assert_eq!(float_arg(&OscType::Float(1.5)).unwrap(), 1.5);
        assert_eq!(float_arg(&OscType::Double(2.25)).unwrap(), 2.25);
        assert!(float_arg(&OscType::String("x".to_string())).is_err());
    }

    #[test]
    fn bus_and_index_reject_negatives() {
        assert_eq!(bus_index(&OscType::Int(7)).unwrap(), 7);
        assert!(bus_index(&OscType::Int(-1)).is_err());
        assert!(index_arg(&OscType::Int(-1)).is_err());
        assert!(count_arg(None).is_err());
        assert_eq!(count_arg(Some(&OscType::Int(4))).unwrap(), 4);
    }

    #[test]
    fn map_bus_negative_means_unmap() {
        assert_eq!(map_bus(&OscType::Int(2)).unwrap(), Some(2));
        assert_eq!(map_bus(&OscType::Int(-1)).unwrap(), None);
    }

    #[test]
    fn last_blob_only_for_trailing_blob() {
        assert_eq!(
            last_blob(&[OscType::Int(0), OscType::Blob(vec![1, 2, 3])]),
            Some(&[1u8, 2, 3][..])
        );
        assert_eq!(last_blob(&[OscType::Int(0)]), None);
        assert_eq!(last_blob(&[]), None);
    }
}
