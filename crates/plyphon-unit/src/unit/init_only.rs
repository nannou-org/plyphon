//! The declared init-only input table and the auxiliary-allocation bound.
//!
//! An *init-only* input is consumed entirely at build time to size per-instance auxiliary
//! memory - scsynth reads it once at ctor (`ZIN0`) and never again. The table below declares,
//! per registry name, exactly which input indices are init-only: the initialization evaluator
//! (`SynthDef::specialize_init` in the `plyphon` crate) proves and rewrites *only* declared
//! inputs, so a parameter consumed both as an aux size and as a live signal keeps its live use
//! untouched by construction.
//!
//! The table is exactly the set of [`BuildError::AuxRequiresConstant`] raise sites, with two
//! deliberate exclusions: `FFT`/`IFFT` window size (stays syntactic-constant-only, so no
//! parameter-derived FFT-plan respecialization exists) and `SendReply`/`Poll` label characters
//! (OSC paths are encoded constants, not sizes).

use crate::error::BuildError;

/// The per-allocation element bound: an allocation site whose total element count exceeds this
/// fails with [`BuildError::AuxSizeOutOfRange`] instead of allocating. The committed corpus
/// maximum is a 24 s `DelayC` line - 8.4M elements after power-of-two rounding at a 192 kHz
/// graph rate - so the bound admits every shipped definition at every supported device rate
/// while still rejecting out-of-range preset or parameter-derived sizes deterministically at
/// build rather than as truncation, wrap, or silent pool exhaustion.
pub const MAX_AUX_ELEMS: u64 = 1 << 24;

/// The declared init-only input indices for `unit_name`, empty for units with none.
///
/// Kept in exact correspondence with the `AuxRequiresConstant` raise sites (minus the documented
/// `FFT`/`IFFT` exclusion); a test pins the correspondence so a new aux site upstream fails until
/// it is declared here or explicitly excluded.
pub fn init_only_inputs(unit_name: &str) -> &'static [usize] {
    match unit_name {
        // The delay family's `maxdelaytime` (`build_line`'s MAXDELAY).
        "DelayN" | "DelayL" | "DelayC" | "CombN" | "CombL" | "CombC" | "AllpassN" | "AllpassL"
        | "AllpassC" => &[1],
        // `Pluck`'s `maxdelaytime`.
        "Pluck" => &[2],
        // `PitchShift`'s `windowSize`.
        "PitchShift" => &[1],
        // `LocalBuf`'s `channels` and `frames`.
        "LocalBuf" => &[0, 1],
        // `Gendy1`'s `initCPs`.
        "Gendy1" => &[8],
        // The `Limiter`/`Normalizer` look-ahead duration.
        "Limiter" | "Normalizer" => &[2],
        // `Median`'s window length (clamped into a fixed array; no aux allocation, so no bound
        // check applies at its site).
        "Median" => &[0],
        // `GVerb`'s `roomsize`, `spread`, and `maxroomsize`.
        "GVerb" => &[1, 5, 9],
        _ => &[],
    }
}

/// Bound-check the total element count an allocation site is about to allocate.
///
/// `elements` must be accumulated with saturating arithmetic (saturating float→int casts and
/// saturating add/multiply) *before* any narrowing cast, so an out-of-range size arrives here as
/// a large value rather than a truncated or wrapped small one. A count strictly greater than
/// [`MAX_AUX_ELEMS`] fails.
pub fn checked_aux_elems(
    unit: &'static str,
    input: usize,
    elements: u64,
) -> Result<(), BuildError> {
    if elements > MAX_AUX_ELEMS {
        return Err(BuildError::AuxSizeOutOfRange {
            unit: unit.into(),
            input,
            elements,
            limit: MAX_AUX_ELEMS,
        });
    }
    Ok(())
}
