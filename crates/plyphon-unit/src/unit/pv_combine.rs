//! Two-buffer spectral (`PV_*`) operators - plyphon's ports of scsynth's `PV_Add`, `PV_Mul`,
//! `PV_Div`, `PV_Min`, `PV_Max`, `PV_CopyPhase` and `PV_Copy` (`PV_UGens.cpp`).
//!
//! Each but `PV_Copy` reads a second FFT-chain buffer `B` and combines it into buffer `A` in place,
//! through the shared two-buffer preamble [`pv::pv_pair`] (as `PV_MagMul` does): both buffers are
//! converted in place to the form the op works in, and when both inputs name one buffer the op runs
//! on that buffer against itself, as in scsynth. Compiled only with the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, pv, unit_spec};

/// Which complex per-bin combination a [`PvComplex`] applies.
#[derive(Copy, Clone)]
pub enum ComplexKind {
    /// `PV_Add` - complex sum `A + B`.
    Add,
    /// `PV_Mul` - complex product `A * B`.
    Mul,
    /// `PV_Div` - complex quotient `A / B`.
    Div,
}

impl ComplexKind {
    fn to_tag(self) -> u32 {
        match self {
            ComplexKind::Add => 0,
            ComplexKind::Mul => 1,
            ComplexKind::Div => 2,
        }
    }
}

/// Which polar (magnitude-compare) combination a [`PvPolar`] applies.
#[derive(Copy, Clone)]
pub enum PolarKind {
    /// `PV_Max` - keep whichever bin (A or B) has the larger magnitude.
    Max,
    /// `PV_Min` - keep whichever bin (A or B) has the smaller magnitude.
    Min,
}

impl PolarKind {
    fn to_tag(self) -> u32 {
        match self {
            PolarKind::Max => 0,
            PolarKind::Min => 1,
        }
    }
}

/// `PV_Add`/`PV_Mul`/`PV_Div(bufferA, bufferB)`: combine two spectra bin-by-bin in Cartesian form.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvComplex {
    kind: u32,
}

impl Unit for PvComplex {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let kind = self.kind;
        pv::pv_pair(ctx, pv::to_complex, |p, q| match kind {
            1 => {
                *p.dc *= q.dc(*p.dc);
                *p.nyq *= q.nyq(*p.nyq);
                for (i, bin) in p.bins.iter_mut().enumerate() {
                    let qb = q.bin(i, *bin);
                    // scsynth's three-multiplication complex product, in its order.
                    let preal = bin.x;
                    let realmul = preal * qb.x;
                    let imagmul = bin.y * qb.y;
                    bin.x = realmul - imagmul;
                    // When `B` is `A`, `B`'s real part is the one just written.
                    let qreal = if q.is_same() { bin.x } else { qb.x };
                    bin.y = (preal + bin.y) * (qreal + qb.y) - realmul - imagmul;
                }
            }
            2 => {
                *p.dc /= q.dc(*p.dc);
                *p.nyq /= q.nyq(*p.nyq);
                for (i, bin) in p.bins.iter_mut().enumerate() {
                    let qb = q.bin(i, *bin);
                    let hypot = qb.x * qb.x + qb.y * qb.y;
                    let preal = bin.x;
                    bin.x = (preal * qb.x + bin.y * qb.y) / hypot;
                    // When `B` is `A`, `B`'s real part is the one just written.
                    let qreal = if q.is_same() { bin.x } else { qb.x };
                    bin.y = (bin.y * qreal - preal * qb.y) / hypot;
                }
            }
            _ => {
                *p.dc += q.dc(*p.dc);
                *p.nyq += q.nyq(*p.nyq);
                for (i, bin) in p.bins.iter_mut().enumerate() {
                    let qb = q.bin(i, *bin);
                    bin.x += qb.x;
                    bin.y += qb.y;
                }
            }
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`PvComplex`], parameterised by [`ComplexKind`].
pub struct PvComplexCtor(pub ComplexKind);

impl UnitDef for PvComplexCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvComplex {
            kind: self.0.to_tag(),
        }))
    }
}

/// `PV_Max`/`PV_Min(bufferA, bufferB)`: keep whichever of the two spectra has the larger/smaller
/// magnitude in each bin.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvPolar {
    kind: u32,
}

impl Unit for PvPolar {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let is_min = self.kind == 1;
        pv::pv_pair(ctx, pv::to_polar, |p, q| {
            // `dc`/`nyq` compare by absolute value; bins by magnitude.
            let pick_real = |pv: f32, qv: f32| {
                let take = if is_min {
                    qv.abs() < pv.abs()
                } else {
                    qv.abs() > pv.abs()
                };
                if take { qv } else { pv }
            };
            *p.dc = pick_real(*p.dc, q.dc(*p.dc));
            *p.nyq = pick_real(*p.nyq, q.nyq(*p.nyq));
            for (i, bin) in p.bins.iter_mut().enumerate() {
                let qb = q.bin(i, *bin);
                let take = if is_min { qb.x < bin.x } else { qb.x > bin.x };
                if take {
                    *bin = qb;
                }
            }
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`PvPolar`], parameterised by [`PolarKind`].
pub struct PvPolarCtor(pub PolarKind);

impl UnitDef for PvPolarCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvPolar {
            kind: self.0.to_tag(),
        }))
    }
}

/// `PV_CopyPhase(bufferA, bufferB)`: give `A`'s magnitudes `B`'s phases.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvCopyPhase {
    _pad: u32,
}

impl Unit for PvCopyPhase {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        pv::pv_pair(ctx, pv::to_polar, |p, q| {
            // scsynth flips A's real DC/Nyquist sign to agree with B's.
            if (*p.dc > 0.0) == (q.dc(*p.dc) < 0.0) {
                *p.dc = -*p.dc;
            }
            if (*p.nyq > 0.0) == (q.nyq(*p.nyq) < 0.0) {
                *p.nyq = -*p.nyq;
            }
            for (i, bin) in p.bins.iter_mut().enumerate() {
                bin.y = q.bin(i, *bin).y;
            }
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`PvCopyPhase`].
pub struct PvCopyPhaseCtor;

impl UnitDef for PvCopyPhaseCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvCopyPhase { _pad: 0 }))
    }
}

/// `PV_Copy(bufferA, bufferB)`: copy `A`'s whole spectrum into `B` and pass `B` downstream (so `A`
/// stays usable by a parallel chain while `B` carries the copy).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvCopy {
    _pad: u32,
}

impl Unit for PvCopy {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 1 (the destination chain) through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(1);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let fbuf_a = ctx.ins.control(0);
        let fbuf_b = ctx.ins.control(1);
        // PV_Copy uniquely passes buffer *B* downstream (not A).
        *ctx.outs.control(0) = if fbuf_a >= 0.0 && fbuf_b >= 0.0 {
            fbuf_b
        } else {
            -1.0
        };
        // Borrow B mutably (the destination) and A read-only (the source) - the reverse of the usual
        // pairing - then overwrite B's samples and coordinate form with A's.
        if fbuf_a >= 0.0
            && fbuf_b >= 0.0
            && let Some((mut buf_b, buf_a)) = unit::buffer_pair_mut(
                ctx.buffers,
                &mut ctx.local_bufs,
                fbuf_b as usize,
                fbuf_a as usize,
            )
            && buf_a.data().len() == buf_b.data().len()
        {
            let coord = buf_a.coord();
            buf_b.data_mut().copy_from_slice(buf_a.data());
            buf_b.set_coord(coord);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvCopy`].
pub struct PvCopyCtor;

impl UnitDef for PvCopyCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvCopy { _pad: 0 }))
    }
}
