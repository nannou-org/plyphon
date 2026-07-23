//! Two-buffer spectral (`PV_*`) operators - plyphon's ports of scsynth's `PV_Add`, `PV_Mul`,
//! `PV_Div`, `PV_Min`, `PV_Max`, `PV_CopyPhase` and `PV_Copy` (`PV_UGens.cpp`).
//!
//! Each reads a second FFT-chain buffer `B` and combines it into buffer `A` in place, via the shared
//! two-buffer seam [`buffer_pair_mut`](crate::unit::buffer_pair_mut) (as `PV_MagMul` does). That seam
//! lends `A` mutably and `B` read-only, so - like `PV_MagMul` - `B`'s bins are read in whatever form
//! it is already in ([`pv::bin_as_complex`]/[`pv::bin_as_polar`]) and `B` is left untouched, while
//! only `A` is converted. Compiled only with the `fft` feature.

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

/// The B-buffer index if a frame is ready on both A (input 0) and B (input 1). Passes A downstream.
fn frame_pair(ctx: &mut ProcessCtx<'_>) -> Option<(usize, usize)> {
    let fbuf_b = ctx.ins.control(1);
    let a = pv::pv_frame(ctx)?;
    (fbuf_b >= 0.0).then_some((a, fbuf_b as usize))
}

/// `PV_Add`/`PV_Mul`/`PV_Div(bufferA, bufferB)`: combine two spectra bin-by-bin in Cartesian form.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvComplex {
    kind: u32,
}

impl Unit for PvComplex {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let kind = self.kind;
        if let Some((a_idx, b_idx)) = frame_pair(ctx)
            && let Some((mut buf_a, buf_b)) =
                unit::buffer_pair_mut(ctx.buffers, &mut ctx.local_bufs, a_idx, b_idx)
            && buf_a.num_frames() == buf_b.num_frames()
            // A frame needs at least its `[dc, nyq]` header; a shorter buffer (`LocalBuf(1, 1)`)
            // must not panic the audio thread on the raw reads below.
            && buf_b.data().len() >= 2
        {
            let coord_b = buf_b.coord();
            let (b_dc, b_nyq) = (buf_b.data()[0], buf_b.data()[1]);
            let b_bins = pv::bins(buf_b.data());
            if let Some(a) = pv::to_complex(&mut buf_a) {
                match kind {
                    1 => {
                        *a.dc *= b_dc;
                        *a.nyq *= b_nyq;
                        for (p, &qraw) in a.bins.iter_mut().zip(b_bins) {
                            let q = pv::bin_as_complex(coord_b, qraw);
                            let (ar, ai) = (p.x, p.y);
                            p.x = ar * q.x - ai * q.y;
                            p.y = ar * q.y + ai * q.x;
                        }
                    }
                    2 => {
                        *a.dc /= b_dc;
                        *a.nyq /= b_nyq;
                        for (p, &qraw) in a.bins.iter_mut().zip(b_bins) {
                            let q = pv::bin_as_complex(coord_b, qraw);
                            let denom = q.x * q.x + q.y * q.y;
                            let (ar, ai) = (p.x, p.y);
                            p.x = (ar * q.x + ai * q.y) / denom;
                            p.y = (ai * q.x - ar * q.y) / denom;
                        }
                    }
                    _ => {
                        *a.dc += b_dc;
                        *a.nyq += b_nyq;
                        for (p, &qraw) in a.bins.iter_mut().zip(b_bins) {
                            let q = pv::bin_as_complex(coord_b, qraw);
                            p.x += q.x;
                            p.y += q.y;
                        }
                    }
                }
            }
        }
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
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let is_min = self.kind == 1;
        if let Some((a_idx, b_idx)) = frame_pair(ctx)
            && let Some((mut buf_a, buf_b)) =
                unit::buffer_pair_mut(ctx.buffers, &mut ctx.local_bufs, a_idx, b_idx)
            && buf_a.num_frames() == buf_b.num_frames()
            // A frame needs at least its `[dc, nyq]` header; a shorter buffer (`LocalBuf(1, 1)`)
            // must not panic the audio thread on the raw reads below.
            && buf_b.data().len() >= 2
        {
            let coord_b = buf_b.coord();
            let (b_dc, b_nyq) = (buf_b.data()[0], buf_b.data()[1]);
            let b_bins = pv::bins(buf_b.data());
            if let Some(a) = pv::to_polar(&mut buf_a) {
                // `dc`/`nyq` compare by absolute value; bins by (non-negative) magnitude.
                let pick_real = |pv: f32, qv: f32| {
                    let take = if is_min {
                        qv.abs() < pv.abs()
                    } else {
                        qv.abs() > pv.abs()
                    };
                    if take { qv } else { pv }
                };
                *a.dc = pick_real(*a.dc, b_dc);
                *a.nyq = pick_real(*a.nyq, b_nyq);
                for (p, &qraw) in a.bins.iter_mut().zip(b_bins) {
                    let q = pv::bin_as_polar(coord_b, qraw);
                    let take = if is_min { q.x < p.x } else { q.x > p.x };
                    if take {
                        *p = q;
                    }
                }
            }
        }
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
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if let Some((a_idx, b_idx)) = frame_pair(ctx)
            && let Some((mut buf_a, buf_b)) =
                unit::buffer_pair_mut(ctx.buffers, &mut ctx.local_bufs, a_idx, b_idx)
            && buf_a.num_frames() == buf_b.num_frames()
            // A frame needs at least its `[dc, nyq]` header; a shorter buffer (`LocalBuf(1, 1)`)
            // must not panic the audio thread on the raw reads below.
            && buf_b.data().len() >= 2
        {
            let coord_b = buf_b.coord();
            let (b_dc, b_nyq) = (buf_b.data()[0], buf_b.data()[1]);
            let b_bins = pv::bins(buf_b.data());
            if let Some(a) = pv::to_polar(&mut buf_a) {
                // scsynth flips A's real DC/Nyquist sign to agree with B's.
                if (*a.dc > 0.0) == (b_dc < 0.0) {
                    *a.dc = -*a.dc;
                }
                if (*a.nyq > 0.0) == (b_nyq < 0.0) {
                    *a.nyq = -*a.nyq;
                }
                for (p, &qraw) in a.bins.iter_mut().zip(b_bins) {
                    p.y = pv::bin_as_polar(coord_b, qraw).y;
                }
            }
        }
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
