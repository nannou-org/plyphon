# Writing SynthDefs: what fails where

`plyphon_synthdef` builds a [`SynthDef`] from fluent UGen builders with sclang-style
multichannel expansion:

```rust
use plyphon_synthdef::{SynthDefBuilder, UGenBuilder, Out, Pan2, Saw};

let (def, freq_index) = SynthDefBuilder::build_with("detuned-saw", |g| {
    let freq = g.add_control_param("freq", 110.0);
    let saws = Saw::ar(g).freq([freq + 0.0, freq + 1.5]); // still a builder
    let sig = saws * 0.2;                                 // emits: 2 Saws, 2 muls
    Out::ar(g)
        .channels(Pan2::ar(g).input(sig).pos([-0.5, 0.5]))
        .emit();
    freq.param_index().unwrap() // plain data may leave the closure; Channels may not
});
```

`build_with` returns the closure's result alongside the def. `Channel`s borrow the builder and are
confined to the closure, so this is how the parameter indices needed for runtime `set_control`
get out: typically as a struct with one field per parameter (see `example-filtered-saw`).

The core design decision is **deferred emission**: a rate constructor (`SinOsc::ar(g)`) returns a
*builder* holding the input trees, pre-filled with sclang defaults, and nothing is written into
the def until the builder is finalized: by being consumed as an input to another UGen, combined
with an operator, or explicitly via `.signal()`. Expansion happens once, at finalize, which is what
lets a later setter freely change an input's channel count.

This document lists the consequences for someone authoring a def, organized by *when* you find
out something is wrong: at compile time, at def-build time (panic), or never (silent). The last
bucket is the one to internalize.

## The three rules

1. **A builder becomes real when something consumes it**: as an input, in an operator, via a
   math method, or via `.signal()`.
2. **Sinks are consumed by nothing, so always end them with `.emit()`** - and don't ignore
   `unused_must_use` warnings; they are the safety net for rule 1.
3. **To fan one signal out to several places, `.signal()` it once and pass `&handle`**: the borrow
   checker enforces this; a "use of moved value" error on a builder is the signal to apply this
   rule.

Everything below is a case of one of these.

## A. Caught by the compiler (clear error, easy fix)

### 1. Reusing a builder for two consumers

A builder is consumed the first time it's used as an input:

```rust
let osc = SinOsc::ar(g).freq(440.0);
let a = LPF::ar(g).input(osc);
let b = RLPF::ar(g).input(osc);   // error: use of moved value `osc`
```

This is deliberate: the alternative (allowing `From<&Builder>`) would silently emit *two* SinOsc
units. Finalize once and reuse the handle:

```rust
let osc = SinOsc::ar(g).freq(440.0).signal();
let a = LPF::ar(g).input(&osc);
let b = RLPF::ar(g).input(&osc);  // same oscillator, two consumers
```

### 2. The same trap on `Signal`, one level up

`Signal` isn't `Copy` (it can own a channel array) and the by-value operator impls exist, so
`sig * 2.0` *moves* `sig`. If you need `sig` again, write `&sig * 2.0`. `Channel` (a param or a single
unit output) is `Copy`: `freq + 1.0` never consumes `freq`.

### 3. Scalar on the left of a *builder*

`0.5 * SinOsc::ar(g).freq(440.0)` doesn't compile: Rust's orphan rule forbids a blanket
`impl Mul<AnyBuilder> for f32`. Only the concrete types `Channel`, `Signal`, `&Signal` have
scalar-left impls. Write `builder * 0.5`, or finalize first (`0.5 * sig` on a finalized handle is
fine).

### 4. Mixed-type channel arrays

`[440.0, some_signal]` is not a valid Rust array (two element types). Use
`mce![440.0, some_signal]`.

### 5. Generic helper functions need the right bound

`UGenBuilder` is implemented *only by builders*, so a helper like

```rust
fn stereo_verb<'g>(x: impl UGenBuilder<'g>) -> Signal<'g> { /* … */ }   // rejects Signal/Channel!
```

won't accept an already-finalized handle. The bound that means "anything usable as an input" -
builders, `Channel`, `Signal`, `&Signal`, scalars, arrays - is `impl Into<UGenInput<'g>>`. Use
`UGenBuilder` only when you specifically need `.signal()` or the named math methods generically.

Why it's this way: if `Signal` implemented `UGenBuilder`, Rust's method probing would pick the trait's
by-value `midi_cps(self)` over the inherent `midi_cps(&self)` (by-value candidates are checked
before autoref), and `sig.midi_cps()` would consume `sig`. Keeping `Channel`/`Signal` off the trait
keeps math on finalized handles non-consuming.

## B. Panics while building the def (loud, at the exact call site)

- **Cross-builder wiring**: a `Channel` from builder A fed into a constructor or operator finalizing
  into builder B panics with `"input signal for <name> belongs to a different SynthDefBuilder"`. Only
  possible if two `SynthDefBuilder`s are alive at once.
- **Empty channel array**: `.freq(Vec::new())` panics with `"empty multichannel input to SinOsc"`
  at finalize.
- **Duplicate parameter name**: `g.add_control_param("freq", …)` twice panics at the second declaration.

## C. Compiles and builds, but the def is silently wrong: the real footguns

### 1. The forgotten `Out` finalizer

The one to watch for:

```rust
SynthDefBuilder::build_with("synth", |g| {
    let sig = RLPF::ar(g).input(Saw::ar(g).freq(110.0)).freq(800.0);
    Out::ar(g).channels(sig);        // ← missing .emit()
});
```

Because a sink is never *used as an input*, nothing ever finalizes it. The def contains Saw and
RLPF (they were emitted when `sig` was consumed by `.channels()`) but no `Out`: the synth runs
and produces **silence**. The only guard is the `#[must_use]` **warning**
(`"an Out builder emits nothing until it is finalized with .emit()"`). Treat that warning as an
error; `#![deny(unused_must_use)]` in a def-authoring crate is a reasonable policy.

### 2. Any dropped builder is a no-op

More generally, `SinOsc::ar(g).freq(440.0);` as a bare statement emits nothing (plus the same
warning). Unlike sclang, where `SinOsc.ar(440)` inside a SynthDef function *does* instantiate a
UGen even if unused. For pure signal sources the difference is harmless: you didn't want the
unit anyway. It will matter when side-effectful, output-less units get wrapped
(`SendReply`, `DetectSilence`, `LocalOut`, envelopes with `doneAction`): every one of those is
"fire and forget" and must get the `Out` treatment - an explicit inherent `.emit()` terminal -
and forgetting it means the trigger/free/reply simply never happens.

### 3. Unit order is finalization order, not source order

Always topologically valid, so the def is never *broken* - but if you keep index constants (like
`example-filtered-saw`'s `U_SAW = 0`) or diff against a sclang def dump, indices follow the order
builders are *consumed*, which for nested fluent expressions is inside-out: in
`RLPF::ar(g).input(Saw::ar(g).freq(f)).mul_add(amp, 0.0)`, Saw lands first (consumed by `.input()`),
RLPF second (consumed by `.mul_add()`), MulAdd third.

## D. Deferred to engine install time (not this crate's checks)

The builder does **not** validate unit names against the engine registry, rates (an audio-rate
signal into a `kr` unit), or input arity beyond what the setters enforce. Those surface when the
def is compiled at `add_synthdef`/`synth_new`. (Piping a sink's output onward used to land here too;
since `emit()` returns `()`, `Out::…emit() * 2.0` now fails to compile instead.)

[`SynthDef`]: https://docs.rs/plyphon/latest/plyphon/struct.SynthDef.html

## Declaring custom UGens

The `ugen!` macro is exported, so a downstream crate can wrap a unit the builder doesn't cover
yet (or shadow a built-in wrapper) with the same one-invocation declaration used internally -
`crates/examples/filtered-saw` does exactly this for `RLPF`:

```rust
use plyphon_synthdef::{Rate, ugen};

ugen!(
    /// Resonant two-pole low-pass.
    RLPF => RLPFBuilder [ar: Rate::Audio, kr: Rate::Control](
        input = 0.0,
        freq = 440.0,
        rq = 1.0,
    ) -> 1
);
```

The generated builder is indistinguishable from a built-in one: defaults, setters, operators,
the `UGenBuilder` math methods, finalize-on-use, `#[must_use]`. The struct name doubles as the
engine registry name (via `stringify!`), so it must match the unit's registered name exactly.

Units with special input shapes are hand-written against the public core API, mirroring `In`
(structural `num_outputs` as a constructor argument) or `Out` (a sink with an inherent `.emit()`
terminal and `UGenInput::flatten` for the flat-spread channel list): a builder struct holding
`&SynthDefBuilder` + input trees, a `UGenBuilder` impl (sources) or an inherent `.emit()` (sinks)
calling `SynthDefBuilder::add`, a `From<YourBuilder> for UGenInput` impl for finalize-on-use
(sources only), and `impl_builder_ops!(YourBuilder)` for the operators.
