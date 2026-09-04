//! The [`SynthDefBuilder`]: an arena graph of units. The builder owns the nodes, user code holds
//! cheap handles, and multichannel expansion happens eagerly, at node-construction time (like
//! sclang).
//!
//! Because an input must already exist as a handle before it can be wired, arena append order is
//! inherently a valid topological calc order: [`SynthDefBuilder::build`] is a plain serialization
//! pass, and cycles are unrepresentable.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt;

use plyphon::{InputRef, Param, Rate, SynthDef, UnitSpec};

/// The synthdef under construction: an arena of units plus the declared parameters.
///
/// UGen constructors take `&SynthDefBuilder` and return [`Signal`] handles; the handles carry the
/// builder reference, so operators and methods can allocate follow-up nodes without a visible
/// context.
///
/// Unlike the UGen builders (`SinOscBuilder`, `OutBuilder`, ...), which are single-use values
/// consumed when finalized, this is a long-lived *context*: it is only ever passed by `&`, and
/// every constructor and operator appends to it through interior mutability.
#[derive(Default)]
pub struct SynthDefBuilder {
    nodes: RefCell<Vec<NodeData>>,
    params: RefCell<Vec<Param>>,
}

/// One arena unit. Mirrors [`UnitSpec`], with resolved inputs.
#[derive(Debug)]
struct NodeData {
    name: &'static str,
    rate: Rate,
    inputs: Vec<Input>,
    num_outputs: usize,
    special_index: i16,
}

/// A resolved (mono) unit input. Mirrors [`InputRef`], but indexes a different space: `Node`
/// identifies a node in the builder's arena, while `InputRef::Unit` indexes the serialized def's
/// unit list. The two coincide today because [`SynthDefBuilder::build`] serializes every arena node in
/// order, but an optimization pass (op fusion, dead-unit elimination, re-sorting) would break
/// that equivalence - so the arena must not store final unit indices.
#[derive(Copy, Clone, Debug)]
enum Input {
    Constant(f32),
    Param(u32),
    Node { node: u32, output: u32 },
}

/// Where a [`Channel`] comes from.
#[derive(Copy, Clone, Debug)]
enum ChannelSource {
    /// The value of parameter `index` in the builder's parameter list.
    Param(u32),
    /// One output of a unit in the builder's arena.
    Node { node: u32, output: u32 },
}

/// A single mono channel: one output of a unit, or a parameter. `Copy`, so it can be reused freely.
///
/// A parameter counts as a channel for the same reason it does in sclang, where a control is one
/// output of the `Control` UGen.
#[derive(Copy, Clone)]
pub struct Channel<'g> {
    builder: &'g SynthDefBuilder,
    source: ChannelSource,
}

impl fmt::Debug for Channel<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Channel({:?})", self.source)
    }
}

impl<'g> Channel<'g> {
    /// The calc rate of this channel (a parameter's declared rate, or the source unit's rate).
    pub fn rate(&self) -> Rate {
        match self.source {
            ChannelSource::Param(i) => self.builder.params.borrow()[i as usize].rate,
            ChannelSource::Node { node, .. } => self.builder.nodes.borrow()[node as usize].rate,
        }
    }

    /// If this channel is a parameter, its index in the def's parameter list - the index
    /// [`Controller::set_control`](plyphon::Controller::set_control) addresses at runtime.
    /// `None` for a unit output.
    pub fn param_index(&self) -> Option<usize> {
        match self.source {
            ChannelSource::Param(i) => Some(i as usize),
            ChannelSource::Node { .. } => None,
        }
    }
}

/// A possibly multichannel value: what UGen constructors and operators return.
///
/// `Mono` is a single [`Channel`]; `Multi` is a channel array, possibly nested (nested arrays come from
/// expanding a multi-output UGen, e.g. `Pan2` over an array input).
#[derive(Clone, Debug)]
pub enum Signal<'g> {
    Mono(Channel<'g>),
    Multi(Vec<Signal<'g>>),
}

impl<'g> Signal<'g> {
    /// The number of (top-level) channels.
    pub fn num_channels(&self) -> usize {
        match self {
            Signal::Mono(_) => 1,
            Signal::Multi(v) => v.len(),
        }
    }

    /// Duplicate this value into an `n`-channel array (sclang's `sig ! n`), e.g. to feed the same
    /// signal to several `Out` channels.
    pub fn repeat(&self, n: usize) -> Signal<'g> {
        Signal::Multi((0..n).map(|_| self.clone()).collect())
    }
}

impl<'g> From<Channel<'g>> for Signal<'g> {
    fn from(s: Channel<'g>) -> Self {
        Signal::Mono(s)
    }
}

/// What UGen constructors accept: a tree whose leaves are constants or channels. A `Multi` input
/// triggers multichannel expansion.
#[derive(Clone, Debug)]
pub enum UGenInput<'g> {
    Constant(f32),
    Channel(Channel<'g>),
    Multi(Vec<UGenInput<'g>>),
}

impl<'g> UGenInput<'g> {
    /// The first builder found in the tree, if any leaf is a channel.
    pub(crate) fn builder(&self) -> Option<&'g SynthDefBuilder> {
        match self {
            UGenInput::Constant(_) => None,
            UGenInput::Channel(s) => Some(s.builder),
            UGenInput::Multi(v) => v.iter().find_map(|i| i.builder()),
        }
    }

    /// Append the tree's leaves to `out` in order, dissolving all `Multi` nesting. This is the
    /// flat-spread used by array-input units like `Out`, whose input list is a channel array;
    /// public so custom hand-written sinks can do the same.
    pub fn flatten(self, out: &mut Vec<UGenInput<'g>>) {
        match self {
            UGenInput::Multi(v) => v.into_iter().for_each(|i| i.flatten(out)),
            leaf => out.push(leaf),
        }
    }
}

impl<'g> From<f32> for UGenInput<'g> {
    fn from(v: f32) -> Self {
        UGenInput::Constant(v)
    }
}

/// Unsuffixed integer literals are `i32`.
impl<'g> From<i32> for UGenInput<'g> {
    fn from(v: i32) -> Self {
        UGenInput::Constant(v as f32)
    }
}

impl<'g> From<Channel<'g>> for UGenInput<'g> {
    fn from(s: Channel<'g>) -> Self {
        UGenInput::Channel(s)
    }
}

impl<'g> From<Signal<'g>> for UGenInput<'g> {
    fn from(u: Signal<'g>) -> Self {
        match u {
            Signal::Mono(s) => UGenInput::Channel(s),
            Signal::Multi(v) => UGenInput::Multi(v.into_iter().map(Into::into).collect()),
        }
    }
}

impl<'a, 'g> From<&'a Signal<'g>> for UGenInput<'g> {
    fn from(u: &'a Signal<'g>) -> Self {
        u.clone().into()
    }
}

impl<'g, T: Into<UGenInput<'g>>> From<Vec<T>> for UGenInput<'g> {
    fn from(v: Vec<T>) -> Self {
        UGenInput::Multi(v.into_iter().map(Into::into).collect())
    }
}

impl<'g, T: Into<UGenInput<'g>>, const N: usize> From<[T; N]> for UGenInput<'g> {
    fn from(v: [T; N]) -> Self {
        UGenInput::Multi(v.into_iter().map(Into::into).collect())
    }
}

impl<'a, 'g, T: Clone + Into<UGenInput<'g>>> From<&'a [T]> for UGenInput<'g> {
    fn from(v: &'a [T]) -> Self {
        UGenInput::Multi(v.iter().cloned().map(Into::into).collect())
    }
}

/// How a node's calc rate is determined when it is emitted.
#[derive(Copy, Clone)]
pub enum RateMode {
    /// The rate the caller picked (`ar()`/`kr()` constructors).
    Fixed(Rate),
    /// The max rate over the node's actual (post-expansion) inputs.
    /// Used by operators, so `kr_signal * ar_signal` comes out audio-rate.
    MaxOfInputs,
}

/// Rate precedence for [`RateMode::MaxOfInputs`] (`Rate` itself has no ordering).
fn rate_rank(rate: Rate) -> u8 {
    match rate {
        Rate::Scalar => 0,
        Rate::Control => 1,
        Rate::Audio => 2,
        Rate::Demand => 3,
    }
}

impl SynthDefBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build with a closure-scoped builder, so no handle can outlive the build. The closure's
    /// return value is passed back alongside the def - the way to get plain data (typically the
    /// [`Channel::param_index`] of each declared parameter, for runtime `set_control`) out of the
    /// scope the `Channel`s are confined to:
    ///
    /// ```
    /// use plyphon_synthdef::{SynthDefBuilder, UGenBuilder, Out, Saw};
    ///
    /// struct Params { freq: usize }
    ///
    /// let (def, params) = SynthDefBuilder::build_with("saw", |g| {
    ///     let freq = g.add_control_param("freq", 110.0);
    ///     Out::ar(g).channels(Saw::ar(g).freq(freq)).emit();
    ///     Params { freq: freq.param_index().unwrap() }
    /// });
    /// assert_eq!(params.freq, 0);
    /// assert_eq!(def.params[params.freq].name, "freq");
    /// ```
    pub fn build_with<R>(
        name: impl Into<String>,
        f: impl FnOnce(&SynthDefBuilder) -> R,
    ) -> (SynthDef, R) {
        let g = SynthDefBuilder::new();
        let result = f(&g);
        (g.build(name), result)
    }

    /// Declare a parameter. Panics if a parameter with the same name already exists.
    pub fn add_param(&self, param: Param) -> Channel<'_> {
        let mut params = self.params.borrow_mut();
        assert!(
            params.iter().all(|p| p.name != param.name),
            "duplicate parameter name {:?}",
            param.name
        );
        let index = params.len() as u32;
        params.push(param);
        Channel {
            builder: self,
            source: ChannelSource::Param(index),
        }
    }

    /// Declare a control-rate parameter ([`Param::control`]). Panics if a parameter with the
    /// same name already exists.
    pub fn add_control_param(&self, name: impl Into<String>, default: f32) -> Channel<'_> {
        self.add_param(Param::control(name, default))
    }

    /// Declare an audio-rate parameter ([`Param::audio`]). Panics if a parameter with the same
    /// name already exists.
    pub fn add_audio_param(&self, name: impl Into<String>, default: f32) -> Channel<'_> {
        self.add_param(Param::audio(name, default))
    }

    /// Declare a trigger parameter ([`Param::trig`]). Panics if a parameter with the same name
    /// already exists.
    pub fn add_trig_param(&self, name: impl Into<String>, default: f32) -> Channel<'_> {
        self.add_param(Param::trig(name, default))
    }

    /// Declare a lagged parameter ([`Param::lag`]). Panics if a parameter with the same name
    /// already exists.
    pub fn add_lag_param(&self, name: impl Into<String>, default: f32, lag: f32) -> Channel<'_> {
        self.add_param(Param::lag(name, default, lag))
    }

    /// Look up a parameter by name.
    pub fn param(&self, name: &str) -> Option<Channel<'_>> {
        self.params
            .borrow()
            .iter()
            .position(|p| p.name == name)
            .map(|index| Channel {
                builder: self,
                source: ChannelSource::Param(index as u32),
            })
    }

    /// Emit a unit with sclang multichannel-expansion semantics; every wrapper and operator goes
    /// through here. A `Multi` input expands the unit into `max(channel counts)` parallel units,
    /// with shorter inputs wrapping (channel `i` takes element `i % len`); nested `Multi`s expand
    /// recursively into nested results. A unit with `num_outputs > 1` returns a `Multi` of its
    /// output proxies.
    ///
    /// This is the equivalent of sclang's `UGen.multiNew` (the flop-and-recurse in
    /// `multiNewList`), with the builder passed explicitly instead of sclang's implicit
    /// `UGen.buildSynthDef` global.
    pub fn add<'g>(
        &'g self,
        name: &'static str,
        rate: RateMode,
        inputs: &[UGenInput<'g>],
        num_outputs: usize,
        special_index: i16,
    ) -> Signal<'g> {
        self.expand(name, rate, inputs, num_outputs, special_index)
    }

    fn expand<'g>(
        &'g self,
        name: &'static str,
        rate: RateMode,
        inputs: &[UGenInput<'g>],
        num_outputs: usize,
        special_index: i16,
    ) -> Signal<'g> {
        // Expansion factor: max len over top-level Multi inputs (mono inputs don't count).
        let n = inputs
            .iter()
            .filter_map(|i| match i {
                UGenInput::Multi(v) => Some(v.len()),
                _ => None,
            })
            .max();
        match n {
            None => self.add_node(name, rate, inputs, num_outputs, special_index),

            // The caller did something like `.freq(Vec::new())`. This is a non-recoverable error.
            Some(0) => panic!("empty multichannel input to {name}"),

            // Expand into n parallel units: each channel slices Multi inputs at ch % len
            // (shorter arrays wrap), passes mono inputs through, then recurses. Nested
            // Multi inputs produce nested results rather than a flat array.
            Some(n) => Signal::Multi(
                (0..n)
                    .map(|ch| {
                        let sub: Vec<UGenInput<'g>> = inputs
                            .iter()
                            .map(|i| match i {
                                UGenInput::Multi(v) => v[ch % v.len()].clone(),
                                mono => mono.clone(),
                            })
                            .collect();
                        self.expand(name, rate, &sub, num_outputs, special_index)
                    })
                    .collect(),
            ),
        }
    }

    /// The expansion base case: all inputs are mono leaves; emit one node.
    fn add_node<'g>(
        &'g self,
        name: &'static str,
        rate: RateMode,
        inputs: &[UGenInput<'g>],
        num_outputs: usize,
        special_index: i16,
    ) -> Signal<'g> {
        let resolved: Vec<Input> = inputs
            .iter()
            .map(|input| match *input {
                UGenInput::Constant(c) => Input::Constant(c),
                UGenInput::Channel(s) => {
                    assert!(
                        core::ptr::eq(s.builder, self),
                        "input signal for {name} belongs to a different SynthDefBuilder"
                    );
                    match s.source {
                        ChannelSource::Param(p) => Input::Param(p),
                        ChannelSource::Node { node, output } => Input::Node { node, output },
                    }
                }
                UGenInput::Multi(_) => unreachable!("expand() leaves only mono inputs"),
            })
            .collect();

        let rate = match rate {
            RateMode::Fixed(r) => r,
            RateMode::MaxOfInputs => inputs
                .iter()
                .map(|i| match i {
                    UGenInput::Channel(s) => s.rate(),
                    _ => Rate::Scalar,
                })
                .max_by_key(|&r| rate_rank(r))
                .unwrap_or(Rate::Scalar),
        };

        let mut nodes = self.nodes.borrow_mut();
        let id = nodes.len() as u32;
        nodes.push(NodeData {
            name,
            rate,
            inputs: resolved,
            num_outputs,
            special_index,
        });

        let output = |output| Channel {
            builder: self,
            source: ChannelSource::Node { node: id, output },
        };
        if num_outputs <= 1 {
            Signal::Mono(output(0))
        } else {
            Signal::Multi(
                (0..num_outputs as u32)
                    .map(|k| Signal::Mono(output(k)))
                    .collect(),
            )
        }
    }

    /// Serialize into a [`SynthDef`]. Arena order is the calc order.
    pub fn build(&self, name: impl Into<String>) -> SynthDef {
        SynthDef {
            name: name.into(),
            params: self.params.borrow().clone(),
            units: self
                .nodes
                .borrow()
                .iter()
                .map(|n| UnitSpec {
                    name: n.name.to_string(),
                    rate: n.rate,
                    inputs: n
                        .inputs
                        .iter()
                        .map(|i| match *i {
                            Input::Constant(c) => InputRef::Constant(c),
                            Input::Param(p) => InputRef::Param(p),
                            Input::Node { node, output } => InputRef::Unit { unit: node, output },
                        })
                        .collect(),
                    num_outputs: n.num_outputs,
                    special_index: n.special_index,
                })
                .collect(),
        }
    }
}
