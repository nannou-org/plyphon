//! A SuperCollider-compatible OSC front-end for the plyphon engine.
//!
//! [`OscDispatcher`] wraps a plyphon [`Controller`] and applies the OSC server commands a typical
//! SuperCollider client sends - `/d_recv`, `/s_new`, `/n_set`, `/n_free`, `/g_new`, the control-bus
//! setters `/c_set`/`/c_setn`, and the control mappers `/n_map`/`/n_mapn` - translating them into
//! [`Controller`] calls. OSC handling is strictly control-side; the audio thread is never involved.
//! `/s_new`, `/n_set`, and `/n_map` accept a string control name, resolved against the node's
//! SynthDef, so the dispatcher tracks which definition each node was created from.

#![forbid(unsafe_code)]

use std::collections::HashMap;

use plyphon::controller::SynthNewError;
use plyphon::synthdef::read::ReadError;
use plyphon::{AddAction, Controller};
use rosc::{OscMessage, OscPacket, OscType};
use thiserror::Error;

/// An error applying an OSC command.
#[derive(Debug, Error)]
pub enum OscError {
    /// The bytes failed to decode as an OSC packet.
    #[error("OSC decode error: {0}")]
    Decode(#[from] rosc::OscError),
    /// The command address is not supported.
    #[error("unsupported OSC command: {0}")]
    UnsupportedCommand(String),
    /// The arguments did not match the command.
    #[error("bad OSC arguments: {0}")]
    BadArguments(&'static str),
    /// The `addAction` code is not one plyphon supports (only head/tail for now).
    #[error("unsupported addAction: {0}")]
    UnsupportedAddAction(i32),
    /// A `/d_recv` payload failed to load as a SynthDef.
    #[error("bad SynthDef: {0}")]
    SynthDef(#[from] ReadError),
    /// A `/s_new` failed to instantiate.
    #[error("s_new failed: {0}")]
    SynthNew(#[from] SynthNewError),
    /// A command ring was full.
    #[error("command queue full")]
    QueueFull,
    /// A control name was not found on the target node's SynthDef.
    #[error("unknown control: {0}")]
    UnknownParam(String),
    /// A node id was referenced whose SynthDef the dispatcher does not know.
    #[error("unknown node: {0}")]
    UnknownNode(i32),
}

/// Applies SuperCollider OSC commands to a plyphon [`Controller`].
pub struct OscDispatcher {
    controller: Controller,
    /// Tracks the SynthDef each live node was created from, for control-name resolution.
    node_defs: HashMap<i32, String>,
}

impl OscDispatcher {
    /// Wrap a controller.
    pub fn new(controller: Controller) -> Self {
        OscDispatcher {
            controller,
            node_defs: HashMap::new(),
        }
    }

    /// Access the wrapped controller (e.g. to add SynthDefs or register custom UGens).
    pub fn controller(&mut self) -> &mut Controller {
        &mut self.controller
    }

    /// Unwrap the controller.
    pub fn into_controller(self) -> Controller {
        self.controller
    }

    /// Decode and apply a single OSC packet from raw bytes.
    pub fn apply_bytes(&mut self, data: &[u8]) -> Result<(), OscError> {
        let (_, packet) = rosc::decoder::decode_udp(data).map_err(OscError::Decode)?;
        self.apply(&packet)
    }

    /// Apply a decoded OSC packet (a message, or every message in a bundle).
    pub fn apply(&mut self, packet: &OscPacket) -> Result<(), OscError> {
        match packet {
            OscPacket::Message(message) => self.message(message),
            OscPacket::Bundle(bundle) => {
                for inner in &bundle.content {
                    self.apply(inner)?;
                }
                Ok(())
            }
        }
    }

    fn message(&mut self, message: &OscMessage) -> Result<(), OscError> {
        match message.addr.as_str() {
            "/d_recv" => self.d_recv(&message.args),
            "/s_new" => self.s_new(&message.args),
            "/n_set" => self.n_set(&message.args),
            "/n_free" => self.n_free(&message.args),
            "/g_new" => self.g_new(&message.args),
            "/c_set" => self.c_set(&message.args),
            "/c_setn" => self.c_setn(&message.args),
            "/n_map" => self.n_map(&message.args),
            "/n_mapn" => self.n_mapn(&message.args),
            other => Err(OscError::UnsupportedCommand(other.to_string())),
        }
    }

    fn d_recv(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let blob = match args.first() {
            Some(OscType::Blob(bytes)) => bytes,
            _ => return Err(OscError::BadArguments("d_recv expects a blob")),
        };
        let defs = plyphon::synthdef::read::parse(blob).map_err(OscError::SynthDef)?;
        for def in defs {
            self.controller.add_synthdef(def);
        }
        Ok(())
    }

    fn s_new(&mut self, args: &[OscType]) -> Result<(), OscError> {
        if args.len() < 4 {
            return Err(OscError::BadArguments(
                "s_new expects name, id, addAction, target",
            ));
        }
        let name = str_arg(&args[0])?.to_string();
        let id = int_arg(&args[1])?;
        let action = add_action(int_arg(&args[2])?)?;
        let target = int_arg(&args[3])?;

        let id = if id < 0 {
            self.controller
                .synth_new(&name, target, action)
                .map_err(OscError::SynthNew)?
        } else {
            self.controller
                .synth_new_with_id(id, &name, target, action)
                .map_err(OscError::SynthNew)?;
            id
        };
        self.node_defs.insert(id, name.clone());
        self.apply_controls(id, Some(&name), &args[4..])
    }

    fn n_set(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("n_set expects a node"))?,
        )?;
        self.apply_controls(node, None, &args[1..])
    }

    fn n_free(&mut self, args: &[OscType]) -> Result<(), OscError> {
        for arg in args {
            let node = int_arg(arg)?;
            self.controller
                .free(node)
                .map_err(|_| OscError::QueueFull)?;
            self.node_defs.remove(&node);
        }
        Ok(())
    }

    fn g_new(&mut self, args: &[OscType]) -> Result<(), OscError> {
        if args.is_empty() || !args.len().is_multiple_of(3) {
            return Err(OscError::BadArguments(
                "g_new expects id, addAction, target triples",
            ));
        }
        for triple in args.chunks_exact(3) {
            let id = int_arg(&triple[0])?;
            let action = add_action(int_arg(&triple[1])?)?;
            let target = int_arg(&triple[2])?;
            if id < 0 {
                self.controller
                    .new_group(target, action)
                    .map_err(|_| OscError::QueueFull)?;
            } else {
                self.controller
                    .new_group_with_id(id, target, action)
                    .map_err(|_| OscError::QueueFull)?;
            }
        }
        Ok(())
    }

    fn c_set(&mut self, args: &[OscType]) -> Result<(), OscError> {
        if args.is_empty() || !args.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("c_set expects bus/value pairs"));
        }
        for pair in args.chunks_exact(2) {
            let bus = bus_index(&pair[0])?;
            let value = float_arg(&pair[1])?;
            self.controller
                .set_control_bus(bus, value)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    fn c_setn(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let mut i = 0;
        while i < args.len() {
            let start = bus_index(&args[i])?;
            let count = count_arg(args.get(i + 1))?;
            i += 2;
            if i + count > args.len() {
                return Err(OscError::BadArguments(
                    "c_setn value count exceeds arguments",
                ));
            }
            for (j, arg) in args[i..i + count].iter().enumerate() {
                self.controller
                    .set_control_bus(start + j as u32, float_arg(arg)?)
                    .map_err(|_| OscError::QueueFull)?;
            }
            i += count;
        }
        Ok(())
    }

    fn n_map(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("n_map expects a node"))?,
        )?;
        let rest = &args[1..];
        if !rest.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("n_map expects control/bus pairs"));
        }
        for pair in rest.chunks_exact(2) {
            let control = self.control_index(node, &pair[0])?;
            let bus = map_bus(&pair[1])?;
            self.controller
                .map_control(node, control, bus)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    fn n_mapn(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("n_mapn expects a node"))?,
        )?;
        let rest = &args[1..];
        if rest.is_empty() || !rest.len().is_multiple_of(3) {
            return Err(OscError::BadArguments(
                "n_mapn expects control/bus/count triples",
            ));
        }
        for triple in rest.chunks_exact(3) {
            let control = self.control_index(node, &triple[0])?;
            let bus = map_bus(&triple[1])?;
            let count = count_arg(Some(&triple[2]))?;
            self.controller
                .map_control_n(node, control, bus, count)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// Resolve a control argument (an `int` index or a `string` name) to a parameter index.
    fn control_index(&self, node: i32, arg: &OscType) -> Result<usize, OscError> {
        match arg {
            OscType::Int(idx) => {
                usize::try_from(*idx).map_err(|_| OscError::BadArguments("negative control index"))
            }
            OscType::String(name) => self.resolve_param(node, None, name),
            _ => Err(OscError::BadArguments("control must be an int or string")),
        }
    }

    /// Apply `(control, value)` argument pairs to `node`. A control is an `int` index or a `string`
    /// name resolved against the node's SynthDef (`def_name` when known, else the tracked one).
    fn apply_controls(
        &mut self,
        node: i32,
        def_name: Option<&str>,
        args: &[OscType],
    ) -> Result<(), OscError> {
        let mut i = 0;
        while i + 1 < args.len() {
            let index = match &args[i] {
                OscType::Int(idx) => usize::try_from(*idx)
                    .map_err(|_| OscError::BadArguments("negative control index"))?,
                OscType::String(name) => self.resolve_param(node, def_name, name)?,
                _ => return Err(OscError::BadArguments("control must be an int or string")),
            };
            let value = float_arg(&args[i + 1])?;
            self.controller
                .set_control(node, index, value)
                .map_err(|_| OscError::QueueFull)?;
            i += 2;
        }
        Ok(())
    }

    fn resolve_param(
        &self,
        node: i32,
        def_name: Option<&str>,
        name: &str,
    ) -> Result<usize, OscError> {
        let def_name = match def_name {
            Some(d) => d,
            None => self
                .node_defs
                .get(&node)
                .map(String::as_str)
                .ok_or(OscError::UnknownNode(node))?,
        };
        let def = self
            .controller
            .synthdef(def_name)
            .ok_or(OscError::UnknownNode(node))?;
        def.param_index(name)
            .ok_or_else(|| OscError::UnknownParam(name.to_string()))
    }
}

/// Map a SuperCollider `addAction` code to a plyphon [`AddAction`] (only head/tail supported).
fn add_action(code: i32) -> Result<AddAction, OscError> {
    match code {
        0 => Ok(AddAction::Head),
        1 => Ok(AddAction::Tail),
        other => Err(OscError::UnsupportedAddAction(other)),
    }
}

fn int_arg(arg: &OscType) -> Result<i32, OscError> {
    match arg {
        OscType::Int(i) => Ok(*i),
        _ => Err(OscError::BadArguments("expected an int")),
    }
}

/// A non-negative bus index (`/c_set`, `/c_setn`).
fn bus_index(arg: &OscType) -> Result<u32, OscError> {
    u32::try_from(int_arg(arg)?).map_err(|_| OscError::BadArguments("negative bus index"))
}

/// A bus index for `/n_map`/`/n_mapn`, where a negative index means "unmap".
fn map_bus(arg: &OscType) -> Result<Option<u32>, OscError> {
    let bus = int_arg(arg)?;
    Ok(u32::try_from(bus).ok())
}

/// A non-negative count argument (`/c_setn`, `/n_mapn`).
fn count_arg(arg: Option<&OscType>) -> Result<usize, OscError> {
    let arg = arg.ok_or(OscError::BadArguments("expected a count"))?;
    usize::try_from(int_arg(arg)?).map_err(|_| OscError::BadArguments("negative count"))
}

fn float_arg(arg: &OscType) -> Result<f32, OscError> {
    match arg {
        OscType::Float(f) => Ok(*f),
        OscType::Int(i) => Ok(*i as f32),
        OscType::Double(d) => Ok(*d as f32),
        _ => Err(OscError::BadArguments("expected a number")),
    }
}

fn str_arg(arg: &OscType) -> Result<&str, OscError> {
    match arg {
        OscType::String(s) => Ok(s.as_str()),
        _ => Err(OscError::BadArguments("expected a string")),
    }
}
