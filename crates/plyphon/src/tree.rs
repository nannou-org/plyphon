//! The node tree - plyphon's port of scsynth's `Node`/`Group`/`Graph` hierarchy.
//!
//! Nodes live in a fixed-capacity slotmap allocated once at construction, so linking and unlinking
//! on the audio thread is O(1) pointer (index) manipulation with no allocation. Client node ids map
//! to slot indices through a pre-reserved [`HashMap`] that never rehashes while the node count stays
//! within capacity. Synths removed from the tree are handed back to the caller so their `Box` can be
//! dropped off the audio thread.

use std::collections::HashMap;

use crate::io::Io;
use crate::synth::Synth;
use crate::ugen::{DoneAction, ProcessContext};

/// Where to insert a node relative to a target group.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AddAction {
    /// Prepend to the group's children.
    Head,
    /// Append to the group's children.
    Tail,
}

/// A slot in the node arena.
enum Slot {
    Free,
    Node(Node),
}

/// A tree node: its client id, sibling links, paused flag, and kind.
struct Node {
    id: i32,
    parent: Option<u32>,
    prev: Option<u32>,
    next: Option<u32>,
    paused: bool,
    kind: NodeKind,
}

/// A node is either a group (with a child list) or a synth.
enum NodeKind {
    Synth(Box<Synth>),
    Group {
        head: Option<u32>,
        tail: Option<u32>,
    },
}

/// A fixed-capacity tree of groups and synths rooted at a top group.
pub struct NodeTree {
    slots: Vec<Slot>,
    free: Vec<u32>,
    id_map: HashMap<i32, u32>,
    root_id: i32,
    root_index: u32,
}

impl NodeTree {
    /// Create a tree sized for `max_nodes` (including the root group, created with id `root_id`).
    pub fn new(max_nodes: usize, root_id: i32) -> Self {
        let capacity = max_nodes.max(1);
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(Slot::Free);
        }
        // Pop order yields 0, 1, 2, ...; the root takes index 0.
        let mut free: Vec<u32> = (0..capacity as u32).rev().collect();
        let root_index = free.pop().expect("capacity >= 1");
        slots[root_index as usize] = Slot::Node(Node {
            id: root_id,
            parent: None,
            prev: None,
            next: None,
            paused: false,
            kind: NodeKind::Group {
                head: None,
                tail: None,
            },
        });
        let mut id_map = HashMap::with_capacity(capacity);
        id_map.insert(root_id, root_index);
        NodeTree {
            slots,
            free,
            id_map,
            root_id,
            root_index,
        }
    }

    /// The root group's client id.
    pub fn root_id(&self) -> i32 {
        self.root_id
    }

    /// Link a pre-built synth into the tree under group `target`.
    ///
    /// On failure (unknown/non-group target, or the tree is full) the synth is returned so the
    /// caller can route it back to the trash ring.
    pub fn add_synth(
        &mut self,
        id: i32,
        synth: Box<Synth>,
        target: i32,
        action: AddAction,
    ) -> Result<(), Box<Synth>> {
        let group_idx = match self.id_map.get(&target) {
            Some(&i) if self.is_group(i) => i,
            _ => return Err(synth),
        };
        let idx = match self.free.pop() {
            Some(i) => i,
            None => return Err(synth),
        };
        self.slots[idx as usize] = Slot::Node(Node {
            id,
            parent: None,
            prev: None,
            next: None,
            paused: false,
            kind: NodeKind::Synth(synth),
        });
        self.id_map.insert(id, idx);
        self.link_into_group(idx, group_idx, action);
        Ok(())
    }

    /// Create an empty group under group `target`. Returns `false` if it could not be added.
    pub fn add_group(&mut self, id: i32, target: i32, action: AddAction) -> bool {
        let group_idx = match self.id_map.get(&target) {
            Some(&i) if self.is_group(i) => i,
            _ => return false,
        };
        let idx = match self.free.pop() {
            Some(i) => i,
            None => return false,
        };
        self.slots[idx as usize] = Slot::Node(Node {
            id,
            parent: None,
            prev: None,
            next: None,
            paused: false,
            kind: NodeKind::Group {
                head: None,
                tail: None,
            },
        });
        self.id_map.insert(id, idx);
        self.link_into_group(idx, group_idx, action);
        true
    }

    /// Free a node, returning its synth (if it was a leaf synth) for off-RT dropping.
    ///
    /// The root group is never freed. Freeing a group currently just unlinks it (group deep-free is
    /// a later milestone), so callers should free synths individually for now.
    pub fn free_node(&mut self, id: i32) -> Option<Box<Synth>> {
        if id == self.root_id {
            return None;
        }
        let idx = *self.id_map.get(&id)?;
        self.unlink(idx);
        self.id_map.remove(&id);
        let slot = core::mem::replace(&mut self.slots[idx as usize], Slot::Free);
        self.free.push(idx);
        match slot {
            Slot::Node(Node {
                kind: NodeKind::Synth(synth),
                ..
            }) => Some(synth),
            _ => None,
        }
    }

    /// Mutable access to the synth with client id `id`, if it is a synth.
    pub fn synth_mut(&mut self, id: i32) -> Option<&mut Synth> {
        let idx = *self.id_map.get(&id)?;
        match &mut self.slots[idx as usize] {
            Slot::Node(Node {
                kind: NodeKind::Synth(synth),
                ..
            }) => Some(synth.as_mut()),
            _ => None,
        }
    }

    /// Process the whole tree for one block, walking groups head-to-tail. Paused nodes are skipped.
    /// Any node whose synth requested a done action is recorded in `done` as `(slot index, action)`
    /// for the caller to apply after the walk.
    pub fn process(
        &mut self,
        ctx: &ProcessContext<'_>,
        io: &mut Io,
        done: &mut Vec<(u32, DoneAction)>,
    ) {
        let root = self.root_index;
        self.process_group(root, ctx, io, done);
    }

    fn process_group(
        &mut self,
        group_idx: u32,
        ctx: &ProcessContext<'_>,
        io: &mut Io,
        done: &mut Vec<(u32, DoneAction)>,
    ) {
        let mut cur = match &self.slots[group_idx as usize] {
            Slot::Node(Node {
                kind: NodeKind::Group { head, .. },
                ..
            }) => *head,
            _ => None,
        };
        while let Some(idx) = cur {
            let next = match &self.slots[idx as usize] {
                Slot::Node(node) => node.next,
                Slot::Free => None,
            };
            if self.is_group(idx) {
                self.process_group(idx, ctx, io, done);
            } else {
                let active = matches!(&self.slots[idx as usize], Slot::Node(node) if !node.paused);
                if active
                    && let Slot::Node(Node {
                        kind: NodeKind::Synth(synth),
                        ..
                    }) = &mut self.slots[idx as usize]
                {
                    let action = synth.process(ctx, io);
                    if action != DoneAction::Nothing {
                        done.push((idx, action));
                    }
                }
            }
            cur = next;
        }
    }

    /// Free the synth at slot `idx` (done actions locate nodes by index during the walk), returning
    /// its client id and synth for off-RT dropping. No-op for groups or empty slots.
    pub fn free_by_index(&mut self, idx: u32) -> Option<(i32, Box<Synth>)> {
        let id = match &self.slots[idx as usize] {
            Slot::Node(Node {
                id,
                kind: NodeKind::Synth(_),
                ..
            }) => *id,
            _ => return None,
        };
        self.unlink(idx);
        self.id_map.remove(&id);
        let slot = core::mem::replace(&mut self.slots[idx as usize], Slot::Free);
        self.free.push(idx);
        match slot {
            Slot::Node(Node {
                kind: NodeKind::Synth(synth),
                ..
            }) => Some((id, synth)),
            _ => None,
        }
    }

    /// Pause the node at slot `idx`. Returns its client id if found.
    pub fn pause_by_index(&mut self, idx: u32) -> Option<i32> {
        match &mut self.slots[idx as usize] {
            Slot::Node(node) => {
                node.paused = true;
                Some(node.id)
            }
            Slot::Free => None,
        }
    }

    /// Set node `id`'s run state (pausing when `run` is false). Returns the id only if it changed.
    pub fn set_run(&mut self, id: i32, run: bool) -> Option<i32> {
        let idx = *self.id_map.get(&id)?;
        match &mut self.slots[idx as usize] {
            Slot::Node(node) if node.paused == run => {
                node.paused = !run;
                Some(id)
            }
            _ => None,
        }
    }

    fn is_group(&self, idx: u32) -> bool {
        matches!(
            &self.slots[idx as usize],
            Slot::Node(Node {
                kind: NodeKind::Group { .. },
                ..
            })
        )
    }

    fn group_links(&self, idx: u32) -> (Option<u32>, Option<u32>) {
        match &self.slots[idx as usize] {
            Slot::Node(Node {
                kind: NodeKind::Group { head, tail },
                ..
            }) => (*head, *tail),
            _ => (None, None),
        }
    }

    fn set_group_links(&mut self, idx: u32, head: Option<u32>, tail: Option<u32>) {
        if let Slot::Node(Node {
            kind: NodeKind::Group { head: h, tail: t },
            ..
        }) = &mut self.slots[idx as usize]
        {
            *h = head;
            *t = tail;
        }
    }

    fn node_mut(&mut self, idx: u32) -> Option<&mut Node> {
        match &mut self.slots[idx as usize] {
            Slot::Node(node) => Some(node),
            Slot::Free => None,
        }
    }

    fn link_into_group(&mut self, node_idx: u32, group_idx: u32, action: AddAction) {
        let (head, tail) = self.group_links(group_idx);
        if let Some(node) = self.node_mut(node_idx) {
            node.parent = Some(group_idx);
        }
        match action {
            AddAction::Head => {
                if let Some(node) = self.node_mut(node_idx) {
                    node.prev = None;
                    node.next = head;
                }
                match head {
                    Some(h) => {
                        if let Some(hn) = self.node_mut(h) {
                            hn.prev = Some(node_idx);
                        }
                        self.set_group_links(group_idx, Some(node_idx), tail);
                    }
                    None => self.set_group_links(group_idx, Some(node_idx), Some(node_idx)),
                }
            }
            AddAction::Tail => {
                if let Some(node) = self.node_mut(node_idx) {
                    node.prev = tail;
                    node.next = None;
                }
                match tail {
                    Some(t) => {
                        if let Some(tn) = self.node_mut(t) {
                            tn.next = Some(node_idx);
                        }
                        self.set_group_links(group_idx, head, Some(node_idx));
                    }
                    None => self.set_group_links(group_idx, Some(node_idx), Some(node_idx)),
                }
            }
        }
    }

    fn unlink(&mut self, node_idx: u32) {
        let (parent, prev, next) = match self.node_mut(node_idx) {
            Some(node) => (node.parent, node.prev, node.next),
            None => return,
        };
        if let Some(p) = prev
            && let Some(pn) = self.node_mut(p)
        {
            pn.next = next;
        }
        if let Some(nx) = next
            && let Some(nn) = self.node_mut(nx)
        {
            nn.prev = prev;
        }
        if let Some(g) = parent {
            let (mut head, mut tail) = self.group_links(g);
            if head == Some(node_idx) {
                head = next;
            }
            if tail == Some(node_idx) {
                tail = prev;
            }
            self.set_group_links(g, head, tail);
        }
        if let Some(node) = self.node_mut(node_idx) {
            node.parent = None;
            node.prev = None;
            node.next = None;
        }
    }
}
