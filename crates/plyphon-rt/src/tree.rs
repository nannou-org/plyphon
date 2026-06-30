//! The node tree - plyphon's port of scsynth's `Node`/`Group`/`Graph` hierarchy.
//!
//! Nodes live in a fixed-capacity slotmap allocated once at construction, so linking, unlinking, and
//! moving on the audio thread is O(1) pointer (index) manipulation with no allocation. Client node
//! ids map to slot indices through a pre-reserved [`HashMap`] that never rehashes while the node
//! count stays within capacity. Synths removed from the tree are handed back to the caller (through a
//! pre-allocated sink, so freeing even a whole group allocates nothing) for it to reclaim each
//! graph's pool block on the audio thread.
//!
//! This is plyphon's take on scsynth's pooled `Node`s + `mNodeLib`: scsynth `World_Alloc`s each
//! `Node` from the rt-pool and frees it on death; plyphon instead collapses that into one contiguous
//! fixed slab here, so node create/free is O(1) and never touches an allocator. Only the variable-
//! size, churning per-instance *state* is pooled (inside each [`Graph`]'s own `Region`).

use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::command::{NodeNotify, Reply};
use crate::graph::{Block, Graph, Pool};
use plyphon_unit::unit::DoneAction;

/// Where to place a node relative to a target, mirroring scsynth's `addAction` codes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AddAction {
    /// Prepend to the target *group*'s children (`addToHead`, code 0).
    Head,
    /// Append to the target *group*'s children (`addToTail`, code 1).
    Tail,
    /// Immediately before the target *node*, among its siblings (`addBefore`, code 2).
    Before,
    /// Immediately after the target *node*, among its siblings (`addAfter`, code 3).
    After,
    /// Replace the target *node*: the new node takes its exact slot, and the target - with its whole
    /// subtree - is freed (`addReplace`, code 4). Unlike the others this is not a plain placement: it
    /// frees a node, so it runs through the dedicated replace path, not `resolve_placement`.
    Replace,
}

/// A resolved placement: where a node is to be linked, by slot index.
#[derive(Copy, Clone)]
enum Placement {
    /// Head of the group at this index.
    Head(u32),
    /// Tail of the group at this index.
    Tail(u32),
    /// Before the node at `node`, within its parent `group`.
    Before { group: u32, node: u32 },
    /// After the node at `node`, within its parent `group`.
    After { group: u32, node: u32 },
}

/// A node removed by a free, handed back to the caller: its tree position captured at the moment of
/// removal (for `/n_end`, scsynth's `Node_StateMsg` before `Node_Remove`) and, for a synth, the
/// graph whose pool block the caller reclaims via `dealloc`.
pub(crate) type FreedNode = (NodeNotify, Option<Graph>);

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
    Synth(Graph),
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
    /// Live synth count (scsynth's `mNumGraphs`), maintained O(1) on add/free so `NumRunningSynths`
    /// can read it without an O(nodes) scan. Always equals `counts().0`.
    synth_count: usize,
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
            synth_count: 0,
        }
    }

    /// Live synth count (scsynth's `mNumGraphs`), O(1). Equals [`counts`](Self::counts)`.0`.
    pub(crate) fn running_synths(&self) -> usize {
        self.synth_count
    }

    /// The root group's client id.
    pub fn root_id(&self) -> i32 {
        self.root_id
    }

    /// Link a freshly built graph into the tree at `target`/`action`.
    ///
    /// On failure (unresolvable placement, or the tree is full) the graph is returned so the caller
    /// can reclaim its pool block.
    pub(crate) fn add_synth(
        &mut self,
        id: i32,
        graph: Graph,
        target: i32,
        action: AddAction,
    ) -> Result<(), Graph> {
        let placement = match self.resolve_placement(target, action) {
            Some(p) => p,
            None => return Err(graph),
        };
        let idx = match self.free.pop() {
            Some(i) => i,
            None => return Err(graph),
        };
        self.slots[idx as usize] = Slot::Node(Node {
            id,
            parent: None,
            prev: None,
            next: None,
            paused: false,
            kind: NodeKind::Synth(graph),
        });
        self.id_map.insert(id, idx);
        self.link_at(idx, placement);
        self.synth_count += 1;
        Ok(())
    }

    /// Create an empty group at `target`/`action`. Returns `false` if it could not be added.
    pub fn add_group(&mut self, id: i32, target: i32, action: AddAction) -> bool {
        let placement = match self.resolve_placement(target, action) {
            Some(p) => p,
            None => return false,
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
        self.link_at(idx, placement);
        true
    }

    /// Replace `target` with the freshly built synth `graph` (scsynth's `/s_new` `addReplace`): the
    /// new synth `id` takes `target`'s exact slot, and `target` (with its subtree) is freed into
    /// `sink`. The new node keeps its own `id`. On failure (target unknown, target is the root, or the
    /// tree is full) the graph is returned for reclamation and nothing is freed.
    pub(crate) fn replace_with_synth(
        &mut self,
        id: i32,
        graph: Graph,
        target: i32,
        sink: &mut Vec<FreedNode>,
    ) -> Result<(), Graph> {
        let Some((idx, group, prev, next)) = self.vacate_for_replace(target, sink) else {
            return Err(graph);
        };
        self.slots[idx as usize] = Slot::Node(Node {
            id,
            parent: None,
            prev: None,
            next: None,
            paused: false,
            kind: NodeKind::Synth(graph),
        });
        self.id_map.insert(id, idx);
        self.insert(idx, group, prev, next);
        self.synth_count += 1;
        Ok(())
    }

    /// Replace `target` with a fresh empty group `id` (scsynth's `/g_new` `addReplace`): the new group
    /// takes `target`'s slot, and `target` (with its subtree) is freed into `sink`. Returns `false`
    /// if `target` is unknown, is the root, or the tree is full (in which case nothing is freed).
    pub fn replace_with_group(&mut self, id: i32, target: i32, sink: &mut Vec<FreedNode>) -> bool {
        let Some((idx, group, prev, next)) = self.vacate_for_replace(target, sink) else {
            return false;
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
        self.insert(idx, group, prev, next);
        true
    }

    /// Free `target` (and its subtree) and reserve a free slot for a replacement to take its place,
    /// returning `(new slot, parent group, prev sibling, next sibling)` - the position `target`
    /// vacated. `None` if `target` is unknown, is the root, or the tree is full; nothing is freed in
    /// that case (the slot is reserved *before* the free, mirroring scsynth's `Graph_New`-then-
    /// `Node_Replace`, so a full tree fails the replace rather than half-applying it). The replaced
    /// node's `/n_end` reports `-1` links: `target` is unlinked before it is destroyed, just as
    /// scsynth's `Node_Replace` nulls the links before deleting the node.
    fn vacate_for_replace(
        &mut self,
        target: i32,
        sink: &mut Vec<FreedNode>,
    ) -> Option<(u32, u32, Option<u32>, Option<u32>)> {
        let target_idx = match self.id_map.get(&target) {
            Some(&i) if i != self.root_index => i,
            _ => return None,
        };
        let idx = self.free.pop()?;
        let group = self
            .node_parent(target_idx)
            .expect("a non-root node always has a parent group");
        let prev = self.node_prev(target_idx);
        let next = self.node_next(target_idx);
        // Free the replaced node first - unlinked (so its `/n_end` reports `-1`, scsynth's
        // `Node_Replace`), which also releases its id, so the replacement may even reuse it.
        self.unlink(target_idx);
        self.destroy(target_idx, sink);
        Some((idx, group, prev, next))
    }

    /// Move an existing node to `target`/`action` (scsynth's `/g_head`/`/g_tail`/`/n_before`/
    /// `/n_after`/`/n_order`). Returns `false` if the node or placement is invalid, or the move would
    /// put a group inside its own subtree.
    pub fn move_node(&mut self, id: i32, target: i32, action: AddAction) -> bool {
        let node_idx = match self.id_map.get(&id) {
            Some(&i) if i != self.root_index => i,
            _ => return false,
        };
        let placement = match self.resolve_placement(target, action) {
            Some(p) => p,
            None => return false,
        };
        // A node cannot be placed relative to itself, nor moved into itself or its own descendant.
        if let Placement::Before { node, .. } | Placement::After { node, .. } = placement
            && node == node_idx
        {
            return false;
        }
        let dest = self.dest_group(placement);
        if dest == node_idx || self.is_descendant(dest, node_idx) {
            return false;
        }
        self.unlink(node_idx);
        self.link_at(node_idx, placement);
        true
    }

    /// Free node `id`, deeply: a synth is removed; a group is removed along with its whole subtree.
    /// Every removed node is pushed to `sink` (its id, and its boxed synth if it was one) for the
    /// caller to drop and notify off the audio thread. The root is never freed. Returns whether the
    /// node existed.
    pub fn free_node(&mut self, id: i32, sink: &mut Vec<FreedNode>) -> bool {
        if id == self.root_id {
            return false;
        }
        let idx = match self.id_map.get(&id) {
            Some(&i) => i,
            None => return false,
        };
        self.free_at(idx, sink);
        true
    }

    /// Free every node in group `id` (deeply), leaving the group itself empty (scsynth's
    /// `/g_freeAll`). Returns whether the group existed.
    pub fn free_all(&mut self, id: i32, sink: &mut Vec<FreedNode>) -> bool {
        let group_idx = match self.id_map.get(&id) {
            Some(&i) if self.is_group(i) => i,
            _ => return false,
        };
        self.free_all_at(group_idx, sink);
        true
    }

    /// Free every *synth* in group `id` and its subgroups, leaving the group structure intact
    /// (scsynth's `/g_deepFree`). Returns whether the group existed.
    pub fn deep_free(&mut self, id: i32, sink: &mut Vec<FreedNode>) -> bool {
        let group_idx = match self.id_map.get(&id) {
            Some(&i) if self.is_group(i) => i,
            _ => return false,
        };
        self.deep_free_group(group_idx, sink);
        true
    }

    /// Mutable access to the graph with client id `id`, if it is a synth.
    pub(crate) fn synth_mut(&mut self, id: i32) -> Option<&mut Graph> {
        let idx = *self.id_map.get(&id)?;
        match &mut self.slots[idx as usize] {
            Slot::Node(Node {
                kind: NodeKind::Synth(graph),
                ..
            }) => Some(graph),
            _ => None,
        }
    }

    /// Read-only access to the synth with client id `id` (for `/s_get`). `None` if no such synth.
    pub(crate) fn synth(&self, id: i32) -> Option<&Graph> {
        let idx = *self.id_map.get(&id)?;
        self.synth_ref(idx)
    }

    /// Describe node `id`'s tree position (for `/n_query` and the node-lifecycle notifications).
    /// `None` if no such node.
    pub(crate) fn node_info(&self, id: i32) -> Option<NodeNotify> {
        let idx = *self.id_map.get(&id)?;
        Some(self.node_info_at(idx))
    }

    /// Describe the node at slot `idx` (its client id read from the slot). Sibling/parent links that
    /// point at an already-freed (`Slot::Free`) slot read back as `-1`, which is exactly what scsynth
    /// reports during a teardown - a removed predecessor leaves `-1` behind.
    fn node_info_at(&self, idx: u32) -> NodeNotify {
        let parent = self.opt_id(self.node_parent(idx));
        let prev = self.opt_id(self.node_prev(idx));
        let next = self.opt_id(self.node_next(idx));
        let (is_group, head, tail) = if self.is_group(idx) {
            let (h, t) = self.group_links(idx);
            (1, self.opt_id(h), self.opt_id(t))
        } else {
            (0, -1, -1)
        };
        NodeNotify {
            node: self.node_id(idx),
            parent,
            prev,
            next,
            is_group,
            head,
            tail,
        }
    }

    /// Live `(synths, groups, ugens)` counts for `/status` (groups includes the root). A bounded scan
    /// over the slot arena; `/status` is infrequent, so this beats maintaining live counters.
    pub(crate) fn counts(&self) -> (usize, usize, usize) {
        let (mut synths, mut groups, mut ugens) = (0, 0, 0);
        for slot in &self.slots {
            if let Slot::Node(node) = slot {
                match &node.kind {
                    NodeKind::Synth(graph) => {
                        synths += 1;
                        ugens += graph.num_units();
                    }
                    NodeKind::Group { .. } => groups += 1,
                }
            }
        }
        (synths, groups, ugens)
    }

    /// Stream the subtree rooted at group `group` into `out` in pre-order (for `/g_queryTree`),
    /// emitting one [`Reply::QueryTreeNode`] per node (a synth then adds [`Reply::QueryTreeSynth`] and,
    /// when `flag`, one [`Reply::QueryTreeControl`] per control). No-op if `group` is unknown or not a
    /// group. Capped at `out`'s capacity so an adversarial tree can never reallocate on the audio
    /// thread (a capped dump is still well-formed - header + partial body + end).
    pub(crate) fn query_tree(&self, group: i32, flag: bool, pool: &Pool, out: &mut Vec<Reply>) {
        let Some(&idx) = self.id_map.get(&group) else {
            return;
        };
        if self.is_group(idx) {
            self.emit_subtree(idx, flag, pool, out);
        }
    }

    /// Process the whole tree for one block, walking groups head-to-tail. Paused nodes are skipped.
    /// Any node whose synth requested a done action is recorded in `done` as `(slot index, action)`
    /// for the caller to apply after the walk.
    pub(crate) fn process(&mut self, block: &mut Block<'_>, done: &mut Vec<(u32, DoneAction)>) {
        let root = self.root_index;
        self.process_group(root, block, done);
    }

    fn process_group(
        &mut self,
        group_idx: u32,
        block: &mut Block<'_>,
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
                self.process_group(idx, block, done);
            } else {
                let active = matches!(&self.slots[idx as usize], Slot::Node(node) if !node.paused);
                if active
                    && let Slot::Node(Node {
                        id,
                        kind: NodeKind::Synth(synth),
                        ..
                    }) = &mut self.slots[idx as usize]
                {
                    let action = synth.process(block, *id);
                    if action != DoneAction::Nothing {
                        done.push((idx, action));
                    }
                }
            }
            cur = next;
        }
    }

    /// Free the synth at slot `idx`, returning its position (captured while still linked, the `/n_end`
    /// payload) and graph for the caller to `dealloc` (the leaf step of
    /// [`deep_free_group`](Self::deep_free_group)). No-op for groups or empty slots.
    fn free_by_index(&mut self, idx: u32) -> Option<(NodeNotify, Graph)> {
        let info = match &self.slots[idx as usize] {
            Slot::Node(Node {
                kind: NodeKind::Synth(_),
                ..
            }) => self.node_info_at(idx),
            _ => return None,
        };
        self.unlink(idx);
        self.id_map.remove(&info.node);
        let slot = core::mem::replace(&mut self.slots[idx as usize], Slot::Free);
        self.free.push(idx);
        // The early guard above means the removed slot was a synth, so always one fewer.
        self.synth_count = self.synth_count.saturating_sub(1);
        match slot {
            Slot::Node(Node {
                kind: NodeKind::Synth(graph),
                ..
            }) => Some((info, graph)),
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

    /// Free the node at slot `idx` and its whole subtree (a synth, or a group), pushing each removed
    /// node to `sink` with its position captured at removal. The surviving parent/siblings are
    /// repaired *afterwards*, mirroring scsynth's `Node_Dtor` (state message, *then* `Node_Remove`):
    /// the freed node's own `/n_end` reports its real position because it is still linked when its
    /// position is captured.
    fn free_at(&mut self, idx: u32, sink: &mut Vec<FreedNode>) {
        let parent = self.node_parent(idx);
        let prev = self.node_prev(idx);
        let next = self.node_next(idx);
        self.destroy(idx, sink);
        self.repair_parent(parent, prev, next, idx);
    }

    /// Free every node in the group at slot `group_idx` (deeply), leaving the group itself empty.
    fn free_all_at(&mut self, group_idx: u32, sink: &mut Vec<FreedNode>) {
        let mut cur = self.group_links(group_idx).0;
        while let Some(child) = cur {
            let next = self.node_next(child);
            self.destroy(child, sink);
            cur = next;
        }
        self.set_group_links(group_idx, None, None);
    }

    /// Apply the done action a unit requested for the synth at slot `idx` (collected during the tree
    /// walk). Freed nodes stream into `freed` for the caller to `dealloc` and notify off the audio
    /// thread; paused node ids collect in `paused` for the caller to notify. No-op if `idx` is no
    /// longer a live synth - an earlier done action this block may already have freed it as a
    /// neighbour. The neighbour/parent links are resolved before any free, since freeing relinks the
    /// tree; `unlink` keeps the relinking allocation-free, so the chain variants need no scratch.
    pub(crate) fn apply_done_action(
        &mut self,
        idx: u32,
        action: DoneAction,
        freed: &mut Vec<FreedNode>,
        paused: &mut Vec<NodeNotify>,
    ) {
        if !matches!(
            &self.slots[idx as usize],
            Slot::Node(Node {
                kind: NodeKind::Synth(_),
                ..
            })
        ) {
            return;
        }
        match action {
            DoneAction::Nothing => {}
            DoneAction::PauseSelf => {
                if self.pause_by_index(idx).is_some() {
                    paused.push(self.node_info_at(idx));
                }
            }
            DoneAction::FreeSelf => self.free_at(idx, freed),
            DoneAction::FreeSelfAndPrev => {
                if let Some(p) = self.node_prev(idx) {
                    self.free_at(p, freed);
                }
                self.free_at(idx, freed);
            }
            DoneAction::FreeSelfAndNext => {
                if let Some(n) = self.node_next(idx) {
                    self.free_at(n, freed);
                }
                self.free_at(idx, freed);
            }
            DoneAction::FreeSelfAndFreeAllPrev => {
                if let Some(p) = self.node_prev(idx) {
                    if self.is_group(p) {
                        self.free_all_at(p, freed);
                    } else {
                        self.free_at(p, freed);
                    }
                }
                self.free_at(idx, freed);
            }
            DoneAction::FreeSelfAndFreeAllNext => {
                if let Some(n) = self.node_next(idx) {
                    if self.is_group(n) {
                        self.free_all_at(n, freed);
                    } else {
                        self.free_at(n, freed);
                    }
                }
                self.free_at(idx, freed);
            }
            // Repeatedly free the immediate predecessor: each `free_at` relinks `idx` to the
            // next-earlier sibling, so the loop walks to the group head, then frees self.
            DoneAction::FreeSelfToHead => {
                while let Some(p) = self.node_prev(idx) {
                    self.free_at(p, freed);
                }
                self.free_at(idx, freed);
            }
            DoneAction::FreeSelfToTail => {
                while let Some(n) = self.node_next(idx) {
                    self.free_at(n, freed);
                }
                self.free_at(idx, freed);
            }
            DoneAction::FreeSelfPausePrev => {
                if let Some(p) = self.node_prev(idx)
                    && self.pause_by_index(p).is_some()
                {
                    paused.push(self.node_info_at(p));
                }
                self.free_at(idx, freed);
            }
            DoneAction::FreeSelfPauseNext => {
                if let Some(n) = self.node_next(idx)
                    && self.pause_by_index(n).is_some()
                {
                    paused.push(self.node_info_at(n));
                }
                self.free_at(idx, freed);
            }
            DoneAction::FreeSelfAndDeepFreePrev => {
                if let Some(p) = self.node_prev(idx) {
                    if self.is_group(p) {
                        self.deep_free_group(p, freed);
                    } else {
                        self.free_at(p, freed);
                    }
                }
                self.free_at(idx, freed);
            }
            DoneAction::FreeSelfAndDeepFreeNext => {
                if let Some(n) = self.node_next(idx) {
                    if self.is_group(n) {
                        self.deep_free_group(n, freed);
                    } else {
                        self.free_at(n, freed);
                    }
                }
                self.free_at(idx, freed);
            }
            // Empty the enclosing group, which frees self along with every sibling.
            DoneAction::FreeAllInGroup => {
                if let Some(parent) = self.node_parent(idx) {
                    self.free_all_at(parent, freed);
                }
            }
            // Free the enclosing group and its whole subtree (self included). The root is unfreeable.
            DoneAction::FreeGroup => {
                if let Some(parent) = self.node_parent(idx)
                    && parent != self.root_index
                {
                    self.free_at(parent, freed);
                }
            }
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

    /// Resolve a `target`/`action` to a concrete [`Placement`], or `None` if it is invalid (a
    /// head/tail target that is not a group, a before/after target with no parent, or an unknown id).
    fn resolve_placement(&self, target: i32, action: AddAction) -> Option<Placement> {
        let target_idx = *self.id_map.get(&target)?;
        match action {
            AddAction::Head => self
                .is_group(target_idx)
                .then_some(Placement::Head(target_idx)),
            AddAction::Tail => self
                .is_group(target_idx)
                .then_some(Placement::Tail(target_idx)),
            AddAction::Before => self.node_parent(target_idx).map(|group| Placement::Before {
                group,
                node: target_idx,
            }),
            AddAction::After => self.node_parent(target_idx).map(|group| Placement::After {
                group,
                node: target_idx,
            }),
            // Replace is not a plain placement (it frees the target); see `replace_with_synth` /
            // `replace_with_group`. A move command that is handed Replace thus resolves to nothing.
            AddAction::Replace => None,
        }
    }

    /// The group a placement lands a node in.
    fn dest_group(&self, placement: Placement) -> u32 {
        match placement {
            Placement::Head(group) | Placement::Tail(group) => group,
            Placement::Before { group, .. } | Placement::After { group, .. } => group,
        }
    }

    /// Whether `idx` is `ancestor` or sits anywhere below it.
    fn is_descendant(&self, idx: u32, ancestor: u32) -> bool {
        let mut cur = self.node_parent(idx);
        while let Some(p) = cur {
            if p == ancestor {
                return true;
            }
            cur = self.node_parent(p);
        }
        false
    }

    /// Link `node_idx` into the tree per `placement`.
    fn link_at(&mut self, node_idx: u32, placement: Placement) {
        match placement {
            Placement::Head(group) => {
                let (head, _) = self.group_links(group);
                self.insert(node_idx, group, None, head);
            }
            Placement::Tail(group) => {
                let (_, tail) = self.group_links(group);
                self.insert(node_idx, group, tail, None);
            }
            Placement::Before { group, node } => {
                let prev = self.node_prev(node);
                self.insert(node_idx, group, prev, Some(node));
            }
            Placement::After { group, node } => {
                let next = self.node_next(node);
                self.insert(node_idx, group, Some(node), next);
            }
        }
    }

    /// Insert `node_idx` into `group_idx` between siblings `prev` and `next` (either may be `None`,
    /// making it the group's new head/tail).
    fn insert(&mut self, node_idx: u32, group_idx: u32, prev: Option<u32>, next: Option<u32>) {
        if let Some(node) = self.node_mut(node_idx) {
            node.parent = Some(group_idx);
            node.prev = prev;
            node.next = next;
        }
        match prev {
            Some(p) => {
                if let Some(pn) = self.node_mut(p) {
                    pn.next = Some(node_idx);
                }
            }
            None => {
                let (_, tail) = self.group_links(group_idx);
                self.set_group_links(group_idx, Some(node_idx), tail);
            }
        }
        match next {
            Some(n) => {
                if let Some(nn) = self.node_mut(n) {
                    nn.prev = Some(node_idx);
                }
            }
            None => {
                let (head, _) = self.group_links(group_idx);
                self.set_group_links(group_idx, head, Some(node_idx));
            }
        }
    }

    /// Recursively remove `idx` and its whole subtree, pushing each removed node to `sink`. The
    /// caller must have already unlinked `idx` from its parent.
    fn destroy(&mut self, idx: u32, sink: &mut Vec<FreedNode>) {
        let head = match &self.slots[idx as usize] {
            Slot::Node(Node {
                kind: NodeKind::Group { head, .. },
                ..
            }) => *head,
            Slot::Node(_) => None,
            Slot::Free => return,
        };
        let mut cur = head;
        while let Some(child) = cur {
            let next = self.node_next(child);
            self.destroy(child, sink);
            cur = next;
        }
        // Capture the position now, after the children are gone (so a group reports head/tail `-1`)
        // but before this node leaves the slot - its own parent/sibling links are still intact, the
        // `/n_end` payload scsynth sends from `Node_Dtor` before `Node_Remove`.
        let info = match &self.slots[idx as usize] {
            Slot::Node(_) => self.node_info_at(idx),
            Slot::Free => return,
        };
        self.id_map.remove(&info.node);
        let slot = core::mem::replace(&mut self.slots[idx as usize], Slot::Free);
        self.free.push(idx);
        let synth = match slot {
            Slot::Node(Node {
                kind: NodeKind::Synth(synth),
                ..
            }) => Some(synth),
            _ => None,
        };
        if synth.is_some() {
            self.synth_count = self.synth_count.saturating_sub(1);
        }
        sink.push((info, synth));
    }

    /// Free every synth in `group_idx` and its subgroups, keeping the groups.
    fn deep_free_group(&mut self, group_idx: u32, sink: &mut Vec<FreedNode>) {
        let mut cur = self.group_links(group_idx).0;
        while let Some(child) = cur {
            let next = self.node_next(child);
            if self.is_group(child) {
                self.deep_free_group(child, sink);
            } else if let Some((info, synth)) = self.free_by_index(child) {
                sink.push((info, Some(synth)));
            }
            cur = next;
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

    fn node_parent(&self, idx: u32) -> Option<u32> {
        match &self.slots[idx as usize] {
            Slot::Node(node) => node.parent,
            Slot::Free => None,
        }
    }

    fn node_prev(&self, idx: u32) -> Option<u32> {
        match &self.slots[idx as usize] {
            Slot::Node(node) => node.prev,
            Slot::Free => None,
        }
    }

    fn node_next(&self, idx: u32) -> Option<u32> {
        match &self.slots[idx as usize] {
            Slot::Node(node) => node.next,
            Slot::Free => None,
        }
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

    /// The client id of the node at slot `idx`, or `-1` if the slot is free.
    fn node_id(&self, idx: u32) -> i32 {
        match &self.slots[idx as usize] {
            Slot::Node(node) => node.id,
            Slot::Free => -1,
        }
    }

    /// Translate an optional slot index to a client id, `-1` for `None`.
    fn opt_id(&self, idx: Option<u32>) -> i32 {
        idx.map(|i| self.node_id(i)).unwrap_or(-1)
    }

    /// Read-only access to the synth at slot `idx` (for the `/g_queryTree` walk).
    fn synth_ref(&self, idx: u32) -> Option<&Graph> {
        match &self.slots[idx as usize] {
            Slot::Node(Node {
                kind: NodeKind::Synth(graph),
                ..
            }) => Some(graph),
            _ => None,
        }
    }

    /// Direct child count of the group at slot `idx` (head -> next chain).
    fn count_children(&self, idx: u32) -> i32 {
        let mut cur = self.group_links(idx).0;
        let mut n = 0;
        while let Some(c) = cur {
            n += 1;
            cur = self.node_next(c);
        }
        n
    }

    /// Pre-order emit of the subtree at slot `idx` into `out` (see [`query_tree`](Self::query_tree)).
    /// Stops pushing once `out` is at capacity, so it never reallocates on the audio thread.
    fn emit_subtree(&self, idx: u32, flag: bool, pool: &Pool, out: &mut Vec<Reply>) {
        if out.len() >= out.capacity() {
            return;
        }
        if self.is_group(idx) {
            out.push(Reply::QueryTreeNode {
                node: self.node_id(idx),
                num_children: self.count_children(idx),
            });
            let mut cur = self.group_links(idx).0;
            while let Some(child) = cur {
                let next = self.node_next(child);
                self.emit_subtree(child, flag, pool, out);
                cur = next;
            }
        } else {
            out.push(Reply::QueryTreeNode {
                node: self.node_id(idx),
                num_children: -1,
            });
            if let Some(graph) = self.synth_ref(idx) {
                let nparams = graph.num_params();
                if out.len() >= out.capacity() {
                    return;
                }
                out.push(Reply::QueryTreeSynth {
                    num_controls: if flag { nparams as i32 } else { 0 },
                });
                if flag {
                    for p in 0..nparams {
                        if out.len() >= out.capacity() {
                            return;
                        }
                        out.push(Reply::QueryTreeControl {
                            index: p as i32,
                            value: graph.control_value(pool, p).unwrap_or(0.0),
                        });
                    }
                }
            }
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

    fn unlink(&mut self, node_idx: u32) {
        let (parent, prev, next) = match self.node_mut(node_idx) {
            Some(node) => (node.parent, node.prev, node.next),
            None => return,
        };
        self.repair_parent(parent, prev, next, node_idx);
        if let Some(node) = self.node_mut(node_idx) {
            node.parent = None;
            node.prev = None;
            node.next = None;
        }
    }

    /// Repair a surviving parent and siblings after the node at `idx` leaves its sibling chain -
    /// scsynth's `Node_Remove`. `idx`'s own links are deliberately *not* touched (it may already be a
    /// freed slot, as in [`free_at`](Self::free_at)); only the live neighbours that `prev`/`next`/
    /// `parent` point to are fixed up.
    fn repair_parent(
        &mut self,
        parent: Option<u32>,
        prev: Option<u32>,
        next: Option<u32>,
        idx: u32,
    ) {
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
            if head == Some(idx) {
                head = next;
            }
            if tail == Some(idx) {
                tail = prev;
            }
            self.set_group_links(g, head, tail);
        }
    }
}
