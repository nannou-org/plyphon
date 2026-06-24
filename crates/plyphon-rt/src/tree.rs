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

use crate::graph::{Block, Graph};
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

/// A node removed by a free, handed back to the caller: its id and (for synths) the graph, whose
/// pool block the caller reclaims via `dealloc`.
pub(crate) type FreedNode = (i32, Option<Graph>);

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
        self.unlink(idx);
        self.destroy(idx, sink);
        true
    }

    /// Free every node in group `id` (deeply), leaving the group itself empty (scsynth's
    /// `/g_freeAll`). Returns whether the group existed.
    pub fn free_all(&mut self, id: i32, sink: &mut Vec<FreedNode>) -> bool {
        let group_idx = match self.id_map.get(&id) {
            Some(&i) if self.is_group(i) => i,
            _ => return false,
        };
        let mut cur = self.group_links(group_idx).0;
        while let Some(child) = cur {
            let next = self.node_next(child);
            self.destroy(child, sink);
            cur = next;
        }
        self.set_group_links(group_idx, None, None);
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
                        kind: NodeKind::Synth(synth),
                        ..
                    }) = &mut self.slots[idx as usize]
                {
                    let action = synth.process(block);
                    if action != DoneAction::Nothing {
                        done.push((idx, action));
                    }
                }
            }
            cur = next;
        }
    }

    /// Free the synth at slot `idx` (done actions locate nodes by index during the walk), returning
    /// its client id and graph for the caller to `dealloc`. No-op for groups or empty slots.
    pub(crate) fn free_by_index(&mut self, idx: u32) -> Option<(i32, Graph)> {
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
                kind: NodeKind::Synth(graph),
                ..
            }) => Some((id, graph)),
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
        let id = match &self.slots[idx as usize] {
            Slot::Node(node) => node.id,
            Slot::Free => return,
        };
        self.id_map.remove(&id);
        let slot = core::mem::replace(&mut self.slots[idx as usize], Slot::Free);
        self.free.push(idx);
        let synth = match slot {
            Slot::Node(Node {
                kind: NodeKind::Synth(synth),
                ..
            }) => Some(synth),
            _ => None,
        };
        sink.push((id, synth));
    }

    /// Free every synth in `group_idx` and its subgroups, keeping the groups.
    fn deep_free_group(&mut self, group_idx: u32, sink: &mut Vec<FreedNode>) {
        let mut cur = self.group_links(group_idx).0;
        while let Some(child) = cur {
            let next = self.node_next(child);
            if self.is_group(child) {
                self.deep_free_group(child, sink);
            } else if let Some((id, synth)) = self.free_by_index(child) {
                sink.push((id, Some(synth)));
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
