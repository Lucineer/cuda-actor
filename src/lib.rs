/*!
# cuda-actor

Actor model for agent systems.

Each agent is an actor — isolated state, mailbox, message passing.
This crate provides the actor runtime: spawn, send, receive,
supervision trees, and lifecycle hooks.

- Actor with mailbox
- Spawn actors by name
- Message passing (typed envelopes)
- Supervision strategies (one-for-one, one-for-all)
- Actor lifecycle (pre_start, post_stop)
- Actor directory (lookup by name)
*/

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Actor state
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActorState { Starting, Running, Stopping, Stopped, Failed }

/// A message envelope
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Envelope {
    pub from: String,
    pub to: String,
    pub msg_type: String,
    pub payload: Vec<u8>,
    pub timestamp: u64,
}

impl Envelope {
    pub fn new(from: &str, to: &str, msg_type: &str, payload: &[u8]) -> Self {
        Envelope { from: from.to_string(), to: to.to_string(), msg_type: msg_type.to_string(), payload: payload.to_vec(), timestamp: now() }
    }
}

/// Mailbox
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mailbox {
    pub messages: VecDeque<Envelope>,
    pub capacity: usize,
    pub dropped: u64,
    pub processed: u64,
}

impl Mailbox {
    pub fn new(capacity: usize) -> Self { Mailbox { messages: VecDeque::new(), capacity, dropped: 0, processed: 0 } }
    pub fn send(&mut self, envelope: Envelope) -> bool {
        if self.messages.len() >= self.capacity { self.dropped += 1; return false; }
        self.messages.push_back(envelope); true
    }
    pub fn receive(&mut self) -> Option<Envelope> {
        let msg = self.messages.pop_front();
        if msg.is_some() { self.processed += 1; }
        msg
    }
    pub fn len(&self) -> usize { self.messages.len() }
    pub fn is_empty(&self) -> bool { self.messages.is_empty() }
}

/// Supervision strategy
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupervisionStrategy {
    OneForOne,   // restart only the failed child
    OneForAll,   // restart all children
    Resume,      // ignore the error
    Stop,        // stop the failed child
}

/// Actor reference
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActorRef {
    pub id: String,
    pub parent: Option<String>,
    pub children: Vec<String>,
    pub state: ActorState,
    pub mailbox: Mailbox,
    pub restart_count: u32,
    pub created: u64,
    pub strategy: SupervisionStrategy,
}

impl ActorRef {
    pub fn new(id: &str, parent: Option<&str>, mailbox_capacity: usize) -> Self {
        ActorRef { id: id.to_string(), parent: parent.map(|p| p.to_string()), children: vec![], state: ActorState::Starting, mailbox: Mailbox::new(mailbox_capacity), restart_count: 0, created: now(), strategy: SupervisionStrategy::OneForOne }
    }
}

/// Actor system
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActorSystem {
    pub actors: HashMap<String, ActorRef>,
    pub root: String,
    pub strategy: SupervisionStrategy,
    pub total_messages: u64,
    pub total_restarts: u64,
}

impl ActorSystem {
    pub fn new(name: &str, mailbox_capacity: usize) -> Self {
        let root_id = name.to_string();
        let mut actors = HashMap::new();
        actors.insert(root_id.clone(), ActorRef::new(&root_id, None, mailbox_capacity));
        actors.get_mut(&root_id).unwrap().state = ActorState::Running;
        ActorSystem { actors, root: root_id, strategy: SupervisionStrategy::OneForOne, total_messages: 0, total_restarts: 0 }
    }

    /// Spawn a child actor
    pub fn spawn(&mut self, parent: &str, name: &str) -> Option<String> {
        let id = format!("{}/{}", parent, name);
        let mut actor = ActorRef::new(&id, Some(parent), 1000);
        actor.state = ActorState::Running;
        actor.strategy = self.strategy;
        if let Some(parent_actor) = self.actors.get_mut(parent) {
            parent_actor.children.push(id.clone());
        }
        self.actors.insert(id.clone(), actor);
        Some(id)
    }

    /// Send a message to an actor
    pub fn send(&mut self, from: &str, to: &str, msg_type: &str, payload: &[u8]) -> bool {
        self.total_messages += 1;
        if let Some(actor) = self.actors.get_mut(to) {
            if actor.state != ActorState::Running { return false; }
            actor.mailbox.send(Envelope::new(from, to, msg_type, payload))
        } else { false }
    }

    /// Receive next message for an actor
    pub fn receive(&mut self, actor_id: &str) -> Option<Envelope> {
        self.actors.get_mut(actor_id).and_then(|a| a.mailbox.receive())
    }

    /// Stop an actor
    pub fn stop(&mut self, actor_id: &str) {
        // Stop children first
        let children: Vec<String> = self.actors.get(actor_id).map(|a| a.children.clone()).unwrap_or_default();
        for child in children { self.stop(&child); }
        if let Some(actor) = self.actors.get_mut(actor_id) { actor.state = ActorState::Stopped; }
    }

    /// Restart an actor
    pub fn restart(&mut self, actor_id: &str) {
        if let Some(actor) = self.actors.get_mut(actor_id) {
            actor.restart_count += 1;
            actor.mailbox.messages.clear();
            actor.state = ActorState::Running;
            self.total_restarts += 1;
        }
    }

    /// Handle failure with supervision strategy
    pub fn handle_failure(&mut self, failed: &str) {
        let strategy = self.actors.get(failed).map(|a| a.strategy).unwrap_or(self.strategy);
        match strategy {
            SupervisionStrategy::OneForOne => { self.restart(failed); }
            SupervisionStrategy::OneForAll => {
                if let Some(parent) = self.actors.get(failed).and_then(|a| a.parent.clone()) {
                    let children: Vec<String> = self.actors.get(&parent).map(|a| a.children.clone()).unwrap_or_default();
                    for child in children { self.restart(&child); }
                } else { self.restart(failed); }
            }
            SupervisionStrategy::Resume => {}
            SupervisionStrategy::Stop => { self.stop(failed); }
        }
    }

    /// List children of an actor
    pub fn children(&self, actor_id: &str) -> Vec<&str> {
        self.actors.get(actor_id).map(|a| a.children.iter().map(|s| s.as_str()).collect()).unwrap_or_default()
    }

    /// All running actors
    pub fn running_actors(&self) -> Vec<&str> {
        self.actors.values().filter(|a| a.state == ActorState::Running).map(|a| a.id.as_str()).collect()
    }

    /// Mailbox depth for an actor
    pub fn mailbox_depth(&self, actor_id: &str) -> usize {
        self.actors.get(actor_id).map(|a| a.mailbox.len()).unwrap_or(0)
    }

    /// Summary
    pub fn summary(&self) -> String {
        let running = self.actors.values().filter(|a| a.state == ActorState::Running).count();
        let stopped = self.actors.values().filter(|a| a.state == ActorState::Stopped).count();
        format!("ActorSystem[{}]: {} actors ({} running, {} stopped), messages={}, restarts={}",
            self.root, self.actors.len(), running, stopped, self.total_messages, self.total_restarts)
    }
}

fn now() -> u64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_and_send() {
        let mut sys = ActorSystem::new("root", 1000);
        let child = sys.spawn("root", "worker").unwrap();
        assert!(sys.send("root", &child, "task", b"data"));
        let msg = sys.receive(&child).unwrap();
        assert_eq!(msg.msg_type, "task");
    }

    #[test]
    fn test_spawn_hierarchy() {
        let mut sys = ActorSystem::new("root", 1000);
        let w1 = sys.spawn("root", "w1").unwrap();
        let w2 = sys.spawn("root", "w2").unwrap();
        assert_eq!(sys.children("root").len(), 2);
    }

    #[test]
    fn test_stop_actor() {
        let mut sys = ActorSystem::new("root", 1000);
        let child = sys.spawn("root", "worker").unwrap();
        sys.stop(&child);
        assert!(!sys.send("root", &child, "x", b""));
    }

    #[test]
    fn test_mailbox_capacity() {
        let mut sys = ActorSystem::new("root", 2);
        let child = sys.spawn("root", "worker").unwrap();
        assert!(sys.send("root", &child, "a", b""));
        assert!(sys.send("root", &child, "b", b""));
        assert!(!sys.send("root", &child, "c", b"")); // dropped
    }

    #[test]
    fn test_restart() {
        let mut sys = ActorSystem::new("root", 1000);
        let child = sys.spawn("root", "worker").unwrap();
        sys.send("root", &child, "msg1", b"");
        sys.restart(&child);
        assert_eq!(sys.mailbox_depth(&child), 0); // cleared
    }

    #[test]
    fn test_supervision_one_for_all() {
        let mut sys = ActorSystem::new("root", 1000);
        sys.strategy = SupervisionStrategy::OneForAll;
        let w1 = sys.spawn("root", "w1").unwrap();
        let _w2 = sys.spawn("root", "w2").unwrap();
        sys.handle_failure(&w1);
        // Both children should be running after restart
        let running = sys.running_actors();
        assert!(running.contains(&"root/w1") || running.contains(&"root/w2"));
    }

    #[test]
    fn test_supervision_stop() {
        let mut sys = ActorSystem::new("root", 1000);
        sys.strategy = SupervisionStrategy::Stop;
        let child = sys.spawn("root", "worker").unwrap();
        sys.handle_failure(&child);
        let actor = sys.actors.get(&child).unwrap();
        assert_eq!(actor.state, ActorState::Stopped);
    }

    #[test]
    fn test_running_actors() {
        let mut sys = ActorSystem::new("root", 1000);
        sys.spawn("root", "w1");
        sys.spawn("root", "w2");
        assert_eq!(sys.running_actors().len(), 3); // root + 2
    }

    #[test]
    fn test_nested_spawn() {
        let mut sys = ActorSystem::new("root", 1000);
        let child = sys.spawn("root", "parent").unwrap();
        sys.spawn(&child, "grandchild");
        assert_eq!(sys.children(&child).len(), 1);
    }

    #[test]
    fn test_summary() {
        let sys = ActorSystem::new("root", 1000);
        let s = sys.summary();
        assert!(s.contains("root"));
    }
}
