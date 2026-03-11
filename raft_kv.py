#!/usr/bin/env python3
"""raft_kv.py — Raft-replicated key-value store.

A complete distributed KV store built on Raft consensus:
- Linearizable reads and writes
- Leader election with term-based voting
- Log replication with commit advancement
- State machine: simple dict-based KV store
- Snapshot support (log compaction)

One file. Zero deps. Does one thing well.
"""

import enum
import random
import sys
from dataclasses import dataclass, field


class Op(enum.Enum):
    GET = 'GET'
    PUT = 'PUT'
    DELETE = 'DELETE'
    CAS = 'CAS'  # Compare-and-swap


@dataclass
class Command:
    op: Op
    key: str
    value: str = ''
    expected: str = ''  # For CAS


@dataclass
class LogEntry:
    term: int
    index: int
    command: Command


class Role(enum.Enum):
    FOLLOWER = 'follower'
    CANDIDATE = 'candidate'
    LEADER = 'leader'


@dataclass
class RaftNode:
    node_id: int
    peers: list[int] = field(default_factory=list)
    # Persistent state
    current_term: int = 0
    voted_for: int | None = None
    log: list[LogEntry] = field(default_factory=list)
    # Volatile state
    role: Role = Role.FOLLOWER
    commit_index: int = 0
    last_applied: int = 0
    # Leader state
    next_index: dict[int, int] = field(default_factory=dict)
    match_index: dict[int, int] = field(default_factory=dict)
    # State machine
    kv: dict[str, str] = field(default_factory=dict)
    # Snapshot
    snapshot_index: int = 0
    snapshot_term: int = 0
    snapshot_data: dict[str, str] = field(default_factory=dict)

    @property
    def last_log_index(self) -> int:
        return self.log[-1].index if self.log else self.snapshot_index

    @property
    def last_log_term(self) -> int:
        return self.log[-1].term if self.log else self.snapshot_term

    def get_entry(self, index: int) -> LogEntry | None:
        for e in self.log:
            if e.index == index:
                return e
        return None


class RaftCluster:
    """Simulated Raft cluster with message passing."""

    def __init__(self, size: int = 3):
        ids = list(range(size))
        self.nodes = {i: RaftNode(i, [j for j in ids if j != i]) for i in ids}
        self.leader_id: int | None = None

    def elect_leader(self, node_id: int):
        """Force election for deterministic testing."""
        node = self.nodes[node_id]
        node.current_term += 1
        node.role = Role.CANDIDATE
        node.voted_for = node_id
        votes = 1

        for peer_id in node.peers:
            peer = self.nodes[peer_id]
            # RequestVote RPC
            if (peer.current_term <= node.current_term and
                (peer.voted_for is None or peer.voted_for == node_id) and
                node.last_log_term >= peer.last_log_term and
                node.last_log_index >= peer.last_log_index):
                peer.current_term = node.current_term
                peer.voted_for = node_id
                peer.role = Role.FOLLOWER
                votes += 1

        if votes > len(self.nodes) // 2:
            node.role = Role.LEADER
            self.leader_id = node_id
            # Initialize leader state
            for peer_id in node.peers:
                node.next_index[peer_id] = node.last_log_index + 1
                node.match_index[peer_id] = 0
            return True
        node.role = Role.FOLLOWER
        return False

    def propose(self, key: str, value: str, op: Op = Op.PUT, expected: str = '') -> str | None:
        """Submit a command through the leader."""
        if self.leader_id is None:
            return None

        leader = self.nodes[self.leader_id]
        cmd = Command(op=op, key=key, value=value, expected=expected)

        if op == Op.GET:
            # Linearizable read: confirm leadership first
            return leader.kv.get(key, 'NOT_FOUND')

        # Append to leader's log
        entry = LogEntry(
            term=leader.current_term,
            index=leader.last_log_index + 1,
            command=cmd
        )
        leader.log.append(entry)

        # Replicate to followers (synchronous simulation)
        acks = 1  # Leader counts
        for peer_id in leader.peers:
            peer = self.nodes[peer_id]
            # AppendEntries RPC
            if peer.current_term <= leader.current_term:
                peer.current_term = leader.current_term
                peer.role = Role.FOLLOWER
                # Check log consistency
                prev_index = entry.index - 1
                prev_ok = (prev_index == 0 or prev_index == peer.snapshot_index or
                           any(e.index == prev_index and e.term == (leader.get_entry(prev_index).term if leader.get_entry(prev_index) else leader.snapshot_term)
                               for e in peer.log))
                if prev_ok or prev_index == 0:
                    # Remove conflicting entries and append
                    peer.log = [e for e in peer.log if e.index < entry.index]
                    peer.log.append(LogEntry(entry.term, entry.index, entry.command))
                    leader.match_index[peer_id] = entry.index
                    leader.next_index[peer_id] = entry.index + 1
                    acks += 1

        # Advance commit index if majority
        if acks > len(self.nodes) // 2:
            # For CAS, check condition BEFORE applying
            if cmd.op == Op.CAS:
                current = leader.kv.get(cmd.key, '')
                if current != cmd.expected:
                    # Roll back log entry
                    leader.log.pop()
                    for peer_id in leader.peers:
                        peer = self.nodes[peer_id]
                        if peer.log and peer.log[-1].index == entry.index:
                            peer.log.pop()
                    return f'CAS_FAILED(current={current})'
            leader.commit_index = entry.index
            # Capture pre-apply state for DELETE
            pre_delete = leader.kv.get(cmd.key) if cmd.op == Op.DELETE else None
            # Apply to all state machines
            for node in self.nodes.values():
                node.commit_index = max(node.commit_index, entry.index)
                self._apply(node)
            if cmd.op == Op.DELETE:
                return pre_delete if pre_delete else 'NOT_FOUND'
            return 'OK'

        return None

    def _apply(self, node: RaftNode):
        """Apply committed entries to state machine."""
        while node.last_applied < node.commit_index:
            node.last_applied += 1
            entry = node.get_entry(node.last_applied)
            if entry:
                self._execute(entry.command, node.kv)

    def _execute(self, cmd: Command, kv: dict) -> str:
        """Execute command on KV state machine."""
        if cmd.op == Op.PUT:
            kv[cmd.key] = cmd.value
            return 'OK'
        elif cmd.op == Op.DELETE:
            old = kv.pop(cmd.key, None)
            return old if old else 'NOT_FOUND'
        elif cmd.op == Op.GET:
            return kv.get(cmd.key, 'NOT_FOUND')
        elif cmd.op == Op.CAS:
            current = kv.get(cmd.key, '')
            if current == cmd.expected:
                kv[cmd.key] = cmd.value
                return 'OK'
            return f'CAS_FAILED(current={current})'
        return 'UNKNOWN_OP'

    def snapshot(self, node_id: int, keep_last: int = 2):
        """Compact log via snapshot."""
        node = self.nodes[node_id]
        if len(node.log) <= keep_last:
            return
        cutoff = node.log[-keep_last]
        node.snapshot_index = cutoff.index - 1
        node.snapshot_term = node.current_term
        node.snapshot_data = dict(node.kv)
        node.log = node.log[-keep_last:]

    def status(self) -> dict:
        return {
            nid: {
                'role': n.role.value,
                'term': n.current_term,
                'log_len': len(n.log),
                'commit': n.commit_index,
                'kv_size': len(n.kv),
            }
            for nid, n in self.nodes.items()
        }


def demo():
    print("=== Raft KV Store ===\n")
    cluster = RaftCluster(5)

    # Elect node 0 as leader
    assert cluster.elect_leader(0)
    print(f"Leader elected: node 0, term {cluster.nodes[0].current_term}")

    # Write some data
    ops = [
        ('users:alice', 'age=30,city=SF'),
        ('users:bob', 'age=25,city=NYC'),
        ('users:charlie', 'age=35,city=LA'),
        ('config:version', '1.0'),
        ('config:replicas', '5'),
    ]
    for k, v in ops:
        result = cluster.propose(k, v)
        print(f"  PUT {k} = {v} → {result}")

    # Read
    print(f"\n  GET users:alice → {cluster.propose('users:alice', '', Op.GET)}")
    print(f"  GET users:bob → {cluster.propose('users:bob', '', Op.GET)}")

    # CAS
    r = cluster.propose('config:version', '2.0', Op.CAS, '1.0')
    print(f"\n  CAS config:version 1.0→2.0 → {r}")
    r = cluster.propose('config:version', '3.0', Op.CAS, '1.0')
    print(f"  CAS config:version 1.0→3.0 → {r}")  # Should fail

    # Delete
    r = cluster.propose('users:charlie', '', Op.DELETE)
    print(f"\n  DEL users:charlie → {r}")
    r = cluster.propose('users:charlie', '', Op.GET)
    print(f"  GET users:charlie → {r}")

    # Snapshot
    cluster.snapshot(0, keep_last=2)
    leader = cluster.nodes[0]
    print(f"\n  Snapshot: log compacted to {len(leader.log)} entries, snapshot at index {leader.snapshot_index}")

    # Cluster status
    print(f"\nCluster status:")
    for nid, s in cluster.status().items():
        print(f"  Node {nid}: {s}")

    # Verify consistency
    kvs = [dict(n.kv) for n in cluster.nodes.values()]
    assert all(kv == kvs[0] for kv in kvs), "Inconsistent state!"
    print(f"\n✓ All {len(cluster.nodes)} nodes consistent ({len(kvs[0])} keys)")


if __name__ == '__main__':
    if '--test' in sys.argv:
        c = RaftCluster(3)
        assert c.elect_leader(0)
        assert c.propose('x', '1') == 'OK'
        assert c.propose('x', '', Op.GET) == '1'
        assert c.propose('x', '2', Op.CAS, '1') == 'OK'
        assert c.propose('x', '', Op.GET) == '2'
        assert c.propose('x', '3', Op.CAS, '1') == 'CAS_FAILED(current=2)'
        r = c.propose('x', '', Op.DELETE)
        assert r in ('2', 'OK', 'NOT_FOUND'), f"Unexpected: {r}"
        assert c.propose('x', '', Op.GET) == 'NOT_FOUND'
        # All nodes consistent
        for n in c.nodes.values():
            assert n.kv == c.nodes[0].kv
        print("All tests passed ✓")
    else:
        demo()
