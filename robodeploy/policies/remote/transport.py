"""
IPolicyTransport — wire protocol abstraction for distributed inference.

Separates "how to serialize and send an Observation" from "what policy
to run". This means RemotePolicy doesn't care whether it talks gRPC, ZMQ,
or HTTP — only the transport changes.

Topology:

    ┌─────────────────────┐  network   ┌──────────────────────────┐
    │  Robot machine      │ ─────────► │  GPU inference server    │
    │                     │            │                          │
    │  ControlLoop (100Hz)│            │  PolicyServer            │
    │  InferenceLoop      │            │    └── any IPolicy       │
    │    └── RemotePolicy │            │    └── IPolicyTransport  │
    │         └── Transport────────────►                          │
    └─────────────────────┘            └──────────────────────────┘

The transport is responsible for:
  1. Serializing Observation → bytes (or dict)
  2. Sending to the server
  3. Receiving Action bytes (or dict) back
  4. Deserializing → Action

Two transports are provided:
  GrpcTransport  — production: low latency, typed, streaming support.
                   Requires: grpcio, grpcio-tools, generated stubs.
  ZmqTransport   — development/research: zero setup, push/pull or req/rep.
                   Requires: pyzmq. Ideal for same-machine or LAN testing.

Adding a new transport: implement IPolicyTransport (3 methods), pass it to
RemotePolicy. No other changes anywhere.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod

import numpy as np

from robodeploy.core.types import Action, Observation


# ---------------------------------------------------------------------------
# Serialization helpers — shared by all transports
# ---------------------------------------------------------------------------

def obs_to_bytes(obs: Observation) -> bytes:
    """Serialize an Observation to bytes using pickle.

    For production, replace with a faster serializer (msgpack, flatbuffers,
    or the generated gRPC proto). Pickle is used here to avoid a schema
    dependency during development.

    Args:
        obs: Observation to serialize.

    Returns:
        Bytes representation.
    """
    return pickle.dumps(obs)


def bytes_to_obs(data: bytes) -> Observation:
    """Deserialize bytes to an Observation."""
    return pickle.loads(data)


def action_to_bytes(action: Action) -> bytes:
    """Serialize an Action to bytes."""
    return pickle.dumps(action)


def bytes_to_action(data: bytes) -> Action:
    """Deserialize bytes to an Action."""
    return pickle.loads(data)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class IPolicyTransport(ABC):
    """Wire protocol for sending observations and receiving actions.

    Implement this to add a new transport (HTTP, Shared Memory, etc.).
    All transports must be thread-safe: RemotePolicy.get_action() may be
    called from the InferenceLoop thread.
    """

    @abstractmethod
    def connect(self) -> None:
        """Open connection to the PolicyServer.

        Called once by RemotePolicy before the first get_action() call.

        Raises:
            ConnectionRefusedError: If the server is not reachable.
        """
        ...

    @abstractmethod
    def send_obs_recv_action(self, obs: Observation) -> Action:
        """Send an observation, block until the server returns an action.

        This is the hot path — called every inference step. Keep fast.

        Args:
            obs: Observation to send to the remote policy.

        Returns:
            Action computed by the remote policy.

        Raises:
            TimeoutError:   If the server does not respond within the timeout.
            RuntimeError:   If the connection drops mid-request.
        """
        ...

    @abstractmethod
    def send_reset(self) -> None:
        """Notify the server to reset the policy's episode state.

        Called by RemotePolicy.reset() at the start of each episode.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the connection. Called by RemotePolicy.close()."""
        ...


# ---------------------------------------------------------------------------
# ZMQ transport (development / research)
# ---------------------------------------------------------------------------

class ZmqTransport(IPolicyTransport):
    """REQ/REP ZMQ transport. Zero setup — ideal for same-machine or LAN.

    Protocol:
        Client sends: b"ACT" + obs_bytes
        Client sends: b"RST" (reset signal)
        Server replies: action_bytes for ACT, b"OK" for RST

    Args:
        host:        Hostname or IP of the PolicyServer.
        port:        TCP port the server listens on.
        timeout_ms:  Receive timeout in milliseconds (default 5000).
    """

    def __init__(
        self,
        host:       str = "localhost",
        port:       int = 5555,
        timeout_ms: int = 5000,
    ) -> None:
        self._addr       = f"tcp://{host}:{port}"
        self._timeout_ms = timeout_ms
        self._socket     = None

    def connect(self) -> None:
        try:
            import zmq  # type: ignore
        except ImportError:
            raise ImportError(
                "ZmqTransport requires pyzmq.\n"
                "Install with: pip install pyzmq"
            )
        ctx            = zmq.Context.instance()
        self._socket   = ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.connect(self._addr)

    def send_obs_recv_action(self, obs: Observation) -> Action:
        if self._socket is None:
            raise RuntimeError("ZmqTransport not connected. Call connect() first.")
        self._socket.send(b"ACT" + obs_to_bytes(obs))
        reply = self._socket.recv()
        return bytes_to_action(reply)

    def send_reset(self) -> None:
        if self._socket is None:
            return
        self._socket.send(b"RST")
        self._socket.recv()   # consume b"OK"

    def close(self) -> None:
        if self._socket:
            self._socket.close()
            self._socket = None


# ---------------------------------------------------------------------------
# gRPC transport (production)
# ---------------------------------------------------------------------------

class GrpcTransport(IPolicyTransport):
    """gRPC transport for production deployments.

    Requires generated stubs from robodeploy/policies/remote/proto/policy.proto.
    Generate with:
        python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/policy.proto

    Until the proto file and generated stubs are added, this class raises
    ImportError with clear instructions. The interface is fully defined.

    Args:
        host:        Hostname or IP of the PolicyServer.
        port:        gRPC port (default 50051).
        timeout_s:   RPC deadline in seconds (default 5.0).
        credentials: Optional gRPC channel credentials for TLS.
    """

    def __init__(
        self,
        host:        str   = "localhost",
        port:        int   = 50051,
        timeout_s:   float = 5.0,
        credentials        = None,
    ) -> None:
        self._target      = f"{host}:{port}"
        self._timeout_s   = timeout_s
        self._credentials = credentials
        self._stub        = None
        self._channel     = None

    def connect(self) -> None:
        try:
            import grpc  # type: ignore
            # from robodeploy.policies.remote.proto import policy_pb2_grpc  # generated stubs
        except ImportError:
            raise ImportError(
                "GrpcTransport requires grpcio.\n"
                "Install with: pip install grpcio grpcio-tools\n"
                "Then generate stubs from policies/remote/proto/policy.proto"
            )
        # Stub connection — activate once proto stubs are generated:
        # if self._credentials:
        #     self._channel = grpc.secure_channel(self._target, self._credentials)
        # else:
        #     self._channel = grpc.insecure_channel(self._target)
        # self._stub = policy_pb2_grpc.PolicyServiceStub(self._channel)
        raise NotImplementedError(
            "GrpcTransport requires proto stubs. "
            "Generate them from policies/remote/proto/policy.proto and uncomment the stub code."
        )

    def send_obs_recv_action(self, obs: Observation) -> Action:
        # request = policy_pb2.ObsRequest(payload=obs_to_bytes(obs))
        # response = self._stub.GetAction(request, timeout=self._timeout_s)
        # return bytes_to_action(response.payload)
        raise NotImplementedError("GrpcTransport stubs not yet generated.")

    def send_reset(self) -> None:
        # self._stub.Reset(policy_pb2.ResetRequest(), timeout=self._timeout_s)
        raise NotImplementedError("GrpcTransport stubs not yet generated.")

    def close(self) -> None:
        if self._channel:
            self._channel.close()
