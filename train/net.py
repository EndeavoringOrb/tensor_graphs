import base64
import json
import socket
import struct

import numpy as np


def create_server_socket(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(5)
    return sock


def create_client_socket(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    return sock


def _encode_value(value):
    """Encode protocol values without invoking executable deserialization."""
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": base64.b64encode(np.ascontiguousarray(value).tobytes()).decode("ascii"),
        }
    if hasattr(value, "detach") and hasattr(value, "dtype") and hasattr(value, "shape"):
        tensor = value.detach().cpu().contiguous()
        import torch

        raw = tensor.view(torch.uint8).numpy().tobytes()
        return {
            "__tensor__": True,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "data": base64.b64encode(raw).decode("ascii"),
        }
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported network value type: {type(value).__name__}")


def _decode_value(value):
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("__bytes__") is not None:
        return base64.b64decode(value["__bytes__"], validate=True)
    raw = base64.b64decode(value["data"], validate=True) if value.get("data") else b""
    shape = tuple(int(dim) for dim in value.get("shape", []))
    if value.get("__ndarray__"):
        return np.frombuffer(raw, dtype=np.dtype(value["dtype"])).reshape(shape).copy()
    if value.get("__tensor__"):
        import torch

        dtype = getattr(torch, value["dtype"], None)
        if dtype is None:
            raise ValueError(f"Unsupported tensor dtype: {value['dtype']}")
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()
    return {key: _decode_value(item) for key, item in value.items()}


def send_msg(sock: socket.socket, msg: dict) -> None:
    envelope = {"version": 1, "message": _encode_value(msg)}
    data = json.dumps(envelope, separators=(",", ":"), allow_nan=False).encode("utf-8")
    sock.sendall(struct.pack(">I", len(data)) + data)


def recvall(sock: socket.socket, n: int) -> bytearray | None:
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def recv_msg(sock: socket.socket) -> dict | None:
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack(">I", raw_msglen)[0]
    data = recvall(sock, msglen)
    if data is None:
        return None
    envelope = json.loads(bytes(data).decode("utf-8"))
    if envelope.get("version") != 1 or not isinstance(envelope.get("message"), dict):
        raise ValueError("Unsupported training message envelope")
    return _decode_value(envelope["message"])
