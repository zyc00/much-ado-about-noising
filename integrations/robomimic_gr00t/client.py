"""Small GR00T wire client for the isolated RoboMimic simulator environment.

Uses the existing server's ZeroMQ/msgpack_numpy protocol without importing
gr00t.policy.__init__ (which eagerly imports the entire Transformers stack).
"""
import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


def encode(value):
    if isinstance(value, np.ndarray) and value.dtype.hasobject:
        raise TypeError('Object arrays are not supported')
    return mnp.encode(value)


def decode(value):
    if value.get(b'nd', value.get('nd', False)) and value.get(b'kind', value.get('kind')) in (b'O', 'O'):
        raise ValueError('Refusing pickle-bearing object array')
    return mnp.decode(value)


class PolicyClient:
    def __init__(self, host='127.0.0.1', port=5555, timeout_ms=120000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(f'tcp://{host}:{port}')

    def request(self, endpoint, data):
        self.socket.send(msgpack.packb({'endpoint': endpoint, 'data': data}, default=encode))
        raw = self.socket.recv()
        if raw == b'ERROR':
            raise RuntimeError('GR00T server rejected request; inspect server log')
        result = msgpack.unpackb(raw, object_hook=decode, raw=False)
        if isinstance(result, dict) and 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def get_action(self, observation):
        return self.request('get_action', {'observation': observation, 'options': None})

    def reset(self):
        return self.request('reset', {'options': None})

    def close(self):
        self.socket.close(linger=0)
        self.context.term()
