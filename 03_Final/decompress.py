#!/usr/bin/env python
import struct, wave
import numpy as np
from scipy.fftpack import idct

if __name__ == '__main__':
    with open('compressed', 'rb') as f: data = f.read()
    o = 0
    b, p = struct.unpack_from('<ii', data, o); o += 8
    q = struct.unpack_from('<d', data, o)[0]; o += 8
    n, pb, he = struct.unpack_from('<iii', data, o); o += 12

    stc = {}
    for _ in range(he):
        s = struct.unpack_from('<h', data, o)[0]; o += 2
        cl = struct.unpack_from('<H', data, o)[0]; o += 2
        nb = struct.unpack_from('<B', data, o)[0]; o += 1
        cb = data[o:o+nb]; o += nb
        stc[s] = ''.join(bin(c)[2:].zfill(8) for c in cb)[:cl]
    bs = ''.join(bin(c)[2:].zfill(8) for c in data[o:])
    bs = bs[:-pb] if pb > 0 else bs

    cts = {v: k for k, v in stc.items()}
    ds, c = [], ''
    for b in bs:
        c += b
        if c in cts:
            ds.append(int(cts[c]))  # Convert symbol to integer
            c = ''
    qb = np.array(ds, dtype=np.int16).reshape(n, b)
    db = qb.astype(np.float64) * q
    tb = idct(db, type=2, norm='ortho', axis=1).flatten()
    if p > 0: tb = tb[:-p]
    i = (np.clip(tb, -1.0, 1.0) * (2**15)).astype('<h')

    with wave.open('out.wav', 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(44100)
        f.writeframes(i.tobytes())
