#!/usr/bin/env python3

import wave
import struct
import numpy as np
from scipy.fftpack import dct
import sys
import heapq
from collections import Counter
from math import log2

# =============================================================================
# Helper Functions
# =============================================================================

def read_wav(filename):
    """Read a mono 16-bit PCM wav file and return samples as a NumPy float array."""
    with wave.open(filename, 'r') as f:
        params = f.getparams()
        nframes = params.nframes
        raw_bytes = f.readframes(nframes)
        samples = np.frombuffer(raw_bytes, dtype='<h').astype(np.float64)
    # Normalize samples to [-1,1]
    samples /= 2 ** 15
    return samples, params

def write_compressed(filename, header_data, bitstream):
    """Write the compressed header and bitstream to a binary file."""
    with open(filename, 'wb') as f:
        f.write(header_data)
        f.write(bitstream)

def block_partition(samples, block_size):
    """Partition the signal into blocks of length block_size. Pad with zeros if needed."""
    n = len(samples)
    pad = 0
    if n % block_size != 0:
        pad = block_size - (n % block_size)
        samples = np.concatenate((samples, np.zeros(pad)))
    blocks = samples.reshape(-1, block_size)
    return blocks, pad

def block_dct(blocks):
    """Apply DCT-II to each block along its rows."""
    return dct(blocks, type=2, norm='ortho', axis=1)

def uniform_quantize(data, step):
    """Uniform scalar quantization: round(data/step)."""
    return np.round(data / step).astype(np.int16)

def entropy(symbol_freq):
    """Calculate the entropy of the symbol distribution."""
    total = sum(symbol_freq.values())
    return -sum((freq / total) * log2(freq / total) for freq in symbol_freq.values() if freq > 0)

# =============================================================================
# Huffman Coding
# =============================================================================

class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(symbols_freq):
    """Build a Huffman tree given a dictionary of symbol frequencies."""
    heap = [HuffmanNode(sym, freq) for sym, freq in symbols_freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        parent = HuffmanNode()
        parent.freq = node1.freq + node2.freq
        parent.left = node1
        parent.right = node2
        heapq.heappush(heap, parent)
    return heapq.heappop(heap)

def build_huffman_codes(root):
    """Build Huffman codes from Huffman tree."""
    codes = {}

    def traverse(node, code):
        if node.symbol is not None:
            codes[node.symbol] = code
            return
        if node.left:
            traverse(node.left, code + '0')
        if node.right:
            traverse(node.right, code + '1')

    traverse(root, "")
    return codes

def huffman_encode(data, codes):
    """Encode the data using Huffman codes."""
    return ''.join(codes[sym] for sym in data)

def pad_bitstring(bitstring):
    """Pad bitstring to a multiple of 8 bits."""
    extra_bits = (8 - (len(bitstring) % 8)) % 8
    bitstring += '0' * extra_bits
    return bitstring, extra_bits

def bitstring_to_bytes(bitstring):
    """Convert a bitstring to bytes."""
    return bytes(int(bitstring[i:i + 8], 2) for i in range(0, len(bitstring), 8))

# =============================================================================
# Main Compression Steps
# =============================================================================

def compress(block_size, quant_step):
    # Read the input wav
    samples, params = read_wav('../step.wav')
    blocks, pad = block_partition(samples, block_size)

    # Apply DCT
    dct_blocks = block_dct(blocks)

    # Quantize DCT coefficients
    quantized = uniform_quantize(dct_blocks, quant_step)
    quantized_flat = quantized.flatten()

    # Frequency analysis
    symbol_freq = Counter(quantized_flat)
    data_entropy = entropy(symbol_freq)

    # Build Huffman tree and codes
    root = build_huffman_tree(symbol_freq)
    codes = build_huffman_codes(root)

    # Encode quantized data
    bitstring = huffman_encode(quantized_flat, codes)
    bitstring, padding_bits = pad_bitstring(bitstring)
    compressed_data = bitstring_to_bytes(bitstring)

    # Prepare header
    nblocks = quantized.shape[0]
    header = bytearray()
    header += struct.pack('<i', block_size)
    header += struct.pack('<i', pad)
    header += struct.pack('<d', quant_step)
    header += struct.pack('<i', nblocks)
    header += struct.pack('<i', padding_bits)
    header += struct.pack('<d', data_entropy)  # Store entropy
    header += struct.pack('<i', len(codes))

    for sym, code in codes.items():
        header += struct.pack('<h', sym)
        header += struct.pack('<h', len(code))
        header += code.encode('ascii')

    write_compressed('compressed', header, compressed_data)
    return len(header) + len(compressed_data)
