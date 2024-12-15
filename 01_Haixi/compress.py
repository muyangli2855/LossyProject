#!/usr/bin/env python3

import wave
import struct
import numpy as np
from scipy.fftpack import dct
import sys
import heapq


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
        # Write header data first
        f.write(header_data)
        # Write the compressed bitstream
        f.write(bitstream)


def block_partition(samples, block_size):
    """Partition the signal into blocks of length block_size.
       If not divisible, pad with zeros at the end."""
    n = len(samples)
    pad = 0
    if n % block_size != 0:
        pad = block_size - (n % block_size)
        samples = np.concatenate((samples, np.zeros(pad)))
    # Reshape into blocks
    blocks = samples.reshape(-1, block_size)
    return blocks, pad


def block_dct(blocks):
    """Apply DCT-II to each block along its rows."""
    # type=2, norm='ortho' gives a proper orthonormal DCT
    return dct(blocks, type=2, norm='ortho', axis=1)


def block_idct(dct_blocks):
    """Inverse DCT for testing or reference (not needed for compression step)."""
    from scipy.fftpack import idct
    return idct(dct_blocks, type=2, norm='ortho', axis=1)


def uniform_quantize(data, step):
    """Uniform scalar quantization: round(data/step)."""
    return np.round(data / step).astype(np.int16)


def uniform_dequantize(qdata, step):
    """Inverse of uniform quantization: qdata * step."""
    return qdata.astype(np.float64) * step


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
    heap = []
    for sym, freq in symbols_freq.items():
        node = HuffmanNode(sym, freq)
        heapq.heappush(heap, node)

    if len(heap) == 1:
        # Edge case: only one symbol
        node = heapq.heappop(heap)
        root = HuffmanNode()
        root.left = node
        return root

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
    """Encode the data using given Huffman codes. Return a bitstring."""
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
    # Partition into blocks
    blocks, pad = block_partition(samples, block_size)

    # Apply DCT
    dct_blocks = block_dct(blocks)

    # Quantize all DCT coefficients
    quantized = uniform_quantize(dct_blocks, quant_step)
    quantized_flat = quantized.flatten()

    # Build frequency table for Huffman coding
    symbol_freq = {}
    for val in quantized_flat:
        symbol_freq[val] = symbol_freq.get(val, 0) + 1

    # Build Huffman tree and codes
    root = build_huffman_tree(symbol_freq)
    codes = build_huffman_codes(root)

    # Encode quantized data using Huffman codes
    bitstring = huffman_encode(quantized_flat, codes)
    # Pad and convert to bytes
    bitstring, padding_bits = pad_bitstring(bitstring)
    compressed_data = bitstring_to_bytes(bitstring)

    # Prepare header:
    # We'll store:
    # [block_size (int32), pad (int32), quant_step (float64), nblocks (int32), Huffman table size, Huffman table entries...]
    # Then the compressed bitstream.
    nblocks = quantized.shape[0]
    # Store Huffman code table:
    # We'll store it as: number_of_entries, then for each entry: symbol (int16), code_length (int16), code_bits (as string)
    # Convert code strings to a binary representation for compactness.
    huffman_table = []
    for sym, c in codes.items():
        # sym is int16
        code_len = len(c)
        # We'll just store the code as ASCII '0'/'1' characters.
        # (You can improve this by packing bits)
        huffman_table.append((sym, code_len, c))

    header = bytearray()
    # block_size -> int32
    header += struct.pack('<i', block_size)
    # pad -> int32
    header += struct.pack('<i', pad)
    # quant_step -> float64
    header += struct.pack('<d', quant_step)
    # nblocks -> int32
    header += struct.pack('<i', nblocks)
    # padding_bits -> int32
    header += struct.pack('<i', padding_bits)
    # number_of_entries in Huffman table
    header += struct.pack('<i', len(huffman_table))

    # Each entry: symbol (int16), code_len(int16), code as bytes (ASCII)
    for (sym, code_len, code_str) in huffman_table:
        header += struct.pack('<h', sym)
        header += struct.pack('<h', code_len)
        # Store code_str as raw ASCII
        header += code_str.encode('ascii')

    # Write everything out
    write_compressed('compressed', header, compressed_data)

    # print("Compression complete. 'compressed' file created.")
    # print(f"Original size: {samples.nbytes} bytes")
    # print(f"Compressed size: {len(header) + len(compressed_data)} bytes")
    return len(header) + len(compressed_data)

