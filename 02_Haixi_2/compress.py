#!/usr/bin/env python3

import wave
import struct
import numpy as np
from scipy.fftpack import dct
import heapq

# =============================================================================
# Helper Functions
# =============================================================================

def read_wav(filename):
    """Read a mono 16-bit PCM wav file and return samples as a NumPy float array and params."""
    with wave.open(filename, 'r') as f:
        params = f.getparams()  # (nchannels, sampwidth, framerate, nframes, comptype, compname)
        nframes = params.nframes
        raw_bytes = f.readframes(nframes)
    samples = np.frombuffer(raw_bytes, dtype='<h').astype(np.float64)
    # Normalize samples to [-1,1]
    samples /= 2**15
    return samples, params

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
    """Apply DCT-II to each block."""
    return dct(blocks, type=2, norm='ortho', axis=1)

def uniform_quantize(data, step):
    """Uniform scalar quantization."""
    return np.round(data / step).astype(np.int16)

# =============================================================================
# Huffman Coding
# =============================================================================

class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(symbol_freq):
    """Build a Huffman tree for given symbol frequencies."""
    heap = []
    for sym, freq in symbol_freq.items():
        heapq.heappush(heap, HuffmanNode(sym, freq))

    if len(heap) == 1:
        # Edge case: only one symbol
        node = heapq.heappop(heap)
        root = HuffmanNode()
        root.left = node
        return root

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        parent = HuffmanNode(None, node1.freq + node2.freq, node1, node2)
        heapq.heappush(heap, parent)
    return heap[0]

def generate_codes(node, prefix="", code_map=None):
    if code_map is None:
        code_map = {}
    if node.symbol is not None:
        code_map[node.symbol] = prefix
        return code_map
    if node.left:
        generate_codes(node.left, prefix + '0', code_map)
    if node.right:
        generate_codes(node.right, prefix + '1', code_map)
    return code_map

def pack_code_bits(code_str):
    """Pack a string of '0'/'1' bits into bytes."""
    code_length = len(code_str)
    num_code_bytes = (code_length + 7) // 8
    code_str_padded = code_str.ljust(num_code_bytes * 8, '0')
    code_bytes = bytearray()
    for i in range(0, len(code_str_padded), 8):
        byte = code_str_padded[i:i+8]
        code_bytes.append(int(byte, 2))
    return code_length, num_code_bytes, code_bytes

def huffman_encode(data, code_map):
    """Encode the data using the Huffman code_map. Returns a bitstring."""
    return ''.join(code_map[sym] for sym in data)

def pad_bitstring(bitstring):
    """Pad the bitstring to a multiple of 8 bits."""
    extra_bits = (8 - (len(bitstring) % 8)) % 8
    return bitstring + ('0' * extra_bits), extra_bits

def bitstring_to_bytes(bitstring):
    """Convert a bitstring to bytes."""
    return bytes(int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8))

# =============================================================================
# Main Compression Steps
# =============================================================================

if __name__ == "__main__":
    # Parameters - adjust as needed
    block_size = 1240
    quant_step = 0.0382  # Tune this for desired quality and file size

    # Read the input wav
    samples, params = read_wav('step.wav')
    # Partition into blocks
    blocks, pad = block_partition(samples, block_size)

    # Apply DCT
    dct_blocks = block_dct(blocks)

    # Quantize
    quantized = uniform_quantize(dct_blocks, quant_step)
    quantized_flat = quantized.flatten()

    # Frequency table for Huffman
    symbol_freq = {}
    for val in quantized_flat:
        symbol_freq[val] = symbol_freq.get(val, 0) + 1

    # Build Huffman tree & codes
    root = build_huffman_tree(symbol_freq)
    codes = generate_codes(root)

    # Huffman encode the quantized data
    bitstring = huffman_encode(quantized_flat, codes)
    bitstring, padding_bits = pad_bitstring(bitstring)
    compressed_data = bitstring_to_bytes(bitstring)

    # Prepare header
    # Format:
    # block_size (int32)
    # pad (int32)
    # quant_step (float64)
    # nblocks (int32)
    # padding_bits (int32)
    # huff_entries (int32)
    #
    # For each huffman entry:
    #   symbol (int16)
    #   code_length (uint16)
    #   num_code_bytes (uint8)
    #   code_bytes (...)

    nblocks = quantized.shape[0]
    huff_entries = len(codes)

    header = bytearray()
    header += struct.pack('<i', block_size)
    header += struct.pack('<i', pad)
    header += struct.pack('<d', quant_step)
    header += struct.pack('<i', nblocks)
    header += struct.pack('<i', padding_bits)
    header += struct.pack('<i', huff_entries)

    # Write Huffman table entries
    for sym, code_str in codes.items():
        code_length, num_code_bytes, code_bytes = pack_code_bits(code_str)
        # symbol: int16
        header += struct.pack('<h', sym)
        # code_length: uint16
        header += struct.pack('<H', code_length)
        # num_code_bytes: uint8
        header += struct.pack('<B', num_code_bytes)
        # code_bytes
        header += code_bytes

    # Finally, write out compressed = header + compressed_data
    with open('compressed', 'wb') as f:
        f.write(header)
        f.write(compressed_data)

    print("Compression complete. 'compressed' file created.")
