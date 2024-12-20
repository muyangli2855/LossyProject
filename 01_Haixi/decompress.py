#!/usr/bin/env python

import struct
import wave
import numpy as np
from scipy.fftpack import idct

def read_compressed(filename):
    """Read the compressed file and parse the header."""
    with open(filename, 'rb') as f:
        data = f.read()

    # Parse header:
    # block_size (int32), pad (int32), quant_step (float64), nblocks (int32), padding_bits(int32), huff_entries(int32)
    offset = 0
    block_size = struct.unpack_from('<i', data, offset)[0]
    offset += 4
    pad = struct.unpack_from('<i', data, offset)[0]
    offset += 4
    quant_step = struct.unpack_from('<d', data, offset)[0]
    offset += 8
    nblocks = struct.unpack_from('<i', data, offset)[0]
    offset += 4
    padding_bits = struct.unpack_from('<i', data, offset)[0]
    offset += 4
    huff_entries = struct.unpack_from('<i', data, offset)[0]
    offset += 4

    # Read Huffman table entries
    # Each entry: symbol(int16), code_len(int16), code_str(ascii)
    huffman_table = {}
    for _ in range(huff_entries):
        symbol = struct.unpack_from('<h', data, offset)[0]
        offset += 2
        code_len = struct.unpack_from('<h', data, offset)[0]
        offset += 2
        code_str = data[offset:offset+code_len].decode('ascii')
        offset += code_len
        huffman_table[symbol] = code_str

    # The rest is compressed bitstream
    compressed_bitstream = data[offset:]

    return block_size, pad, quant_step, nblocks, padding_bits, huffman_table, compressed_bitstream

def build_code_to_symbol_map(huffman_table):
    """Invert the huffman_table from symbol->code to code->symbol."""
    # huffman_table: {symbol: code_str}
    code_to_symbol = {code: sym for sym, code in huffman_table.items()}
    return code_to_symbol

def bytes_to_bitstring(byte_data):
    """Convert bytes to a bitstring."""
    return ''.join(bin(byte_val)[2:].zfill(8) for byte_val in byte_data)

def huffman_decode(bitstring, code_to_symbol, padding_bits):
    """Decode the bitstring using the Huffman code_to_symbol map."""
    # Remove the extra padding bits at the end
    if padding_bits > 0:
        bitstring = bitstring[:-padding_bits]

    # Since Huffman codes are prefix-free, we can decode symbol by accumulating bits
    decoded_symbols = []
    current_code = ''
    # To speed up lookups, note that since codes differ in length, we must check incrementally.
    # We'll accumulate bits in current_code and check if it forms a complete symbol.
    # Because codes are prefix-free, any match we find is a complete symbol.
    code_set = set(code_to_symbol.keys())
    # Maximum code length could help performance, but not strictly needed.
    # We'll just do a loop lookup.
    for bit in bitstring:
        current_code += bit
        if current_code in code_to_symbol:
            decoded_symbols.append(code_to_symbol[current_code])
            current_code = ''
    return decoded_symbols

def uniform_dequantize(qdata, step):
    return qdata.astype(np.float64) * step

def decompress():
    # Read compressed file and parse header
    block_size, pad, quant_step, nblocks, padding_bits, huffman_table, compressed_bitstream = read_compressed(
        'compressed')

    # Invert Huffman table to code->symbol
    code_to_symbol = build_code_to_symbol_map(huffman_table)

    # Convert compressed data to a bitstring
    bitstring = bytes_to_bitstring(compressed_bitstream)

    # Decode Huffman
    quantized_list = huffman_decode(bitstring, code_to_symbol, padding_bits)
    quantized_array = np.array(quantized_list, dtype=np.int16)
    # Reshape to (nblocks, block_size)
    quantized_blocks = quantized_array.reshape(nblocks, block_size)

    # Dequantize
    dct_blocks = uniform_dequantize(quantized_blocks, quant_step)

    # Inverse DCT (type=2, norm='ortho')
    # Matches the DCT settings from compress.py
    time_blocks = idct(dct_blocks, type=2, norm='ortho', axis=1)

    # Reassemble the full signal
    reconstructed = time_blocks.flatten()

    # Remove padding samples
    if pad > 0:
        reconstructed = reconstructed[:-pad]

    # Convert back to 16-bit PCM
    # Clip to avoid overflow
    reconstructed = np.clip(reconstructed, -1.0, 1.0)
    int_samples = (reconstructed * (2**15)).astype('<h')

    # Write to out.wav
    # Use the original parameters from step.wav: 1 channel, 16-bit, 44100 Hz
    # The instructions specify these parameters remain consistent.
    with wave.open('out.wav', 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(int_samples.tobytes())

    # print("Decompression complete. 'out.wav' created.")


