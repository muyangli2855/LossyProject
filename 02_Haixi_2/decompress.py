# decompress.py
#!/usr/bin/env python

import struct
import wave
import numpy as np
from scipy.fftpack import idct

def read_compressed(filename):
    """Read the compressed file and parse the header."""

    with open(filename, 'rb') as f:
        data = f.read()

    offset = 0
    # block_size (int32)
    block_size = struct.unpack_from('<i', data, offset)[0]
    offset += 4
    # pad (int32)
    pad = struct.unpack_from('<i', data, offset)[0]
    offset += 4
    # quant_step (float64)
    quant_step = struct.unpack_from('<d', data, offset)[0]
    offset += 8
    # nblocks (int32)
    nblocks = struct.unpack_from('<i', data, offset)[0]
    offset += 4
    # padding_bits (int32)
    padding_bits = struct.unpack_from('<i', data, offset)[0]
    offset += 4
    # huff_entries (int32)
    huff_entries = struct.unpack_from('<i', data, offset)[0]
    offset += 4

    # Read Huffman table entries
    # For each entry:
    # symbol (int16)
    # code_length (uint16)
    # num_code_bytes (uint8)
    # code_bytes (num_code_bytes)
    symbol_to_code = {}
    for _ in range(huff_entries):
        sym = struct.unpack_from('<h', data, offset)[0]
        offset += 2
        code_length = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        num_code_bytes = struct.unpack_from('<B', data, offset)[0]
        offset += 1

        code_bytes = data[offset:offset+num_code_bytes]
        offset += num_code_bytes

        # Convert code_bytes to a bitstring
        bitstring = ''.join(bin(byte_val)[2:].zfill(8) for byte_val in code_bytes)
        # Truncate to code_length bits
        code_str = bitstring[:code_length]
        symbol_to_code[sym] = code_str

    # The rest is compressed bitstream
    compressed_bitstream = data[offset:]

    return block_size, pad, quant_step, nblocks, padding_bits, symbol_to_code, compressed_bitstream

def build_code_to_symbol_map(symbol_to_code):
    """Invert the symbol_to_code dictionary to code_to_symbol for decoding."""
    code_to_symbol = {c: s for s, c in symbol_to_code.items()}
    return code_to_symbol

def bytes_to_bitstring(byte_data):
    """Convert bytes to a bitstring."""
    return ''.join(bin(byte_val)[2:].zfill(8) for byte_val in byte_data)

def huffman_decode(bitstring, code_to_symbol, padding_bits):
    """Decode the bitstring using the Huffman code_to_symbol map."""
    # Remove padding bits from the end
    if padding_bits > 0:
        bitstring = bitstring[:-padding_bits]

    decoded_symbols = []
    current_code = ''
    # Because we have a set of prefix-free codes, we can decode incrementally.
    # To avoid excessive lookup complexity, store code_to_symbol keys in a set for quick checking.
    code_prefixes = code_to_symbol.keys()

    for bit in bitstring:
        current_code += bit
        if current_code in code_to_symbol:
            decoded_symbols.append(code_to_symbol[current_code])
            current_code = ''
    return decoded_symbols

def uniform_dequantize(qdata, step):
    return qdata.astype(np.float64) * step

if  __name__ == '__main__':
    # Read compressed file and parse header
    (block_size, pad, quant_step, nblocks, padding_bits, symbol_to_code, 
     compressed_bitstream) = read_compressed('compressed')

    # Invert Huffman table
    code_to_symbol = build_code_to_symbol_map(symbol_to_code)

    # Convert compressed data to a bitstring
    bitstring = bytes_to_bitstring(compressed_bitstream)

    # Huffman decode
    quantized_list = huffman_decode(bitstring, code_to_symbol, padding_bits)
    quantized_array = np.array(quantized_list, dtype=np.int16)
    # Reshape to (nblocks, block_size)
    quantized_blocks = quantized_array.reshape(nblocks, block_size)

    # Dequantize
    dct_blocks = uniform_dequantize(quantized_blocks, quant_step)

    # Inverse DCT
    time_blocks = idct(dct_blocks, type=2, norm='ortho', axis=1)

    # Reassemble the full signal
    reconstructed = time_blocks.flatten()

    # Remove padding samples
    if pad > 0:
        reconstructed = reconstructed[:-pad]

    # Convert back to 16-bit PCM
    reconstructed = np.clip(reconstructed, -1.0, 1.0)
    int_samples = (reconstructed * (2**15)).astype('<h')

    # Write out.wav
    # Using original parameters: 1 channel, 16-bit, 44100 Hz
    with wave.open('out.wav', 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(int_samples.tobytes())

    print("Decompression complete. 'out.wav' created.")
