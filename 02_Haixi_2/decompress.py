import struct, wave# Modules for handling binary data and WAV files
import numpy as np# For numerical operations
from scipy.fftpack import idct# For applying the inverse discrete cosine transform
with open('compressed', 'rb') as f: d = f.read()# Read the compressed file
o = 0# Initialize the offset
bs, p = struct.unpack_from('<ii', d, o); o += 8# Extract block size and padding length (integers, 4 bytes each)
q = struct.unpack_from('<d', d, o)[0]; o += 8
n, pb, he = struct.unpack_from('<iii', d, o); o += 12# Extract number of blocks, padding bits, and number of Huffman entries (integers, 4 bytes each)
stc = {}# Initialize the symbol-to-code mapping dictionary for Huffman decoding
for _ in range(he):# Parse the Huffman table entries
    s = struct.unpack_from('<h', d, o)[0]; o += 2# Read the symbol (2 bytes, signed integer)
    cl = struct.unpack_from('<H', d, o)[0]; o += 2# Read the code length (2 bytes, unsigned short)
    nb = struct.unpack_from('<B', d, o)[0]; o += 1# Read the number of bytes in the code (1 byte, unsigned char)
    cb = d[o:o+nb]; o += nb# Read the actual code bytes
    stc[s] = ''.join(bin(c)[2:].zfill(8) for c in cb)[:cl]# Convert code bytes to a binary string of specified length
bitstring = ''.join(bin(c)[2:].zfill(8) for c in d[o:])# Convert the rest of the data to a binary string
bitstring = bitstring[:-pb] if pb > 0 else bitstring# Remove padding bits if any
cts = {v: k for k, v in stc.items()}# Create the code-to-symbol mapping for Huffman decoding
decoded, code = [], ''# Initialize the decoded list and temporary code buffer
for b in bitstring:
    code += b# Append bit to the current code
    if code in cts:# If the code is in the Huffman dictionary
        decoded.append(cts[code])# Append the corresponding symbol to the decoded list
        code = ''# Reset the code buffer
qb = np.array(decoded, dtype=np.int16).reshape(n, bs)# Reshape the decoded data into blocks of size (n, bs)
db = qb.astype(np.float64) * q# Multiply by the quantization step size to recover approximate values
tb = idct(db, type=2, norm='ortho', axis=1).flatten()  # Flatten the transformed blocks into a single array
if p > 0: tb = tb[:-p]# Remove padding samples if any
samples = (np.clip(tb, -1.0, 1.0) * (2**15)).astype('<h')# Clip the reconstructed signal to the range [-1.0, 1.0] and convert to 16-bit PCM
with wave.open('out.wav', 'wb') as f:# Write the reconstructed audio signal to an output WAV file
    f.setnchannels(1)# Set to mono (1 channel)
    f.setsampwidth(2)# Set sample width to 2 bytes (16-bit)
    f.setframerate(44100)# Set sampling rate to 44100 Hz
    f.writeframes(samples.tobytes())# Write the PCM samples to the file