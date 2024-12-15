import py_compile
import os

# Compile 'decompress.py' and capture the path to the compiled bytecode file
try:
    compiled_path = py_compile.compile('decompress.py', cfile=None, dfile=None, doraise=True)
except py_compile.PyCompileError as e:
    print(f"Compilation failed: {e.msg}")
    exit(1)

# Ensure the compiled file exists
if not os.path.isfile(compiled_path):
    print(f"Compiled file not found at {compiled_path}")
    exit(1)

# Get the size of the compiled bytecode file in bytes
size_bytes = os.path.getsize(compiled_path)

# Optional: Convert size to a more readable format (e.g., KB, MB)
def format_size(bytes_size):
    for unit in ['bytes', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

formatted_size = format_size(size_bytes)

# Print out the size of the compiled bytecode file
print(f"Compiled 'decompress.pyc' size: {formatted_size}")
