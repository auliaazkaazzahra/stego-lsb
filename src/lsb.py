import numpy as np
import struct

def bytes_to_bits(data: bytes) -> list[int]:
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def bits_to_bytes(bits: list[int]) -> bytes:
    result = []
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i+8]:
            byte = (byte << 1) | bit
        result.append(byte)
    return bytes(result)

def get_capacity(image: np.ndarray) -> int:
    """Menghitung kapasitas maksimum embedding dalam bytes."""
    total_pixels_channels = image.flatten().shape[0]
    return (total_pixels_channels - 64) // 8

def embed_lsb(cover_image: np.ndarray, payload: bytes) -> np.ndarray:
    stego = cover_image.copy().flatten()
    capacity = len(stego)

    # Header: panjang payload (4 byte) + magic number (4 byte)
    header = struct.pack('>II', len(payload), 0xDEADBEEF)
    full_data = header + payload
    bits = bytes_to_bits(full_data)

    if len(bits) > capacity:
        raise ValueError(f"Payload terlalu besar! Kapasitas: {capacity // 8} bytes")

    for i, bit in enumerate(bits):
        stego[i] = (stego[i] & 0xFE) | bit

    return stego.reshape(cover_image.shape)

def extract_lsb(stego_image: np.ndarray) -> bytes:
    flat = stego_image.flatten()

    # Ekstrak header (64 bit pertama)
    header_bits = [flat[i] & 1 for i in range(64)]
    header = bits_to_bytes(header_bits)
    payload_len, magic = struct.unpack('>II', header)

    if magic != 0xDEADBEEF:
        raise ValueError("Magic number tidak valid! Citra bersih atau rusak.")

    total_bits = (payload_len + 8) * 8
    all_bits = [flat[i] & 1 for i in range(total_bits)]
    all_bytes = bits_to_bytes(all_bits)

    return all_bytes[8:]

from PIL import Image

def load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert('RGB'))

def save_image(image: np.ndarray, path: str):
    Image.fromarray(image.astype(np.uint8)).save(path, format='PNG')

def embed(cover_image: np.ndarray, payload: bytes) -> np.ndarray:
    return embed_lsb(cover_image, payload)

def extract(stego_image: np.ndarray) -> bytes:
    return extract_lsb(stego_image)

def calculate_metrics(cover: np.ndarray, stego: np.ndarray) -> dict:
    import math
    cover_f = cover.astype(np.float64)
    stego_f = stego.astype(np.float64)
    mse = np.mean((cover_f - stego_f) ** 2)
    psnr = float('inf') if mse == 0 else 10 * math.log10((255.0 ** 2) / mse)
    return {'mse': round(mse, 6), 'psnr': round(psnr, 4)}