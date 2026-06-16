import numpy as np
import struct
from PIL import Image


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
    total_channels = image.flatten().shape[0]
    return (total_channels // 8) - 8


def embed_lsb(cover_image: np.ndarray, payload: bytes) -> np.ndarray:
    stego = cover_image.copy().flatten()

    # Header: panjang payload (4 byte) + magic number (4 byte)
    header = struct.pack('>II', len(payload), 0xDEADBEEF)
    full_data = header + payload
    bits = bytes_to_bits(full_data)

    if len(bits) > len(stego):
        raise ValueError(f"Payload terlalu besar! Kapasitas: {get_capacity(cover_image)} bytes")

    for i, bit in enumerate(bits):
        stego[i] = (stego[i] & 0xFE) | bit

    return stego.reshape(cover_image.shape)


def extract_lsb(stego_image: np.ndarray) -> bytes:
    flat = stego_image.flatten()

    # Ekstrak header (64 bit = 8 byte pertama)
    header_bits = [flat[i] & 1 for i in range(64)]
    header = bits_to_bytes(header_bits)
    payload_len, magic = struct.unpack('>II', header)

    if magic != 0xDEADBEEF:
        raise ValueError("Magic number tidak valid! Citra bersih atau rusak.")

    # Ekstrak bit payload mulai dari offset 64
    payload_bits = [flat[i] & 1 for i in range(64, 64 + payload_len * 8)]
    return bits_to_bytes(payload_bits)


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


def calculate_ber(original_payload: bytes, extracted_payload: bytes) -> float:
    min_len = min(len(original_payload), len(extracted_payload))
    if min_len == 0:
        return 1.0
    errors = sum(
        bin(a ^ b).count('1')
        for a, b in zip(original_payload[:min_len], extracted_payload[:min_len])
    )
    return round(errors / (min_len * 8), 6)