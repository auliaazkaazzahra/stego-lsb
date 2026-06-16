import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

def generate_key(password: str) -> bytes:
    """Membangkitkan kunci AES-128 (16 byte) dari password menggunakan SHA-256."""
    return hashlib.sha256(password.encode()).digest()[:16]

def aes_encrypt(plaintext: bytes, key: bytes) -> tuple[bytes, bytes]:
    """Enkripsi plaintext dengan AES-128 CBC mode. Returns: (iv, ciphertext)"""
    iv = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
    return iv, ciphertext

def aes_decrypt(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    """Dekripsi ciphertext dengan AES-128 CBC mode."""
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)
    return plaintext

def encrypt(plaintext: bytes, key: bytes) -> tuple[bytes, bytes]:
    return aes_encrypt(plaintext, key)

def decrypt(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    return aes_decrypt(ciphertext, key, iv)