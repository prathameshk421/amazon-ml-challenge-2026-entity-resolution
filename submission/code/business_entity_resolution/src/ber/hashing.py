"""Stable 64-bit key hashes and packed pool ids (plan §5.5, §4)."""

import hashlib

import numpy as np


def key64(family, key):
    """Stable 64-bit key hash; the family code is part of the hashed string."""
    b = (family + "\x1f" + key).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


def keys64(family, keys):
    """key64 of every key, as a uint64 array (same values, same order)."""
    return np.fromiter((key64(family, k) for k in keys), dtype=np.uint64, count=len(keys))


def pack_ids(src, num):
    """(src << 32) | num as uint64 (scalars or arrays)."""
    return (np.asarray(src).astype(np.uint64) << np.uint64(32)) | np.asarray(num).astype(np.uint64)


def format_id(src, num):
    """Official record id string, e.g. S2-17."""
    return f"S{int(src)}-{int(num)}"
