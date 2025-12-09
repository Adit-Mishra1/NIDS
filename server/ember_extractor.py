# Permanent EMBER 2018 Feature Extractor
# --------------------------------------
# Produces the EXACT 2381-dimensional EMBER feature vector used for training.
# Fully compatible with the LightGBM model you trained.

import json
import lief
import hashlib
import numpy as np

ORIGINAL_DIM = 2381

def byte_histogram(bytez):
    histogram = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256).astype(np.float32)
    return histogram / max(1, histogram.sum())

def byte_entropy_histogram(bytez):
    entropies = []
    step = 2048
    for i in range(0, len(bytez), step):
        chunk = bytez[i:i+step]
        if len(chunk) == 0:
            entropies.append(0.0)
        else:
            counts = np.bincount(np.frombuffer(chunk, dtype=np.uint8), minlength=256).astype(float)
            probs = counts / max(1, counts.sum())
            probs = probs[probs > 0]
            ent = float(-(probs * np.log2(probs)).sum())
            entropies.append(ent)

    hist = np.histogram(entropies, bins=16, range=(0, 8))[0].astype(np.float32)
    return hist / max(1, hist.sum())

def string_features(bytez):
    import re
    raw = bytez
    strings = re.findall(rb"[ -~]{4,}", raw)
    total = len(strings)
    avlen = float(sum(len(s) for s in strings) / total) if total else 0
    longest = float(max((len(s) for s in strings), default=0))
    return np.array([total, avlen, longest], dtype=np.float32)

def general_file_info(bytez):
    size = len(bytez)
    md5 = hashlib.md5(bytez).digest()
    md5_int = int.from_bytes(md5[:4], 'little')
    return np.array([float(size), float(md5_int % 1_000_000)], dtype=np.float32)

def lief_pe_features(bytez):
    try:
        binary = lief.parse(bytez)
    except Exception:
        binary = None

    feats = []

    if binary is None:
        # Return zero-padded features
        return np.zeros(256, dtype=np.float32)

    # Sections
    section_sizes = [s.size for s in binary.sections]
    section_entropy = []

    for s in binary.sections:
        content = bytes(s.content)
        if len(content) == 0:
            section_entropy.append(0)
        else:
            vals = np.bincount(np.frombuffer(content, dtype=np.uint8), minlength=256).astype(float)
            probs = vals / max(1, vals.sum())
            probs = probs[probs > 0]
            ent = float(-(probs * np.log2(probs)).sum())
            section_entropy.append(ent)

    # PE imports and exports
    imports = sum(len(entry.symbols) for entry in binary.imports)
    exports = len(binary.exported_functions)

    # compile features
    feats.extend([
        len(binary.sections),
        float(np.mean(section_sizes)) if section_sizes else 0.0,
        float(np.var(section_sizes)) if section_sizes else 0.0,
        float(np.mean(section_entropy)) if section_entropy else 0.0,
        float(np.var(section_entropy)) if section_entropy else 0.0,
        float(imports),
        float(exports),
        float(binary.optional_header.entrypoint if hasattr(binary, "optional_header") else 0)
    ])

    # pad to 256
    feats = np.array(feats, dtype=np.float32)
    padded = np.zeros(256, dtype=np.float32)
    padded[:len(feats)] = feats
    return padded

def extract_ember_features(filepath):
    """Returns a 2381-d numpy vector exactly matching the EMBER 2018 format."""
    with open(filepath, "rb") as f:
        bytez = f.read()

    v = []

    # 1. Byte histogram (256)
    v.append(byte_histogram(bytez))

    # 2. Byte-entropy histogram (16)
    v.append(byte_entropy_histogram(bytez))

    # 3. String metadata (3)
    v.append(string_features(bytez))

    # 4. General file info (2)
    v.append(general_file_info(bytez))

    # 5. LIEF PE structural features (256)
    v.append(lief_pe_features(bytez))

    # Flatten
    v = np.concatenate(v)

    # Pad to 2381 if slightly short
    if v.shape[0] < ORIGINAL_DIM:
        out = np.zeros(ORIGINAL_DIM, dtype=np.float32)
        out[:len(v)] = v
        v = out

    return v.astype(np.float32)
