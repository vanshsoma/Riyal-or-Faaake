import torch
import numpy as np
import struct
import yaml

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

# ── Helpers ────────────────────────────────────────────────────────────────

def _str_to_bits(s):
    bits = []
    for c in s:
        for b in format(ord(c), '08b'):
            bits.append(int(b))
    return bits

def _bits_to_str(bits):
    chars = []
    for i in range(0, len(bits) - 7, 8):
        chars.append(chr(int(''.join(str(b) for b in bits[i:i+8]), 2)))
    return ''.join(chars)

def _float_to_bits(f):
    packed = struct.pack('!f', f)
    return list(format(int.from_bytes(packed, 'big'), '032b'))

def _bits_to_float(bits):
    bit_str = ''.join(str(b) for b in bits)
    packed = int(bit_str, 2).to_bytes(4, 'big')
    return struct.unpack('!f', packed)[0]

# ── LSB Signature ──────────────────────────────────────────────────────────

def embed_lsb(model, owner_id, layers=["layer3", "layer4"], repeat=3):
    message_bits  = _str_to_bits(owner_id)
    repeated_bits = []
    for bit in message_bits:
        repeated_bits.extend([bit] * repeat)

    bit_idx = 0
    for layer_name in layers:
        layer = getattr(model, layer_name)
        for block in layer:
            for submodule in block.modules():
                if isinstance(submodule, torch.nn.BatchNorm2d):
                    gamma = submodule.weight.data.clone()
                    for i in range(len(gamma)):
                        if bit_idx >= len(repeated_bits):
                            break
                        fb = _float_to_bits(gamma[i].item())
                        fb[-1] = repeated_bits[bit_idx]
                        gamma[i] = _bits_to_float(fb)
                        bit_idx += 1
                    submodule.weight.data = gamma
                if bit_idx >= len(repeated_bits):
                    break

    print(f"[LSB] Embedded {len(message_bits)} bits "
          f"({bit_idx} total with repeat={repeat})")
    return model

def decode_lsb(model, owner_id_length,
               layers=["layer3", "layer4"], repeat=3):
    total_needed = owner_id_length * 8 * repeat
    extracted    = []

    for layer_name in layers:
        layer = getattr(model, layer_name)
        for block in layer:
            for submodule in block.modules():
                if isinstance(submodule, torch.nn.BatchNorm2d):
                    gamma = submodule.weight.data
                    for i in range(len(gamma)):
                        if len(extracted) >= total_needed:
                            break
                        fb = _float_to_bits(gamma[i].item())
                        extracted.append(int(fb[-1]))
                if len(extracted) >= total_needed:
                    break

    # Majority vote
    message_bits = []
    for i in range(0, len(extracted), repeat):
        chunk = extracted[i:i+repeat]
        message_bits.append(1 if sum(chunk) > len(chunk)//2 else 0)

    return _bits_to_str(message_bits)

def verify_lsb(model, owner_id,
               layers=["layer3", "layer4"], repeat=3):
    decoded = decode_lsb(model, len(owner_id), layers, repeat)
    match   = decoded == owner_id
    print(f"[LSB] Expected : {owner_id}")
    print(f"[LSB] Decoded  : {decoded}")
    print(f"[LSB] Result   : {'VERIFIED ✓' if match else 'FAILED ✗'}")
    return match, decoded

# ── Spread-Spectrum Signature ──────────────────────────────────────────────

def embed_spread_spectrum(model, secret_seed,
                          layers=["layer3", "layer4"], alpha=0.001):
    rng = np.random.RandomState(secret_seed)
    total_params = 0

    for layer_name in layers:
        layer = getattr(model, layer_name)
        for block in layer:
            for submodule in block.modules():
                if isinstance(submodule, torch.nn.Conv2d):
                    w = submodule.weight.data
                    v = rng.randn(*w.shape).astype(np.float32)
                    # Normalise so alpha is meaningful
                    v = v / (np.linalg.norm(v) + 1e-8)
                    submodule.weight.data = w + alpha * torch.tensor(
                        v, device=w.device
                    )
                    total_params += w.numel()

    print(f"[SS] Spread-spectrum embedded across "
          f"{total_params:,} conv weight params (alpha={alpha})")
    return model

def verify_spread_spectrum(model, secret_seed,
                           layers=["layer3", "layer4"],
                           alpha=0.001, threshold=0.3):
    rng        = np.random.RandomState(secret_seed)
    dot_scores = []

    for layer_name in layers:
        layer = getattr(model, layer_name)
        for block in layer:
            for submodule in block.modules():
                if isinstance(submodule, torch.nn.Conv2d):
                    w = submodule.weight.data.cpu().numpy().flatten()
                    v = rng.randn(len(w)).astype(np.float32)
                    v = v / (np.linalg.norm(v) + 1e-8)
                    # Normalised dot product
                    score = float(np.dot(w / (np.linalg.norm(w) + 1e-8), v))
                    dot_scores.append(score)

    mean_score = np.mean(dot_scores)
    match      = mean_score > threshold
    print(f"[SS] Correlation score : {mean_score:.6f}")
    print(f"[SS] Threshold         : {threshold}")
    print(f"[SS] Result            : {'VERIFIED ✓' if match else 'FAILED ✗'}")
    return match, mean_score

# ── Combined embed/verify ──────────────────────────────────────────────────

def embed_signature(model, owner_id, secret_seed=42,
                    layers=["layer3", "layer4"],
                    repeat=3, alpha=0.001):
    model = embed_lsb(model, owner_id, layers, repeat)
    model = embed_spread_spectrum(model, secret_seed, layers, alpha)
    return model

def verify_signature(model, owner_id, secret_seed=42,
                     layers=["layer3", "layer4"],
                     repeat=3, threshold=0.3):
    lsb_match, decoded = verify_lsb(model, owner_id, layers, repeat)
    ss_match, score    = verify_spread_spectrum(
        model, secret_seed, layers, threshold=threshold
    )
    overall = lsb_match or ss_match
    print(f"\n[WHITEBOX] Overall: {'OWNERSHIP VERIFIED ✓' if overall else 'NOT VERIFIED ✗'}")
    return overall, decoded, score