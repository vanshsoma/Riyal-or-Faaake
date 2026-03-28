import torch
import numpy as np
import struct

def _str_to_bits(s):
    bits = []
    for c in s:
        b = format(ord(c), '08b')
        bits.extend([int(x) for x in b])
    return bits

def _bits_to_str(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        chars.append(chr(int(''.join(str(b) for b in byte), 2)))
    return ''.join(chars)

def _float_to_bits(f):
    packed = struct.pack('!f', f)
    return list(format(int.from_bytes(packed, 'big'), '032b'))

def _bits_to_float(bits):
    bit_str = ''.join(str(b) for b in bits)
    int_val = int(bit_str, 2)
    packed = int_val.to_bytes(4, 'big')
    return struct.unpack('!f', packed)[0]

def embed_signature(model, owner_id, layers=["layer3", "layer4"],
                    repeat=3):
    """
    Embeds owner_id into LSBs of BatchNorm gamma params.
    repeat=3 means each bit is written 3 times for majority-vote resilience.
    """
    message_bits = _str_to_bits(owner_id)
    repeated_bits = []
    for bit in message_bits:
        repeated_bits.extend([bit] * repeat)

    bit_idx = 0
    total_embedded = 0

    for layer_name in layers:
        layer = getattr(model, layer_name)
        for block in layer:
            for submodule in block.modules():
                if isinstance(submodule, torch.nn.BatchNorm2d):
                    gamma = submodule.weight.data.clone()
                    for i in range(len(gamma)):
                        if bit_idx >= len(repeated_bits):
                            break
                        val = gamma[i].item()
                        float_bits = _float_to_bits(val)
                        float_bits[-1] = repeated_bits[bit_idx]
                        gamma[i] = _bits_to_float(float_bits)
                        bit_idx += 1
                        total_embedded += 1
                    submodule.weight.data = gamma
                if bit_idx >= len(repeated_bits):
                    break

    print(f"Embedded {len(message_bits)} bits "
          f"({total_embedded} total with repeat={repeat})")
    return model

def decode_signature(model, owner_id_length,
                     layers=["layer3", "layer4"], repeat=3):
    """
    Decodes owner_id from BatchNorm gamma params using majority vote.
    owner_id_length: number of characters in the original owner_id.
    """
    total_bits_needed = owner_id_length * 8 * repeat
    extracted_bits = []

    for layer_name in layers:
        layer = getattr(model, layer_name)
        for block in layer:
            for submodule in block.modules():
                if isinstance(submodule, torch.nn.BatchNorm2d):
                    gamma = submodule.weight.data
                    for i in range(len(gamma)):
                        if len(extracted_bits) >= total_bits_needed:
                            break
                        val = gamma[i].item()
                        float_bits = _float_to_bits(val)
                        extracted_bits.append(int(float_bits[-1]))
                if len(extracted_bits) >= total_bits_needed:
                    break

    # Majority vote over repeated bits
    message_bits = []
    for i in range(0, len(extracted_bits), repeat):
        chunk = extracted_bits[i:i+repeat]
        message_bits.append(1 if sum(chunk) > len(chunk)//2 else 0)

    return _bits_to_str(message_bits)

def verify_signature(model, owner_id,
                     layers=["layer3", "layer4"], repeat=3):
    decoded = decode_signature(model, len(owner_id), layers, repeat)
    match = decoded == owner_id
    print(f"Expected:  {owner_id}")
    print(f"Decoded:   {decoded}")
    print(f"Signature: {'VERIFIED' if match else 'FAILED'}")
    return match, decoded