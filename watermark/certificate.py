from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
import json
import hashlib
import torch
import datetime
import base64

def generate_keypair(save_dir="model/checkpoints"):
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()

    # Save private key
    with open(f"{save_dir}/private_key.pem", "wb") as f:
        f.write(private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()
        ))

    # Save public key
    with open(f"{save_dir}/public_key.pem", "wb") as f:
        f.write(public_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo
        ))

    print("Keypair generated and saved.")
    return private_key, public_key

def hash_model_weights(model):
    h = hashlib.sha256()
    for param in model.parameters():
        h.update(param.data.cpu().numpy().tobytes())
    return h.hexdigest()

def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def create_certificate(model, owner_id, trigger_path,
                       fingerprint_path, private_key_path,
                       save_path="model/checkpoints/certificate.json"):
    # Load private key
    with open(private_key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(), password=None, backend=default_backend()
        )

    # Build certificate payload
    cert = {
        "owner_id": owner_id,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model_weight_hash": hash_model_weights(model),
        "trigger_hash": hash_file(trigger_path),
        "fingerprint_hash": hash_file(fingerprint_path),
    }

    cert_bytes = json.dumps(cert, sort_keys=True).encode()

    # Sign
    signature = private_key.sign(
        cert_bytes,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    cert["signature"] = base64.b64encode(signature).decode()

    with open(save_path, "w") as f:
        json.dump(cert, f, indent=2)

    print(f"Certificate saved to {save_path}")
    return cert

def verify_certificate(cert_path, public_key_path):
    with open(cert_path) as f:
        cert = json.load(f)

    with open(public_key_path, "rb") as f:
        public_key = serialization.load_pem_public_key(
            f.read(), backend=default_backend()
        )

    signature = base64.b64decode(cert.pop("signature"))
    cert_bytes = json.dumps(cert, sort_keys=True).encode()

    try:
        public_key.verify(
            signature,
            cert_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        print("Certificate signature: VALID")
        return True
    except Exception as e:
        print(f"Certificate signature: INVALID ({e})")
        return False