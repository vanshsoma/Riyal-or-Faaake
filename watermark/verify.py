import json
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


def verify_certificate(
    cert_path: str = "certs/ownership_cert.json",
    public_key_path: str = "certs/public_key.pem"
) -> bool:
    with open(cert_path, "r") as f:
        cert_data = json.load(f)

    with open(public_key_path, "rb") as f:
        public_key = serialization.load_pem_public_key(f.read())

    signature = base64.b64decode(cert_data.pop("signature"))

    canonical = json.dumps(cert_data, sort_keys=True, separators=(",", ":"))

    try:
        public_key.verify(
            signature,
            canonical.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        print("Signature VALID — ownership verified.")
        return True
    except Exception as e:
        print(f"Signature INVALID: {e}")
        return False