"""
norm_eu.py - Basque (Euskara) text normalization wrapper
=========================================================

Uses the external binary 'modulo1y2' for Basque text normalization.
Handles encoding conversion (UTF-8 ↔ ISO-8859-15).
"""

import subprocess
from pathlib import Path

# Path to the Basque normalizer resources
NORMALIZER_DIR = Path(__file__).parent.parent / "utils" / "normalizers" / "eu"
BINARY_PATH = NORMALIZER_DIR / "modulo1y2"
DICT_PATH = NORMALIZER_DIR / "eu_dicc"


def normalize_eu(text: str) -> str:
    """
    Normalize Basque text using the external modulo1y2 binary.
    
    This normalizer:
    - Expands abbreviations (EAE → Euskadiko Autonomia Elkartea)
    - Converts ordinals (1. → lehenengo, 2. → bigarren)
    - Handles special characters
    
    Args:
        text: Input text in Basque
        
    Returns:
        Normalized text suitable for ASR training
    """
    if not BINARY_PATH.exists():
        # Fallback to basic lowercase if binary not available
        return _basic_normalize(text)
    
    try:
        # Convert UTF-8 → ISO-8859-15
        encoded = subprocess.run(
            ['iconv', '-f', 'UTF-8', '-t', 'ISO-8859-15'],
            input=text.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True
        )
        
        # Run the normalizer
        normalized = subprocess.run(
            [
                str(BINARY_PATH),
                f'-Lang=eu',
                f'-HDicDB={DICT_PATH}',
                '-TxtMode=Word'
            ],
            input=encoded.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True
        )
        
        # Convert ISO-8859-15 → UTF-8
        decoded = subprocess.run(
            ['iconv', '-f', 'ISO-8859-15', '-t', 'UTF-8'],
            input=normalized.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True
        )
        
        result = decoded.stdout.decode('utf-8').strip()
        return result if result else _basic_normalize(text)
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # Fallback to basic normalization on error
        return _basic_normalize(text)


def _basic_normalize(text: str) -> str:
    """Basic fallback normalization: lowercase and clean whitespace."""
    import re
    # Remove forbidden characters
    forbidden = set(",;:?¿«»-¡!@*{}[]=/\\&#…")
    text = ''.join(ch if ch not in forbidden else ' ' for ch in text)
    # Lowercase and normalize whitespace
    return ' '.join(text.lower().split())


# Basque-specific character set for accepted chars
accepted_chars = "abcdefghijklmnopqrstuvwxyzñ''· -0123456789"


if __name__ == "__main__":
    # Test the normalizer
    test_sentences = [
        "EAEko UZIk 1. mailako eta 2. mailako LH ikasleen txostena.",
        "Eskerrik asko zure laguntzagatik.",
        "100 euro eta 50 zentimo.",
    ]
    
    print("Testing Basque normalizer:")
    for sent in test_sentences:
        print(f"  Input:  {sent}")
        print(f"  Output: {normalize_eu(sent)}")
        print()
