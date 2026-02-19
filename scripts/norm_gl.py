"""
norm_gl.py - Galician text normalization wrapper
=================================================

Uses Cotovia TTS system for Galician text normalization.
Cotovia handles number expansion, abbreviations, and text preprocessing.
"""

import subprocess
from pathlib import Path

# Path to the Galician normalizer (Cotovia)
NORMALIZER_DIR = Path(__file__).parent.parent / "utils" / "normalizers" / "gl"
COTOVIA_PATH = NORMALIZER_DIR / "cotovia"


def normalize_gl(text: str) -> str:
    """
    Normalize Galician text using Cotovia TTS preprocessing.
    
    This normalizer:
    - Expands numbers (100 → cen, 50 → cincuenta)
    - Handles abbreviations
    - Normalizes punctuation
    
    Args:
        text: Input text in Galician
        
    Returns:
        Normalized text suitable for ASR training
    """
    if not COTOVIA_PATH.exists():
        # Fallback to basic normalize if Cotovia not available
        return _basic_normalize(text)
    
    try:
        # Data directory for Cotovia language files
        data_dir = NORMALIZER_DIR / "data"
        
        # Run Cotovia with preprocessing output (-p) to stdout (-S) in Galician (-l gl)
        result = subprocess.run(
            [
                str(COTOVIA_PATH),
                '-p1',              # Preprocessing step 1 (text normalization)
                '-n',               # Line-by-line mode (non-interactive)
                '-S',               # Output to stdout
                '-l', 'gl',         # Language: Galician
                '-D', str(data_dir) # Data directory path
            ],
            input=text.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True
        )
        
        # Cotovia outputs in ISO-8859-1, convert to UTF-8
        output = result.stdout.decode('iso-8859-1').strip()
        return output if output else _basic_normalize(text)
        
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


# Galician-specific character set for accepted chars
accepted_chars = "abcdefghijklmnopqrstuvwxyzñáéíóúàèìòùü''· -0123456789"


if __name__ == "__main__":
    # Test the normalizer
    test_sentences = [
        "Teño 100 euros e 50 céntimos.",
        "A lingua galega é moi importante.",
        "O 1º de xaneiro é Ano Novo.",
    ]
    
    print("Testing Galician normalizer:")
    for sent in test_sentences:
        print(f"  Input:  {sent}")
        print(f"  Output: {normalize_gl(sent)}")
        print()
