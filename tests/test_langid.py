#!/usr/bin/env python3
"""
test_langid.py
==============

End-to-end pytest for verifying the language identification pipeline for
Basque (eu) and Galician (gl).

This test suite runs the full Found-Speech-Pipeline on sample data and verifies
that the final JSON output contains the correct language tags.

Usage:
    conda run -n fsp-bsc pytest tests/test_langid.py -v -s
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pytest

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants and Paths
TEST_FILE_PATH = Path(__file__).resolve()
REPO_ROOT = TEST_FILE_PATH.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
STEPS_DIR = REPO_ROOT / "steps"
INGESTION_DIR = REPO_ROOT / "ingestion"
INPUTS_DIR = REPO_ROOT / "inputs"
ROVER_DIR = REPO_ROOT / "rover_out"  # Defined implicitly in original script, verifying location

# Ensure necessary directories exist
INPUTS_DIR.mkdir(exist_ok=True)

# Test Data
# (video_id, expected_language_code)
TEST_CASES = [
#    ("dtTCbYIHLps", "eu"),
    ("JynlbvTgWzM", "gl"),
]


def run_command(
    label: str,
    cmd: List[str | Path],
    cwd: Path = REPO_ROOT,
    env: Optional[dict] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """
    Execute a shell command with logging and error handling.
    
    Args:
        label: A human-readable label for the step.
        cmd: List of command arguments.
        cwd: Current working directory for the command.
        env: Environment variables dict.
        check: Whether to raise an exception on non-zero exit code.
    """
    cmd_str = " ".join(str(c) for c in cmd)
    logger.info(f"► {label}")
    logger.info(f"  $ {cmd_str}")

    # Ensure we use the current python executable if 'python' or 'python3' is invoked
    # This helps when running inside a conda env via `conda run`.
    # However, since we are calling scripts directly, we can just pass sys.executable
    # as the first argument if the command starts with 'python'.
    
    # Copy current env if not provided
    if env is None:
        env = os.environ.copy()
    
    # Ensure PYTHONPATH includes repo root
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}:{pythonpath}"

    try:
        result = subprocess.run(
            [str(c) for c in cmd],
            cwd=cwd,
            env=env,
            check=check,
            capture_output=False,  # Let stdout/stderr flow through to pytest -s
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"✖  {label} failed with exit code {e.returncode}")
        pytest.fail(f"Step '{label}' failed: {e}")
        raise e


@pytest.mark.parametrize("video_id, expected_lang", TEST_CASES)
def test_pipeline_langid(video_id: str, expected_lang: str):
    """
    Run the full pipeline for a given video ID and verify the detected language.
    """
    logger.info(f"Starting pipeline test for video: {video_id} (Expected: {expected_lang})")

    # Define paths
    raw_tsv = INGESTION_DIR / f"{video_id}.tsv"
    out_json_name = f"final_output_{video_id}.json"
    out_json_path = INPUTS_DIR / "wordlevel_alignment" / out_json_name
    
    # Ensure source TSV exists
    if not raw_tsv.exists():
        pytest.skip(f"Source TSV not found: {raw_tsv}. Skipping test.")

    # 1. Normalize TSV
    # Usage: python scripts/normalize_tsv.py <tsv> <lang> <separator>
    run_command(
        "Normalise TSV",
        [sys.executable, SCRIPTS_DIR / "normalize_tsv.py", raw_tsv, expected_lang, ". "]
    )

    # 2. Ingest Single
    # Usage: python scripts/ingest_single.py --session-id=<vid>
    run_command(
        "Ingest single",
        [sys.executable, SCRIPTS_DIR / "ingest_single.py", f"--session-id={video_id}"]
    )

    # 3. Generate Final Data (The core LangID step)
    # Usage: python steps/generate_final_data.py --session=<vid> --output <json>
    env = os.environ.copy()
    run_command(
        "Generate final data",
        [
            sys.executable,
            STEPS_DIR / "generate_final_data.py",
            f"--session={video_id}",
            "--output", out_json_name,
            "--device", "auto" # allow it to pick cuda if available
        ],
        env=env
    )

    # 4. Duration Filter
    # Usage: python scripts/duration_filter.py <json_path>
    run_command(
        "Duration filter",
        [sys.executable, SCRIPTS_DIR / "duration_filter.py", out_json_path]
    )

    # 5. ROVER Merge
    # Usage: python scripts/rover_merge.py <json_path> --csv --plot --out-dir <dir>
    # Note: ROVER_DIR needs to be defined.
    rover_out = REPO_ROOT / "rover_out"
    rover_out.mkdir(exist_ok=True)
    
    run_command(
        "ROVER merge",
        [
            sys.executable,
            SCRIPTS_DIR / "rover_merge.py",
            out_json_path,
            "--csv",
            "--plot",
            "--out-dir", rover_out
        ]
    )

    # 6. Punctuation & Capitalization
    # Usage: python scripts/punctuate.py <json_path> --device cpu
    run_command(
        "Punctuation & Capitalization",
        [
            sys.executable,
            SCRIPTS_DIR / "punctuate.py",
            out_json_path,
            "--device", "cpu"
        ]
    )

    # --- Verification ---
    logger.info(f"Verifying output file: {out_json_path}")
    assert out_json_path.exists(), f"Output JSON file {out_json_path} was not created."

    with open(out_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # The JSON structure is expected to be:
    # { "segment_id": { "video_id": "...", "results": [ { "language": "...", ... }, ... ] }, ... }
    
    found_languages = []
    
    for segment_id, content in data.items():
        results = content.get("results", [])
        for res in results:
            lang = res.get("language")
            if lang:
                found_languages.append(lang)

    # Basic check: verify that the detected language appears in the results
    from collections import Counter
    lang_counts = Counter(found_languages)
    logger.info(f"Language distribution for {video_id}: {lang_counts}")

    if not found_languages:
        pytest.fail(f"No language tags found in {out_json_path}")

    most_common_lang, count = lang_counts.most_common(1)[0]
    
    assert most_common_lang == expected_lang, (
        f"Expected dominant language {expected_lang}, but got {most_common_lang} "
        f"(Counts: {lang_counts})"
    )

    logger.info(f"✔ Verification Passed: Dominant language is {most_common_lang}")


# ---------------------------------------------------------------------------
# Unit tests for `choose_language` from steps/generate_final_data.py
# ---------------------------------------------------------------------------

# Add steps/ to sys.path so we can import generate_final_data
sys.path.insert(0, str(STEPS_DIR))
from generate_final_data import choose_language  # noqa: E402

@pytest.fixture(scope="session")
def lid_model():
    """Load the FastText LID model once for all choose_language tests."""
    import fasttext
    model_path = str(REPO_ROOT / "utils" / "models" / "lid.176.bin")
    return fasttext.load_model(model_path)


# Sample texts from actual pipeline outputs and representative content
CHOOSE_LANG_CASES = [
    # --- Catalan (ca) ---
    pytest.param(
        "l'emissió es farà des d'un plató virtual que permetrà als conductors tenir més mobilitat",
        "ca",
        id="ca-tv3-newscast",
    ),
    pytest.param(
        "l'objectiu no és altre que oferir una informació més clara més entenedora i més gràfica",
        "ca",
        id="ca-catalan-tokens",
    ),
    pytest.param(
        "els telenotícies migdia continuaran amb l'estil directe d'informar de l'actualitat de catalunya i el món",
        "ca",
        id="ca-catalunya",
    ),
    # --- Spanish (es) ---
    pytest.param(
        "el frente frío número dieciocho provocará posible caída de nieve o aguanieve en sierras de sonora sinaloa durango y chihuahua",
        "es",
        id="es-mexico-news",
    ),
    pytest.param(
        "ucrania acusó a rusia de trasladar a niños ucranianos presuntamente secuestrados a campos de reeducación en corea del norte",
        "es",
        id="es-international",
    ),
    pytest.param(
        "arqueólogos confirmaron que se descubrieron huellas de dinosaurios entre puebla y oaxaca con una antigüedad de ciento veinte millones de años",
        "es",
        id="es-archaeology",
    ),
    # --- Basque (eu) ---
    pytest.param(
        "gaur egun euskal herrian hizkuntza politika garrantzitsua da eta euskararen erabilera sustatzeko neurri berriak hartu dira",
        "eu",
        id="eu-language-policy",
    ),
    pytest.param(
        "bilboko udalak erabaki du parke berri bat eraikitzea hiriko erdigunean auzo guztietako biztanleentzat",
        "eu",
        id="eu-bilbao-news",
    ),
    # --- Galician (gl) ---
    pytest.param(
        "para a academia galicia é a denominación oficial do país e a forma maioritaria na expresión oral e na escrita moderna",
        "gl",
        id="gl-academia",
    ),
    pytest.param(
        "os organismos oficiais adoptan a forma galicia xunta de galicia televisión de galicia augas de galicia",
        "gl",
        id="gl-organismos",
    ),
]


@pytest.mark.parametrize("text, expected_lang", CHOOSE_LANG_CASES)
def test_choose_language(lid_model, text: str, expected_lang: str):
    """
    Verify that choose_language correctly identifies the language of sample texts.
    """
    lang, confidence = choose_language(text, lid_model)

    logger.info(
        f"Text: {text[:60]}... → detected={lang} (conf={confidence:.3f}), expected={expected_lang}"
    )

    assert lang == expected_lang, (
        f"Expected '{expected_lang}' but got '{lang}' (confidence={confidence:.3f}) "
        f"for text: {text[:80]}..."
    )
    assert confidence > 0, f"Confidence should be positive, got {confidence}"


def test_choose_language_catalan_heuristic(lid_model):
    """
    Verify that Catalan-specific token heuristics kick in for ambiguous text
    containing distinctive Catalan markers (l', d', ç, ny, això, qüestió).
    """
    # Text with strong Catalan markers that might be misclassified by FastText
    catalan_text = "això és una qüestió important que s'ha d'analitzar amb cura"
    lang, confidence = choose_language(catalan_text, lid_model)

    logger.info(f"Catalan heuristic test: detected={lang} (conf={confidence:.3f})")
    assert lang == "ca", f"Expected 'ca' for Catalan-heuristic text, got '{lang}'"


def test_choose_language_returns_tuple(lid_model):
    """
    Verify that choose_language always returns a (str, float) tuple.
    """
    lang, conf = choose_language("hello world", lid_model)
    assert isinstance(lang, str), f"Language should be str, got {type(lang)}"
    assert isinstance(conf, float), f"Confidence should be float, got {type(conf)}"


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))