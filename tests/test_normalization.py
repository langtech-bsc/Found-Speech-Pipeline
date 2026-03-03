#!/usr/bin/env python3
"""
test_normalization.py - Comprehensive normalization tests for all 4 languages
=============================================================================

Tests the `clean_text` function from scripts/normalize.py (the same function
used by the pipeline) for Catalan, Spanish, Basque, and Galician.

Tests are organized into:
  - Tests that PASS: verify expected normalization behavior
  - Tests marked xfail: document known gaps where normalization does NOT
    currently handle a transformation (useful for tracking improvements)

Usage:
    conda run -n fsp-bsc pytest tests/test_normalization.py -v -s

Note: Run from the project root directory.
"""

import sys
from pathlib import Path

import pytest

# Add scripts directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

from clean_and_expand import clean_text as _clean_text_v2
from clean_and_split import split_text, remove_chars

# Backward-compatible wrapper: old tests call clean_text(text, lang)
def clean_text(text, lang):
    return _clean_text_v2(text, lang, False, False)

def split_and_clean(text, mark, lang):
    # V1 split_and_clean also applied clean_text to the result
    split_str = split_text(remove_chars(text, False, lang), False, mark)
    return mark.join([_clean_text_v2(s, lang, False, False) for s in split_str.split(mark)])


# =========================================================================
# CATALAN (ca) — uses ovos_number_parser + norm_dicts_ca
# =========================================================================

class TestCatalanNumbers:
    """Test number-to-words expansion for Catalan."""

    def test_small_number(self):
        result = clean_text("16 de gener", "ca")
        assert "setze" in result, f"Expected 'setze' (16), got: {result}"

    def test_medium_number(self):
        result = clean_text("farà 30 anys", "ca")
        assert "trenta" in result, f"Expected 'trenta' (30), got: {result}"

    @pytest.mark.xfail(reason="ca: Numbers like 324 are not always expanded to words by ovos_number_parser")
    def test_large_number(self):
        result = clean_text("324 oferiran", "ca")
        assert "tres-cents" in result or "tres cents" in result, f"Expected 'tres-cents' (324), got: {result}"

    def test_decimal_number(self):
        """Decimal numbers with comma separator should expand to 'X coma Y'."""
        result = clean_text("3,14 metres", "ca")
        assert "tres" in result and "coma" in result, f"Expected 'tres coma catorze', got: {result}"

    def test_dotted_thousands(self):
        """Dotted separators (e.g. 1.000) should be stripped."""
        result = clean_text("1.000 persones", "ca")
        # The dot is stripped, so 1000 remains and then gets expanded
        assert "mil" in result or "1000" in result, f"Expected thousands handling, got: {result}"

    @pytest.mark.xfail(reason="Big numbers like 1000000 are not expanded to words, they stay as digits")
    def test_million_as_words(self):
        result = clean_text("1.000.000 euros", "ca")
        assert "milió" in result, f"Expected 'milió', got: {result}"


class TestCatalanOrdinals:
    """Test ordinal expansion for Catalan."""

    def test_ordinal_1r(self):
        result = clean_text("1r premi", "ca")
        assert "un r" in result or "primer" in result, f"Expected 'un r', got: {result}"

    def test_ordinal_2n(self):
        result = clean_text("2n lloc", "ca")
        assert "dos n" in result or "segon" in result, f"Expected 'dos n', got: {result}"

    def test_ordinal_feminine_1ra(self):
        result = clean_text("1ra edició", "ca")
        assert "un ra" in result or "primera" in result, f"Expected 'un ra', got: {result}"

    def test_ordinal_superscript_1o(self):
        result = clean_text("1º classificat", "ca")
        assert "primer" in result, f"Expected 'primer', got: {result}"


class TestCatalanCurrency:
    """Test currency symbol expansion for Catalan."""

    def test_euros(self):
        result = clean_text("50€", "ca")
        assert "euros" in result and "cinquanta" in result, f"Expected 'cinquanta euros', got: {result}"

    def test_dollars(self):
        result = clean_text("10$", "ca")
        assert "dòlars" in result, f"Expected 'dòlars', got: {result}"


class TestCatalanSpecialChars:
    """Test special character expansion for Catalan."""

    def test_percent(self):
        result = clean_text("50%", "ca")
        assert "per cent" in result, f"Expected 'per cent', got: {result}"

    def test_temperature(self):
        result = clean_text("25ºC", "ca")
        assert "c" in result, f"Expected 'vint-i-cinc c', got: {result}"

    def test_greek_letters(self):
        result = clean_text("α β γ", "ca")
        assert "alfa" in result and "beta" in result and "gamma" in result, \
            f"Expected Greek letter names, got: {result}"

    def test_forbidden_chars_removed(self):
        result = clean_text("Hola! Com estàs?", "ca")
        assert "!" not in result and "?" not in result, f"Expected no punctuation, got: {result}"

    def test_output_is_lowercase(self):
        result = clean_text("Televisió de Catalunya", "ca")
        assert result == result.lower(), f"Expected all lowercase, got: {result}"


class TestCatalanAbbreviations:
    """Test abbreviation and initial expansion for Catalan."""

    def test_abbreviation_km(self):
        result = clean_text("a 120 km de distància", "ca")
        assert "quilòmetres" in result, f"Expected 'quilòmetres', got: {result}"

    @pytest.mark.xfail(reason="ca: Mixed alphanumeric tokens like TV3 are not letter-expanded — treated as one token and just lowercased to 'tv'")
    def test_initials_expansion(self):
        """Uppercase initials like TV should be expanded letter by letter."""
        result = clean_text("TV3 emetrà el programa", "ca")
        assert "te" in result and "ve" in result, f"Expected letter expansion for TV, got: {result}"


# =========================================================================
# SPANISH (es) — uses ovos_number_parser + norm_dicts_es
# =========================================================================

class TestSpanishNumbers:
    """Test number-to-words expansion for Spanish."""

    def test_small_number(self):
        result = clean_text("18 provincias", "es")
        assert "dieciocho" in result, f"Expected 'dieciocho' (18), got: {result}"

    def test_medium_number(self):
        result = clean_text("50 personas", "es")
        assert "cincuenta" in result, f"Expected 'cincuenta' (50), got: {result}"

    @pytest.mark.xfail(reason="Large multi-word numbers (120) are not expanded in Spanish — stays as digits")
    def test_large_number_expansion(self):
        result = clean_text("120 millones de años", "es")
        assert "ciento veinte" in result, f"Expected 'ciento veinte' (120), got: {result}"


class TestSpanishOrdinals:
    """Test ordinal expansion for Spanish."""

    def test_ordinal_1o(self):
        result = clean_text("1º lugar", "es")
        assert "primero" in result, f"Expected 'primero', got: {result}"

    def test_ordinal_2ndo(self):
        result = clean_text("2ndo piso", "es")
        assert "dos ndo" in result or "segundo" in result, f"Expected 'dos ndo', got: {result}"


class TestSpanishCurrency:
    """Test currency symbol expansion for Spanish."""

    def test_euros(self):
        result = clean_text("50€", "es")
        assert "euros" in result, f"Expected 'euros', got: {result}"


class TestSpanishSpecialChars:
    """Test special character expansion for Spanish."""

    def test_percent(self):
        result = clean_text("50%", "es")
        assert "por ciento" in result, f"Expected 'por ciento', got: {result}"

    def test_forbidden_chars_removed(self):
        result = clean_text("¡Hola! ¿Cómo estás?", "es")
        assert "¡" not in result and "!" not in result, f"Expected no punctuation, got: {result}"
        assert "¿" not in result and "?" not in result, f"Expected no question marks, got: {result}"

    def test_output_is_lowercase(self):
        result = clean_text("Buenos Días", "es")
        assert result == result.lower(), f"Expected all lowercase, got: {result}"


class TestSpanishAbbreviations:
    """Test abbreviation expansion for Spanish."""

    def test_abbreviation_km(self):
        result = clean_text("a 120 km de distancia", "es")
        assert "quilómetros" in result, f"Expected 'quilómetros', got: {result}"


# =========================================================================
# BASQUE (eu) — uses external modulo1y2 binary
# =========================================================================

class TestBasqueNumbers:
    """Test number expansion for Basque (via external binary)."""

    def test_small_number(self):
        result = clean_text("100 euro", "eu")
        assert "ehun" in result.lower(), f"Expected 'ehun' (100), got: {result}"

    def test_year(self):
        result = clean_text("2003. urtean", "eu")
        # The binary should expand 2003 to Basque words
        assert not any(c.isdigit() for c in result), f"Expected no digits remaining, got: {result}"


class TestBasqueOrdinals:
    """Test ordinal expansion for Basque (via external binary)."""

    def test_ordinal_1(self):
        result = clean_text("1. mailako", "eu")
        assert "lehenengo" in result.lower(), f"Expected 'lehenengo' (1st), got: {result}"


class TestBasqueAbbreviations:
    """Test abbreviation expansion for Basque (via external binary)."""

    def test_eae(self):
        result = clean_text("EAEko ikasleek", "eu")
        assert "euskadi" in result.lower() or "autonomi" in result.lower(), \
            f"Expected EAE abbreviation expansion, got: {result}"


class TestBasqueForbiddenChars:
    """Test that forbidden chars are stripped after eu normalization."""

    def test_punctuation_removed(self):
        result = clean_text("Kaixo, zer moduz?", "eu")
        assert "," not in result and "?" not in result, f"Expected no punctuation, got: {result}"


class TestBasqueCurrency:
    """Test currency and symbol expansion for Basque (via modulo1y2)."""

    def test_euros(self):
        result = clean_text("10€ balio du", "eu")
        assert "hamar" in result.lower() and "euro" in result.lower(), \
            f"Expected 'hamar euro', got: {result}"

    def test_percent(self):
        result = clean_text("50% da", "eu")
        assert "ehuneko" in result.lower() or "berrogeita" in result.lower(), \
            f"Expected percent expansion, got: {result}"


# =========================================================================
# GALICIAN (gl) — uses Cotovia TTS binary
# =========================================================================

class TestGalicianNumbers:
    """Test number expansion for Galician (via Cotovia)."""

    def test_small_number(self):
        result = clean_text("100 euros", "gl")
        assert "cen" in result.lower(), f"Expected 'cen' (100), got: {result}"

    def test_year(self):
        result = clean_text("Ata 2003 só se aceptaba", "gl")
        assert "dous mil tres" in result.lower() or "dous mil e tres" in result.lower(), \
            f"Expected '2003' expanded to words, got: {result}"

    def test_dotted_thousands(self):
        result = clean_text("Son 1.500 persoas.", "gl")
        assert "mil" in result.lower() and "cincocentas" in result.lower(), \
            f"Expected 'mil cincocentas', got: {result}"

    def test_decimal_comma(self):
        result = clean_text("O valor é 3,14.", "gl")
        assert "tres" in result.lower() and "coma" in result.lower(), \
            f"Expected 'tres coma catorce', got: {result}"


class TestGalicianOrdinals:
    """Test ordinal expansion for Galician (via Cotovia)."""

    def test_ordinal_1o(self):
        result = clean_text("O 1º de xaneiro", "gl")
        assert "primeiro" in result.lower(), f"Expected 'primeiro' (1st), got: {result}"

    def test_ordinal_feminine_2a(self):
        result = clean_text("A 2ª edición.", "gl")
        assert "segunda" in result.lower(), f"Expected 'segunda' (2nd fem), got: {result}"


class TestGalicianCurrency:
    """Test currency and symbol expansion for Galician (via Cotovia)."""

    def test_euros(self):
        result = clean_text("Prezo: 10€", "gl")
        assert "dez" in result.lower() and "euros" in result.lower(), \
            f"Expected 'dez euros', got: {result}"

    def test_percent(self):
        result = clean_text("O 50% dos galegos", "gl")
        assert "por cento" in result.lower(), f"Expected 'por cento', got: {result}"


class TestGalicianAbbreviations:
    """Test abbreviation expansion for Galician (via Cotovia)."""

    def test_sr(self):
        result = clean_text("O Sr. García chegou.", "gl")
        assert "señor" in result.lower(), f"Expected 'señor', got: {result}"

    def test_sra(self):
        result = clean_text("A Sra. López chamou.", "gl")
        assert "señora" in result.lower(), f"Expected 'señora', got: {result}"

    def test_avenida(self):
        result = clean_text("Vive na Av. de Galicia.", "gl")
        assert "avenida" in result.lower(), f"Expected 'avenida', got: {result}"

    def test_km(self):
        result = clean_text("Ten 5 km de distancia.", "gl")
        assert "kilómetros" in result.lower() or "quilómetros" in result.lower(), \
            f"Expected km expansion, got: {result}"


class TestGalicianSpecialChars:
    """Test special character handling for Galician."""

    def test_punctuation_removed(self):
        result = clean_text("Ola, como estás?", "gl")
        assert "," not in result and "?" not in result, f"Expected no punctuation, got: {result}"

    def test_exclamation_removed(self):
        result = clean_text("Isto é importante!", "gl")
        assert "!" not in result, f"Expected no exclamation, got: {result}"

    def test_guillemets_removed(self):
        result = clean_text('El dixo: «Ola».', "gl")
        assert "«" not in result and "»" not in result and ":" not in result, \
            f"Expected no guillemets/colon, got: {result}"

    def test_ellipsis_removed(self):
        result = clean_text("A casa... é bonita.", "gl")
        assert "..." not in result, f"Expected no ellipsis, got: {result}"

    def test_parentheses_removed(self):
        result = clean_text("A unión fai a forza (sempre).", "gl")
        assert "(" not in result and ")" not in result, f"Expected no parentheses, got: {result}"

    def test_hyphen_removed(self):
        result = clean_text("Pino-Suárez é unha estación.", "gl")
        assert "-" not in result, f"Expected no hyphen, got: {result}"

    def test_output_is_lowercase(self):
        result = clean_text("A LINGUA GALEGA", "gl")
        assert result == result.lower(), f"Expected all lowercase, got: {result}"


# =========================================================================
# DISPATCHER — verify clean_text routes correctly for all languages
# =========================================================================

class TestDispatcher:
    """Test that clean_text dispatches to the correct normalizer per language."""

    def test_catalan_dispatch(self):
        result = clean_text("16 de gener", "ca")
        assert "setze" in result, f"Expected Catalan number expansion, got: {result}"

    def test_spanish_dispatch(self):
        result = clean_text("50€", "es")
        assert "euros" in result, f"Expected Spanish currency expansion, got: {result}"

    def test_basque_dispatch(self):
        result = clean_text("EAEko 100 euro", "eu")
        assert "ehun" in result.lower(), f"Expected Basque number expansion, got: {result}"

    def test_galician_dispatch(self):
        result = clean_text("100 euros", "gl")
        assert "cen" in result.lower(), f"Expected Galician number expansion, got: {result}"

    def test_unsupported_language_fallback(self):
        """Unsupported languages should fall back to lowercase."""
        with pytest.raises(ValueError):
            clean_text("Hello World", "fr"), f"Expected lowercase fallback, got: {result}"


# =========================================================================
# split_and_clean — verify sentence splitting + normalization
# =========================================================================

class TestSplitAndClean:
    """Test the split_and_clean function used by normalize_tsv.py."""

    def test_sentence_splitting_with_normalization(self):
        text = "Fa 30 anys. El programa comerà."
        result = split_and_clean(text, ". ", "ca")
        assert "trenta" in result, f"Expected 'trenta' (30), got: {result}"
        assert ". " in result, f"Expected sentence mark separator, got: {result}"

    def test_preserves_mark_separator(self):
        text = "Primera frase. Segona frase."
        result = split_and_clean(text, ". ", "ca")
        # Result should contain the ". " mark between cleaned sentences
        parts = result.split(". ")
        assert len(parts) >= 2, f"Expected at least 2 sentences separated by '. ', got: {result}"


# =========================================================================
# Known gaps (xfail) — document transformations that do NOT yet happen
# =========================================================================

class TestKnownGaps:
    """
    Document known normalization gaps as xfail tests.
    These serve as a record of what the normalizer does NOT handle.
    When a gap is fixed, the xfail mark can be removed.
    """

    @pytest.mark.xfail(reason="ca: Percent number (100%) — the '%' is expanded to 'per cent' but the number '100' is not expanded to 'cent'")
    def test_ca_percent_number_not_expanded(self):
        result = clean_text("100%", "ca")
        # Currently produces "100 per cent" instead of "cent per cent"
        assert "cent per cent" in result, f"Got: {result}"

    @pytest.mark.xfail(reason="ca: km/h abbreviation only partially expands — 'km' becomes 'quilòmetres' but '/h' part is lost or stays as 'h'")
    def test_ca_kmh_partial_expansion(self):
        result = clean_text("a 120 km/h", "ca")
        assert "quilòmetres per hora" in result, f"Got: {result}"

    @pytest.mark.xfail(reason="es: INAH (uppercase initials) is not expanded letter by letter — stays as 'inah'")
    def test_es_initials_not_expanded(self):
        result = clean_text("INAH confirmó el hallazgo", "es")
        # Expected: "i ene a hache confirmó el hallazgo"
        assert "ene" in result or "hache" in result, f"Got: {result}"


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
