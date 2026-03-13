"""
clean_and_expand.py
===================
Text normalization (cleaning and expanding) script.

CLI wrapper for fsp.core.text.clean_text

use:
    python3 clean_and_expand.py -s "<string>" -l "<lang>" <flags>

Flags:
    -p: keep punctuation
    -c: keep capitalisation
"""

from optparse import OptionParser

# Import core logic from fsp package
from fsp.core.text import clean_text


def main(argv=None):
    parser = OptionParser()
    parser.add_option("-s", "--sent", dest="sent", action="store", help="sentence to normalize")
    parser.add_option("-l", "--lang", dest="lang", action="store", help="language")
    parser.add_option(
        "-p", "--punt", dest="punt", action="store_true", help="keep punctuation", default=False
    )
    parser.add_option(
        "-c", "--cap", dest="cap", action="store_true", help="keep capitalisation", default=False
    )

    options, args = parser.parse_args(argv)

    sent = options.sent
    lang = options.lang
    punctuation = options.punt
    capitalisation = options.cap
    print(sent)
    print(clean_text(sent, lang, punctuation, capitalisation))


if __name__ == "__main__":
    main()
