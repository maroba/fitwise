import re
from importlib.metadata import version

import fitwise


def test_version_matches_metadata():
    assert fitwise.__version__ == version("fitwise")


def test_version_is_pep440():
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.dev\d+)?", fitwise.__version__)
