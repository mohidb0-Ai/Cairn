"""Run every test from a throwaway working directory so `.cairn/` never
touches the real project checkout.

Author: Mohid Bin Farooq
"""

import os
import shutil
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_cwd():
    original = os.getcwd()
    workdir = tempfile.mkdtemp()
    os.chdir(workdir)
    try:
        yield
    finally:
        os.chdir(original)
        shutil.rmtree(workdir, ignore_errors=True)
