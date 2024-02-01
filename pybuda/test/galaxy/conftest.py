# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

@pytest.fixture
def microbatch(request):
    return int(request.config.getoption("--microbatch"))

@pytest.fixture
def loops(request):
    return int(request.config.getoption("--loops"))

def pytest_addoption(parser):
    parser.addoption(
        "--scan_chip",
        action="store",
        default="",
        help="Specify chips to test basic ethernet functionality."
    )
    parser.addoption(
        "--test_chips",
        action="store",
        default="",
        help="Target specific chips for bandwidth pushing."
    )
    parser.addoption(
        "--microbatch",
        action="store",
        default=1,
        help="Specify size of microbatch."
    )
    parser.addoption(
        "--loops",
        action="store",
        default=1,
        help="Specify number of loops."
    )

