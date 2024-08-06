# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

def pytest_configure(config):
    config.addinivalue_line(
        "markers", 'slow: marks tests as slow (deselect with -m "not slow")'
    )
    config.addinivalue_line(
        "markers", 'run_in_pp: marks tests to run in pipeline'
    )
