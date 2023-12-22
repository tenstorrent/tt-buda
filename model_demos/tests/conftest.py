import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pybuda
import pytest

environ_before_test = None


@pytest.fixture(autouse=True)
def clear_pybuda():
    yield

    # Clean up after each test
    pybuda.shutdown()
    pybuda.pybuda_reset()
    # TODO: For running on Silicon, reset tensix cores after each test
    # _ = subprocess.run(
    #     ["external_libs/pybuda/third_party/budabackend/device/bin/silicon/tensix-reset"]
    # )
    archive_files()
    _ = subprocess.run(["make", "clean_tt"])


def pytest_runtest_logreport(report):
    if report.when == "setup":
        global environ_before_test
        environ_before_test = os.environ.copy()
    elif report.when == "teardown":
        environ_before_test_keys = set(environ_before_test.keys())
        environ_after_test_keys = set(os.environ.keys())
        added_flags = environ_before_test_keys ^ environ_after_test_keys
        for f in added_flags:
            os.environ.pop(f, None)


def archive_files(src_directory=Path("./"), dest_directory=Path("archive")):
    """
    Archive files post run "_netlist.yaml" to dest_directory.

    Args:
    - src_directory (Path or str): The source directory where to look for files.
    - dest_directory (Path or str): The destination directory where to copy files. Defaults to "archive".
    """
    src_directory = Path(src_directory)
    dest_directory = Path(dest_directory)
    if not src_directory.exists():
        raise ValueError(f"Source directory {src_directory} does not exist!")

    if not dest_directory.exists():
        dest_directory.mkdir(parents=True)

    for file_path in src_directory.glob("*_netlist.yaml"):
        dt_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        dest_path = dest_directory / f"{file_path.stem}_{dt_str}{file_path.suffix}"
        try:
            shutil.copy(file_path, dest_path)
            print(f"Copied {file_path} to {dest_directory}")
        except Exception as e:
            print(f"Failed to copy {file_path}. Reason: {e}")


def pytest_addoption(parser):
    parser.addoption(
        "--silicon-only", action="store_true", default=False, help="run silicon tests only, skip golden/model"
    )
    parser.addoption("--no-silicon", action="store_true", default=False, help="skip silicon tests")
    parser.addoption(
        "--no-skips", action="store_true", default=False, help="ignore pytest.skip() calls, and continue on with test"
    )
