# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from . import layers
from . import models
from .utils import check_version

__version__ = '0.2.7'
check_version(__version__)