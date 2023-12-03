# FAQ & Troubleshooting Guide

## Resetting an accelerator board

If a Tenstorrent chip seems to hang and/or is producing unexpected behaviour,
you may try a software reset of the board.

For Grayskull: `tt-smi -tr all`

If the software reset does not work, unfortunately you will have to power cycle
the board. This usually means rebooting the host of a board.

## `PermissionError` on `/tmp/*.lock` files

If multiple users are running on a system with a shared Tenstorrent device,
you may encounter a `PermissionError: [Errno 13] Permission denied: '/tmp/*.lock'`.

You would need to remove these files between user sessions.
