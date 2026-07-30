"""Output helpers for noisy native chemistry libraries."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def suppress_native_output(*, stdout: bool = True, stderr: bool = True) -> Iterator[None]:
    """Temporarily silence output written directly to process file descriptors.

    Python stream redirection does not catch messages emitted by C/C++ libraries
    such as OpenBabel and RDKit. This context is intended for short, single-threaded
    chemistry calls because file-descriptor redirection affects the whole process.
    """
    file_descriptors = [descriptor for descriptor, enabled in ((1, stdout), (2, stderr)) if enabled]
    if not file_descriptors:
        yield
        return

    for stream in (sys.stdout, sys.stderr):
        stream.flush()

    saved_descriptors: dict[int, int] = {}
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        try:
            for descriptor in file_descriptors:
                saved_descriptors[descriptor] = os.dup(descriptor)
                os.dup2(devnull.fileno(), descriptor)
            yield
        finally:
            for stream in (sys.stdout, sys.stderr):
                stream.flush()
            for descriptor, saved_descriptor in saved_descriptors.items():
                os.dup2(saved_descriptor, descriptor)
                os.close(saved_descriptor)
