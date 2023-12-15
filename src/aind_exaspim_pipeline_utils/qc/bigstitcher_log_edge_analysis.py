"""Utility functions to search for unfitted tile edges in bigstitcher log output"""
import re
import sys
from typing import Iterable, List, Tuple
import numpy as np


def get_unfitted_tile_pairs(lines: Iterable[str]) -> List[List[Tuple[int, int]]]:  # pragma: no cover
    """Extract tile pair numbers from log lines of failed RANSAC correspondence
    finding.

    Extract the tile number pairs from failed RANSAC correspondence finding log messages. The
    results are grouped in blocks based on the RANSAC log messages, i.e. we assume that
    each registration round has the RANSAC related lines in consecutive lines in the input.
    Each non-RANSAC log line starts a new output block.

    Only pairs with a tile number difference of 1 or 3 are considered an edge failure.

    Parameters
    ----------
    lines : iterable
      Lines of log output.

    Returns
    -------
    blocks : List of lists of pairs.
      The extracted number pairs collected in lists.
    """
    newblock = True
    blocks = []
    pairs = []
    for line in lines:
        line = line.strip()
        # RANSAC result lines
        m = re.match(".*\\[TP=(\\d+) ViewId=(\\d+) >>> TP=(\\d+) ViewId=(\\d+)\\]: (\\w+)", line)
        if m:
            if m.group(5) == "NO" or m.group(5) == "Not":  # Failed RANSAC
                a = int(m.group(4))
                b = int(m.group(2))
            else:
                # Successful RANSAC line
                continue
        else:
            # Non-RANSAC line
            newblock = True
            continue
        if b < a:
            a, b = b, a
        if b - a in (1, 3):
            # Failed edge
            if newblock:
                if pairs:
                    blocks.append(pairs)
                    pairs = []
                newblock = False
            pairs.append((a, b))
    if pairs:
        blocks.append(pairs)
    return blocks


def create_ascii_visualization(pairs: Iterable[Tuple[int, int]]) -> str:  # pragma: no cover
    """Assuming the 3x5 tile configuration of exaSPIM, show which edges have failed.

    Tile grid is 3x5 with tile 0 is top right, tile 14 is bottom left. Tiles are marked
    with hexadecimal digits: 0x0-0xe

    Parameters
    ----------
    pairs: Iterable
        Tile number pairs marking edges in the 3x5 grid to visualize. Edges are marked by `X`.

    Returns
    -------
    r : Multi-line string
    """
    A = np.zeros(dtype="U", shape=(5, 9))
    A.fill(".")
    A[0::2, 0::2] = np.array(list("0369c147ad258be"), dtype="U").reshape((3, 5))
    for p in pairs:
        a, b = p
        if b - a == 3:
            A[(a % 3) * 2, (a // 3) * 2 + 1] = "X"
        elif b - a == 1:
            A[(a % 3) * 2 + 1, (a // 3) * 2] = "X"
        else:
            raise ValueError
    return "\n".join(" ".join(x[::-1]) for x in A)


def print_visualization(blocks, file=None):  # pragma: no cover
    """Print ascii visualization of edges for multiple blocks."""

    for i, pairs in enumerate(blocks):
        print(f"Log block {i:1d}:", file=file)
        print("=====================", file=file)
        print(create_ascii_visualization(pairs), file=file)
        print(file=file)


def main():  # pragma: no cover
    """Standalone script entry point."""
    # By default, process stdin
    blocks = get_unfitted_tile_pairs(sys.stdin)
    print_visualization(blocks)


if __name__ == "__main__":  # pragma: no cover
    main()
