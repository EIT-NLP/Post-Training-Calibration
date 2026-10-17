#!/usr/bin/env python3
"""
No-early-exit baseline wrapper.

The baseline is equivalent to offline rollout collection with the paper
decoding settings, so this legacy entrypoint delegates to example_offline.py.
"""

from example_offline import main


if __name__ == "__main__":
    main()
