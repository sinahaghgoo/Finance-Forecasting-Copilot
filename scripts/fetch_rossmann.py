"""Download the Rossmann Store Sales dataset from Kaggle and build the panel.

Uses the public CC0 mirror at kaggle.com/datasets/pratyushakar/rossmann-store-sales
(same train.csv and store.csv as the original competition, no rules acceptance needed).

Prerequisites
-------------
1. pip install -e ".[kaggle]"
2. Authenticate once with the Kaggle CLI:
     kaggle auth login              # OAuth flow, caches ~/.kaggle/credentials.json
   or, with the classic API token:
     https://www.kaggle.com/settings/api  → Create New Token  → ~/.kaggle/kaggle.json
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fin_copilot.data.rossmann import to_panel  # noqa: E402


def main() -> None:
    panel = to_panel(save=True)
    print(f"Rossmann panel: {len(panel):,} rows across "
          f"{panel['category'].nunique()} store types × "
          f"{panel['region'].nunique()} assortments, "
          f"{panel['date'].min().date()} to {panel['date'].max().date()}.")


if __name__ == "__main__":
    main()
