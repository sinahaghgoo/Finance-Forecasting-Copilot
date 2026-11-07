"""Download the Rossmann Store Sales dataset from Kaggle and build the panel.

Prerequisites
-------------
1. Create a Kaggle API token at https://www.kaggle.com/settings/account
   and place it at ~/.kaggle/kaggle.json (chmod 600).
2. Accept the competition rules at
   https://www.kaggle.com/competitions/rossmann-store-sales/rules
3. pip install -e ".[kaggle]"
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
