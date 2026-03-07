from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

import pandas as pd

T = TypeVar("T")
U = TypeVar("U")



def parallel_map(
    func: Callable[[T], U],
    items: Sequence[T],
    *,
    workers: int,
    chunksize: int = 1,
) -> list[U]:
    if workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, items, chunksize=chunksize))



def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
