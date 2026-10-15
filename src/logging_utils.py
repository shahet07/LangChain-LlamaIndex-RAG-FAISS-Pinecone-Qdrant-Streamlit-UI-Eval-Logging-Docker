import os, json, time
from contextlib import contextmanager
from pathlib import Path

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

@contextmanager
def timeit():
    start = time.perf_counter()
    try:
        yield lambda: time.perf_counter() - start
    finally:
        pass

def append_jsonl(record: dict, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
