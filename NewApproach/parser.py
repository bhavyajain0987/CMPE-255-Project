# parser.py
"""
Chunked parallel parser for BGL logs → structured Parquet.
Usage:
    python parser.py --input raw_data/BGL_150k.log \
                     --output data/bgl_structured.parquet \
                     [--chunksize 100000 --workers 4]
"""
import argparse
import pathlib
import pandas as pd
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

# Parser logic
def parse_line(line: str) -> dict | None:
    raw = line.strip().lstrip('- ')
    if not raw:
        return None
    parts = raw.split()
    # drop non-digit tokens until epoch
    while parts and not parts[0].isdigit():
        parts.pop(0)
    if len(parts) < 8:
        return None
    try:
        epoch     = int(parts[0])
        node_id   = parts[2]
        ts2       = parts[3]
        component = parts[6]
        severity  = parts[7].upper()
        message   = ' '.join(parts[8:])
        # parse timestamp
        tmp = ts2.replace('-', ' ', 3).replace('.', ':', 3)
        timestamp = pd.to_datetime(tmp, format="%Y %m %d %H:%M:%S:%f", utc=True, errors='coerce')
        if pd.isna(timestamp):
            return None
    except Exception:
        return None
    return {
        'epoch':       epoch,
        'node_id':     node_id,
        'timestamp':   timestamp,
        'component':   component,
        'severity':    severity,
        'template_id': -1,
        'message':     message,
    }

# Yield next n lines
def next_n_lines(fh, n: int):
    for _ in range(n):
        line = fh.readline()
        if not line:
            break
        yield line

# Main
def main(inp: str, out: str, chunksize: int, workers: int):
    inp_path = pathlib.Path(inp)
    out_path = pathlib.Path(out)
    temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="bgl_parts_"))
    parts = []
    part = 0
    with inp_path.open('r', encoding='utf-8', errors='ignore') as fh:
        while True:
            lines = list(next_n_lines(fh, chunksize))
            if not lines:
                break
            # parse chunk
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as exe:
                    futures = [exe.submit(parse_line, ln) for ln in lines]
                    records = [f.result() for f in as_completed(futures) if f.result()]
            else:
                records = [rec for ln in lines if (rec := parse_line(ln))]
            df = pd.DataFrame(records)
            part_file = temp_dir / f"part_{part}.parquet"
            df.to_parquet(part_file, index=False)
            print(f"✅ Wrote {len(df)} rows to {part_file.name}")
            parts.append(part_file)
            part += 1
    # merge
    print(f"Merging {part} parts...")
    combined = pd.concat(pd.read_parquet(p) for p in parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(f"🎉 Completed. Output: {out_path}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',      required=True)
    parser.add_argument('--output',     required=True)
    parser.add_argument('--chunksize',  type=int, default=100000)
    parser.add_argument('--workers',    type=int, default=1)
    args = parser.parse_args()
    main(args.input, args.output, args.chunksize, args.workers)