#!/usr/bin/env python3
"""
windowizer.py

Sliding-window generator for structured BGL logs → windowed Parquet.
Drops empty windows (no events) to ensure every window has a valid node_id.

Usage:
    python windowizer.py --input data/bgl_structured.parquet \
                         --output data/bgl_windows_v1.parquet \
                         --window 5 --lookahead 10 --stride 1
"""
import argparse
import pathlib
import pandas as pd

def main(inp, out, window, lookahead, stride):
    df = pd.read_parquet(inp)

    # Round timestamps down to the nearest minute
    df['minute'] = df['timestamp'].dt.floor('T')
    all_windows = []

    # Process each node separately
    for node, g in df.groupby('node_id'):
        g = g.sort_values('timestamp')

        # Per-minute aggregations
        temp_counts = g.pivot_table(
            index='minute',
            columns='template_id',
            aggfunc='size',
            fill_value=0
        )
        sev_counts = g.pivot_table(
            index='minute',
            columns='severity',
            aggfunc='size',
            fill_value=0
        )
        total_counts = g.groupby('minute').size().rename('n_events')
        fail_counts = (
            g[g['severity']=='FATAL']
             .groupby('minute').size()
             .rename('n_failures')
        )

        # Combine and fill missing
        df_min = pd.concat([
            total_counts,
            temp_counts,
            sev_counts,
            fail_counts
        ], axis=1).fillna(0)

        # Sliding window sums
        roll = df_min.rolling(window=window, min_periods=1).sum()
        roll.index.name = 'window_start'

        # Drop windows with zero events
        roll = roll[ roll['n_events'] > 0 ]

        # Lookahead labeling functions
        failure_minutes = df_min.index[df_min['n_failures'] > 0]
        def has_fail(ts):
            return int(
                ((failure_minutes > ts) &
                 (failure_minutes <= ts + pd.Timedelta(minutes=lookahead)))
                .any()
            )
        def time_to(ts):
            fut = failure_minutes[failure_minutes > ts]
            if fut.empty:
                return float(lookahead)
            return float((fut.min() - ts).total_seconds() / 60)

        # Apply labels
        roll['y_cls'] = roll.index.to_series().apply(has_fail).values
        roll['y_reg'] = roll.index.to_series().apply(time_to).values
        # Binary indicator if event occurs before lookahead
        roll['delta'] = (roll['y_reg'] < lookahead).astype(int).values

        # Apply stride (subsampling)
        if stride > 1:
            roll = roll.iloc[::stride]

        # Add metadata
        roll['node_id'] = node
        roll['hour']    = roll.index.hour

        # Collect
        all_windows.append(roll.reset_index())

    # Concatenate all nodes' windows
    out_df = pd.concat(all_windows, ignore_index=True)

    # Write to Parquet
    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)

    print(f"✅ {len(out_df)} windows → {out}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input',     required=True, help='Parquet input of structured logs')
    p.add_argument('--output',    required=True, help='Parquet output of windowed data')
    p.add_argument('--window',    type=int, default=5, help='Window size in minutes')
    p.add_argument('--lookahead', type=int, default=10, help='Lookahead in minutes')
    p.add_argument('--stride',    type=int, default=1, help='Stride for sliding window')
    args = p.parse_args()
    main(args.input, args.output, args.window, args.lookahead, args.stride)
