"""Generate diverse deterministic examples; split by distinct calendar dates."""
import argparse
import csv
import datetime as dt
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.date_rules import expected_iso


def make_rows(n=600, seed=42):
    rng = random.Random(seed)
    rows = []
    seen = set()
    attempts = 0
    while len(rows) < n and attempts < n * 100:
        attempts += 1
        year = rng.randint(1950, 2049)
        month = rng.randint(1, 12)
        day = rng.randint(1, 31)
        invalid = rng.random() < .16
        if not invalid:
            try:
                dt.date(year, month, day)
            except ValueError:
                continue
        else:
            # Guaranteed invalid day for this month.
            import calendar
            day = rng.randint(calendar.monthrange(year, month)[1] + 1, 31) if calendar.monthrange(year, month)[1] < 31 else 32
        style = rng.randrange(5)
        year_text = f'{year % 100:02d}' if style in (1, 4) else str(year)
        if style == 0:
            raw = f'{dt.date(2000, month, 1):%B} {day}, {year_text}'
        elif style == 1:
            raw = f'{month}/{day}/{year_text}'
        elif style == 2:
            raw = f'{day} {dt.date(2000, month, 1):%b} {year_text}'
        elif style == 3:
            raw = f'{month:02d}-{day:02d}-{year_text}'
        else:
            raw = f'{month:02d}/{day:02d}/{year_text}'
        if raw in seen:
            continue
        seen.add(raw)
        rows.append({'prompt': raw, 'expected_answer': expected_iso(raw)})
    if len(rows) != n:
        raise RuntimeError('Could not generate enough unique rows')
    return rows


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['prompt', 'expected_answer'])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=600)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='data/dates.csv')
    args = p.parse_args()
    write_csv(args.out, make_rows(args.count, args.seed))
    print(f'Generated {args.count} examples at {args.out}')
