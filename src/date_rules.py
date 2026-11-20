"""Unambiguous project-specific date parsing and reference answer generation."""
import calendar
import datetime as dt
import re

MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})


def expand_year(year: str) -> int:
    if len(year) == 2:
        value = int(year)
        return 2000 + value if value <= 49 else 1900 + value
    return int(year)


def expected_iso(raw: str) -> str:
    """US numeric month/day/year; invalid dates yield INVALID."""
    raw = raw.strip().replace(',', ' ')
    raw = re.sub(r'(?<=\d)(st|nd|rd|th)\b', '', raw, flags=re.I)
    raw = re.sub(r'\s+', ' ', raw)
    try:
        m = re.fullmatch(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2}|\d{4})', raw)
        if m:
            month, day, year = m.groups()
        else:
            m = re.fullmatch(r'([A-Za-z]+) (\d{1,2}) (\d{2}|\d{4})', raw)
            if m:
                month, day, year = m.groups()
            else:
                m = re.fullmatch(r'(\d{1,2}) ([A-Za-z]+) (\d{2}|\d{4})', raw)
                if not m:
                    return 'INVALID'
                day, month, year = m.groups()
            month = MONTHS.get(month.lower(), 0)
        date = dt.date(expand_year(year), int(month), int(day))
        return date.isoformat()
    except (ValueError, OverflowError):
        return 'INVALID'
