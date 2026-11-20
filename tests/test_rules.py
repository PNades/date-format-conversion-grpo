from src.date_rules import expected_iso, expand_year
from src.rewards import score, grpo_reward
from scripts.generate_dataset import make_rows


def test_examples():
    assert expected_iso('March 3rd, 2026') == '2026-03-03'
    assert expected_iso('3/3/26') == '2026-03-03'
    assert expected_iso('3 Mar 2026') == '2026-03-03'
    assert expected_iso('03-03-2026') == '2026-03-03'
    assert expected_iso('03/04/26') == '2026-03-04'
    assert expected_iso('Feb 29, 2024') == '2024-02-29'
    assert expected_iso('February 30, 2026') == 'INVALID'
    assert expected_iso('12/31/99') == '1999-12-31'
    assert expected_iso('1 January 2000') == '2000-01-01'
    assert expected_iso('02/29/2025') == 'INVALID'


def test_century_and_validation():
    assert expand_year('49') == 2049
    assert expand_year('50') == 1950
    assert expected_iso('2/29/2000') == '2000-02-29'
    assert expected_iso('2/29/1900') == 'INVALID'
    assert expected_iso('13/1/2026') == 'INVALID'


def test_rewards():
    assert score('2026-03-03', '2026-03-03') == 1
    assert score('2026-03-04', '2026-03-03') == .4
    assert score('2025-03-03', '2026-03-03') == .4
    assert score('INVALID', 'INVALID') == 1
    assert score('2026-02-30', 'INVALID') == 0
    assert score('2026-02-30', '2026-02-28') == 0
    assert grpo_reward(['2026-03-03', 'INVALID'], ['2026-03-03', 'INVALID']) == [1, 1]


def test_dataset():
    rows = make_rows(600)
    assert len(rows) == 600
    assert len({r['prompt'] for r in rows}) == 600
    assert all(expected_iso(r['prompt']) == r['expected_answer'] for r in rows)
    assert any(r['expected_answer'] == 'INVALID' for r in rows)
