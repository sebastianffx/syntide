"""Check that ECG-QA-CoT's 500 Hz leads and PTB-XL's 100 Hz leads are the same recordings.

SYNTIDE reads the 100 Hz signals (the editor, its calibration and Gate 4 work at 100 Hz). This check
decimates a few 500 Hz recordings to 100 Hz and compares them lead by lead, so a mismatch in the
record join or the lead order shows up before anything trains. PTB-XL made its 100 Hz files with its
own filter, so the two never match exactly: typical leads correlate at 0.98-0.998, and only
low-amplitude leads (where noise dominates) fall to about 0.95. A wrong record or lead order gives a
correlation near zero, so a record whose median lead correlation is below ``MIN_MEDIAN_CORR`` fails.
"""

from collections.abc import Sequence

import numpy as np
from scipy.signal import decimate

from syntide.data.schema import LEADS

MIN_MEDIAN_CORR = 0.9


class SignalMismatchError(ValueError):
    """ecg-qa-cot's and PTB-XL's signals for one record id are not the same recording."""


def signal_agreement(adapter, record_ids: Sequence[str]) -> dict:
    """Per-record correlation and RMS difference between decimated 500 Hz and stored 100 Hz leads."""
    if not record_ids:
        return {"records": {}, "summary": "no records checked"}
    low = adapter.load(record_ids).x.astype(np.float64)
    high = adapter.raw_signals(record_ids)
    records = {}
    for k, record_id in enumerate(record_ids):
        down = decimate(high[record_id], 5, ftype="fir", zero_phase=True, axis=-1)
        n = min(down.shape[-1], low.shape[-1])
        a, b = down[:, :n], low[k, :, :n]
        corr = [_corr(a[i], b[i]) for i in range(len(LEADS))]
        records[record_id] = {
            "median_lead_corr": round(float(np.median(corr)), 4),
            "min_lead_corr": round(min(corr), 4),
            "rms_diff_mv": round(float(np.sqrt(np.mean((a - b) ** 2))), 4),
            "rms_signal_mv": round(float(np.sqrt(np.mean(b**2))), 4),
        }
    worst = min(r["min_lead_corr"] for r in records.values())
    bad = {r: v for r, v in records.items() if v["median_lead_corr"] < MIN_MEDIAN_CORR}
    if bad:
        raise SignalMismatchError(f"500 Hz and 100 Hz leads differ (median corr < {MIN_MEDIAN_CORR}): {bad}")
    return {
        "records": records,
        "summary": f"{len(records)} ECGs, worst lead correlation {worst:.4f}, "
        f"max RMS difference {max(r['rms_diff_mv'] for r in records.values()):.4f} mV",
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if not a.std() or not b.std():
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])
