"""Multi-tenant concurrent-slowdown report: xlsx from result/multitenant/ CSVs.

Discovers every result/multitenant/<mode>/concurrent_slowdown.csv (one folder
per power mode, produced by multitenant_sweep.py) and also tolerates the legacy
flat file result/multitenant/concurrent_slowdown.csv (mode read from the
t0_nvpmodel column). Combines all modes into one workbook at
result/multitenant/multitenant_report.xlsx, with modes as column groups
mirroring compare_devices.py's device-side-by-side layout.

Sheets
------
legend       -- column and metric glossary, validity notes.
interference -- per-cell slowdown (per tenant) and P95 ratio.
throughput   -- STP, ANTT, throughput_gain, agg_throughput.
memory       -- per-tenant and pair-level GPU memory.
power_energy -- pair_power_w and pair_energy_j per cell per mode.
validity     -- timed_overlap_frac, calib_fallback, nvpmodel per cell per mode.

Usage
-----
    python multitenant_report.py                    # reads result/multitenant/
    python multitenant_report.py --out path.xlsx    # custom output path
    python multitenant_report.py --selftest         # offline smoke test
    python multitenant_report.py --help
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MT_DIR = REPO_ROOT / "result" / "multitenant"
DEFAULT_OUT = MT_DIR / "multitenant_report.xlsx"

OVERLAP_WARN = 0.9   # cells below this fraction are flagged in the validity sheet

# Colour palette: matches compare_devices._DEVCLR for MAXN/25W/15W; adds MAXN_SUPER.
# Each entry: (header fill hex, light body tint hex).
_MODECLR = {
    "MAXN_SUPER": ("2E7D32", "E8F5E9"),
    "MAXN":       ("55A868", "E6F1EA"),
    "25W":        ("4C72B0", "E4EAF3"),
    "15W":        ("C44E52", "F6E5E7"),
}
_MODECLR_DEFAULT = ("757575", "F5F5F5")   # fallback for unknown modes


def _modeclr(mode: str):
    return _MODECLR.get(mode.upper(), _MODECLR_DEFAULT)


# ---- CSV discovery and loading -----------------------------------------------

def discover_csv_files(base_dir: Path) -> dict:
    """Return {mode_label: Path} for every concurrent_slowdown.csv found.

    Per-mode layout: base_dir/<mode>/concurrent_slowdown.csv. The folder name
    is used as the mode label (upper-cased for consistency with compare_devices).

    Legacy flat file: base_dir/concurrent_slowdown.csv. The mode label is read
    from the t0_nvpmodel column of the first parseable row; falls back to
    "UNKNOWN" when the column is absent or empty.
    """
    found = {}

    # Per-mode subfolders first.
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        csv_path = child / "concurrent_slowdown.csv"
        if csv_path.exists():
            found[child.name.upper()] = csv_path

    # Legacy flat file (not in a subfolder).
    legacy = base_dir / "concurrent_slowdown.csv"
    if legacy.exists():
        mode = _mode_from_flat_csv(legacy)
        # Do not overwrite a per-mode file that already covers this mode.
        if mode not in found:
            found[mode] = legacy

    return found


def _mode_from_flat_csv(path: Path) -> str:
    """Read the t0_nvpmodel value from the first parseable row of a flat CSV."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "UNKNOWN"
    lines = text.splitlines()
    if len(lines) < 2:
        return "UNKNOWN"
    header = lines[0].split(",")
    try:
        idx = header.index("t0_nvpmodel")
    except ValueError:
        return "UNKNOWN"
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) > idx:
            val = parts[idx].strip()
            if val:
                return val.upper()
    return "UNKNOWN"


def load_csv(path: Path) -> list:
    """Read a CSV file into a list of dicts, skipping unparseable lines.

    A truncated final line (no comma) is silently skipped. This mirrors the
    defensive reading already used in multitenant_sweep._count_csv_rows.
    """
    rows = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"[report] warning: cannot read {path}: {exc}")
        return rows
    lines = text.splitlines()
    if not lines:
        return rows
    header = lines[0].split(",")
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if "," not in stripped:
            # Truncated line: skip without raising.
            print(f"[report] skipped truncated line in {path}: {stripped!r}")
            continue
        parts = stripped.split(",")
        rows.append(dict(zip(header, parts)))
    return rows


# ---- cell key ----------------------------------------------------------------

def cell_key(row: dict) -> str:
    """Build a stable human-readable key for a CSV row.

    For a two-tenant row with t0='bert@0' and t1='bert@0', the key is
    'bert@0 + bert@0 (n=2)'. For a single tenant it is 'bert@0 (n=1)'.
    The key is deterministic and identical across mode files for the same
    cell, so rows align when building the side-by-side layout.
    """
    n = int(row.get("n_tenants") or 1)
    parts = []
    for i in range(n):
        val = row.get(f"t{i}", "")
        parts.append(val)
    tenant_str = " + ".join(parts) if parts else "?"
    return f"{tenant_str} (n={n})"


# ---- loading all modes -------------------------------------------------------

def load_all_modes(base_dir: Path) -> tuple:
    """Return (mode_order, data) where data is {mode: {cell_key: row}}.

    mode_order is a list of mode labels in discovery order, with MAXN_SUPER,
    MAXN, 25W, 15W sorted first when present, then any others alphabetically.
    """
    csv_files = discover_csv_files(base_dir)
    if not csv_files:
        return [], {}

    _PREFERRED_ORDER = ["MAXN_SUPER", "MAXN", "25W", "15W"]
    preferred = [m for m in _PREFERRED_ORDER if m in csv_files]
    others = sorted(m for m in csv_files if m not in _PREFERRED_ORDER)
    mode_order = preferred + others

    data = {}
    for mode in mode_order:
        rows = load_csv(csv_files[mode])
        data[mode] = {cell_key(r): r for r in rows}

    return mode_order, data


# ---- styling helpers (in the spirit of compare_devices._style_sheet) ---------

def _style_header_row(ws, modes, submetric_groups, col_offset=2):
    """Write and colour the two-row header for a metric sheet.

    Row 1: merged sub-metric group label spanning all mode columns.
    Row 2: one mode label per column (coloured by mode).

    Returns the total number of data columns written (excluding column A).
    """
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")
    white = Font(bold=True, color="FFFFFF")

    # Column A: "Cell" label spanning two rows.
    ws.merge_cells("A1:A2")
    a = ws["A1"]
    a.value = "Cell"
    a.font = white
    a.alignment = center
    a.border = border
    a.fill = PatternFill("solid", fgColor="404040")

    col = col_offset
    for group_label, _cols in submetric_groups:
        span = len(modes) * len(_cols)
        ws.merge_cells(start_row=1, start_column=col, end_row=1,
                       end_column=col + span - 1)
        h = ws.cell(row=1, column=col)
        h.value = group_label
        h.font = white
        h.alignment = center
        h.border = border
        h.fill = PatternFill("solid", fgColor="595959")
        # Row 2: one column per (sub-metric, mode) pair.
        for _sub_label, _sub_key in _cols:
            for mode in modes:
                hdr_fill, _ = _modeclr(mode)
                v = ws.cell(row=2, column=col)
                v.value = f"{mode}\n{_sub_label}"
                v.font = white
                v.alignment = Alignment(horizontal="center", vertical="center",
                                        wrap_text=True)
                v.border = border
                v.fill = PatternFill("solid", fgColor=hdr_fill)
                col += 1

    return col - col_offset


def _style_data_rows(ws, modes, submetric_groups, start_row=3):
    """Apply borders, fill, and number format to data cells."""
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")

    for r in range(start_row, ws.max_row + 1):
        # Column A: cell key label.
        a = ws.cell(row=r, column=1)
        a.font = Font(bold=True)
        a.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        a.border = border

        col = 2
        for _group_label, sub_cols in submetric_groups:
            for _sub_label, _sub_key in sub_cols:
                for mode in modes:
                    _, body_fill = _modeclr(mode)
                    c = ws.cell(row=r, column=col)
                    c.border = border
                    c.alignment = center
                    c.fill = PatternFill("solid", fgColor=body_fill)
                    if isinstance(c.value, float):
                        c.number_format = "0.####"
                    col += 1

    ws.column_dimensions["A"].width = 32
    for cc in range(2, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(cc)].width = 10
    ws.freeze_panes = "B3"
    ws.row_dimensions[1].height = 18
    ws.row_dimensions[2].height = 36


def _style_legend(ws):
    from openpyxl.styles import PatternFill, Font, Alignment
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="404040")
        c.alignment = Alignment(horizontal="left", vertical="center")
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 72


def _write_legend_sheet(ws):
    """Write the legend rows directly into an openpyxl worksheet."""
    from openpyxl.styles import Alignment
    ws.append(["Term", "Meaning"])
    for term, meaning in _legend_rows():
        ws.append([term, meaning])
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[1].height = 18


# ---- legend sheet ------------------------------------------------------------

def _legend_rows():
    """Return [(Term, Meaning)] for the legend sheet."""
    rows = [
        ("Cell", "Tenant descriptor(s) and tenant count, e.g. 'bert@0 + yolo@3_P3 (n=2)'."),
        ("Mode columns", "One column per power mode (MAXN_SUPER, MAXN, 25W, 15W)."),
        ("t{i}_slowdown", "Shared latency / solo latency for tenant i. Lower is better. "
                          "1.0 means no interference."),
        ("t{i}_p95_ratio", "Shared P95 tail latency / solo P95 for tenant i. Lower is better."),
        ("STP", "System Throughput (Eyerman & Eeckhout 2008): sum_i(shared_thru_i / solo_thru_i). "
                "STP > 1 means co-location is profitable."),
        ("ANTT", "Average Normalised Turnaround Time: mean_i(slowdown_i). The SLO perspective."),
        ("throughput_gain", "sum(shared_thru_i) / sum(solo_thru_i). Fraction of solo capacity "
                            "recovered under co-location."),
        ("agg_throughput", "Sum of shared throughputs across all tenants. Only meaningful "
                           "(same units) when agg_throughput_comparable is True, i.e. all "
                           "tenants share one model family."),
        ("t{i}_gpu_mem_static_mb", "Static GPU memory for tenant i (weights + persistent "
                                    "buffers). This is the admission cost: independent of "
                                    "exit index and batch size."),
        ("t{i}_gpu_mem_dynamic_mb", "Dynamic GPU memory for tenant i (activations, KV cache, "
                                     "workspace). This moves with the exit index and decides "
                                     "whether co-tenants survive a joint peak."),
        ("t{i}_gpu_mem_peak_mb", "Peak GPU memory for tenant i under co-location."),
        ("pair_gpu_mem_static_mb", "Sum of static memory across all tenants (admission cost)."),
        ("pair_gpu_mem_peak_mb", "Sum of peak memory across all tenants (joint-peak risk)."),
        ("pair_power_w", "DEVICE-AGGREGATE power in Watts. The Jetson exposes one shared "
                         "INA3221 rail; this is NOT per-tenant. Do not sum across tenants."),
        ("pair_energy_j", "DEVICE-AGGREGATE energy in Joules (pair_power_w x window). "
                          "Same rail caveat: this is the device cost for the co-located "
                          "window, not per-tenant energy."),
        ("timed_overlap_frac", "Fraction of the measurement window where all tenants were "
                                "simultaneously active (profiler-timed). Values below "
                                f"{OVERLAP_WARN:.0%} are suspect: little true co-location "
                                "occurred, so interference metrics understate the real effect."),
        ("t{i}_calib_fallback", "True when phase-2 calibration fell back to forward latency "
                                 "(wall time was unavailable). The calibrated window is then "
                                 "biased; treat interference numbers from such rows with caution."),
        ("t{i}_nvpmodel", "Jetson nvpmodel mode string recorded in device_caps at run time. "
                           "Confirms which power mode the measurement was actually taken at."),
        ("LOW OVERLAP flag", f"Cells where timed_overlap_frac < {OVERLAP_WARN:.0%} are "
                              "flagged in the validity sheet. Their interference metrics "
                              "are likely understated."),
    ]
    return rows


# ---- generic sheet writer ----------------------------------------------------

def _val(row, key):
    """Return float value from row dict, or None."""
    v = row.get(key, "")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return str(v)


def _write_metric_sheet(ws, cell_keys, modes, data, submetric_groups, note=None):
    """Fill a metric sheet.

    submetric_groups: list of (group_display_label, [(sub_label, csv_col_template)])
    where csv_col_template may contain {i} to be formatted per tenant (expanded
    for each tenant found in the row), or be a plain column name.
    """
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

    _style_header_row(ws, modes, submetric_groups)

    r = 3
    for ck in cell_keys:
        ws.cell(row=r, column=1).value = ck
        col = 2
        for _group_label, sub_cols in submetric_groups:
            for sub_label, sub_key_tmpl in sub_cols:
                for mode in modes:
                    row = data.get(mode, {}).get(ck)
                    val = None
                    if row is not None:
                        # Expand {i} for per-tenant columns: collect values for
                        # all tenants and display them as a newline-separated string
                        # (e.g. "1.23\n1.45" for a two-tenant cell).
                        if "{i}" in sub_key_tmpl:
                            n = int(row.get("n_tenants") or 1)
                            vals = []
                            for i in range(n):
                                v = _val(row, sub_key_tmpl.format(i=i))
                                if v is not None:
                                    vals.append(f"{v:.4g}" if isinstance(v, float) else str(v))
                            val = "\n".join(vals) if vals else None
                        else:
                            val = _val(row, sub_key_tmpl)
                    c = ws.cell(row=r, column=col)
                    c.value = val
                    if isinstance(val, str) and "\n" in val:
                        c.alignment = Alignment(wrap_text=True, horizontal="center",
                                                vertical="center")
                    col += 1
        r += 1

    _style_data_rows(ws, modes, submetric_groups)
    if note:
        _add_note_row(ws, note, r)


def _add_note_row(ws, note, row_idx):
    """Append a note line below the data."""
    from openpyxl.styles import Font, Alignment
    c = ws.cell(row=row_idx + 1, column=1)
    c.value = f"Note: {note}"
    c.font = Font(italic=True, color="666666")
    c.alignment = Alignment(wrap_text=True)
    ws.merge_cells(start_row=row_idx + 1, start_column=1,
                   end_row=row_idx + 1, end_column=ws.max_column or 1)


# ---- sheet definitions -------------------------------------------------------

def _union_cell_keys(modes, data) -> list:
    """Stable-ordered union of all cell keys across all modes."""
    seen = {}
    for mode in modes:
        for ck in data.get(mode, {}):
            seen[ck] = True
    return list(seen.keys())


def write_report(base_dir: Path = MT_DIR, out_path: Path = DEFAULT_OUT):
    """Discover CSVs, build the workbook, write out_path."""
    try:
        import openpyxl
    except ImportError as exc:
        print(f"[report] openpyxl missing ({exc}); pip install openpyxl")
        return False

    mode_order, data = load_all_modes(base_dir)
    if not mode_order:
        print(f"[report] no concurrent_slowdown.csv found under {base_dir}")
        return False

    cell_keys = _union_cell_keys(mode_order, data)
    print(f"[report] modes: {mode_order}  cells: {len(cell_keys)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    # Remove the default sheet openpyxl creates.
    wb.remove(wb.active)

    # legend
    ws_legend = wb.create_sheet("legend")
    _write_legend_sheet(ws_legend)
    _style_legend(ws_legend)

    # interference: slowdown (per tenant) and P95 ratio
    ws_int = wb.create_sheet("interference")
    _write_metric_sheet(
        ws_int,
        cell_keys, mode_order, data,
        submetric_groups=[
            ("Slowdown (lower = less interference)",
             [("t{i} slowdown", "t{i}_slowdown")]),
            ("P95 latency ratio (lower = better tail)",
             [("t{i} P95 ratio", "t{i}_p95_ratio")]),
        ],
        note="Slowdown = shared latency / solo latency per tenant. "
             "P95 ratio = shared P95 / solo P95. Both lower is better.",
    )

    # throughput: STP, ANTT, throughput_gain, agg_throughput
    ws_thru = wb.create_sheet("throughput")
    _write_metric_sheet(
        ws_thru,
        cell_keys, mode_order, data,
        submetric_groups=[
            ("STP (> 1 = co-location profitable)",
             [("STP", "stp")]),
            ("ANTT (lower = better QoS)",
             [("ANTT", "antt")]),
            ("Throughput gain",
             [("gain", "throughput_gain")]),
            ("Aggregate throughput",
             [("agg_thru", "agg_throughput"),
              ("comparable?", "agg_throughput_comparable")]),
        ],
        note="STP > 1 means co-location recovered more than one solo run's worth of "
             "throughput. agg_throughput is only meaningful when agg_throughput_comparable "
             "is True (all tenants share one model family).",
    )

    # memory: per-tenant static/dynamic/peak + pair-level
    ws_mem = wb.create_sheet("memory")
    _write_metric_sheet(
        ws_mem,
        cell_keys, mode_order, data,
        submetric_groups=[
            ("Per-tenant static GPU mem MB (admission cost)",
             [("t{i} static", "t{i}_gpu_mem_static_mb")]),
            ("Per-tenant dynamic GPU mem MB (moves with exit)",
             [("t{i} dynamic", "t{i}_gpu_mem_dynamic_mb")]),
            ("Per-tenant peak GPU mem MB",
             [("t{i} peak", "t{i}_gpu_mem_peak_mb")]),
            ("Pair-level GPU mem MB",
             [("pair static", "pair_gpu_mem_static_mb"),
              ("pair peak", "pair_gpu_mem_peak_mb")]),
        ],
        note="Static memory is the admission cost (weights, resident always). "
             "Dynamic memory moves with the exit index. "
             "Pair peak is the joint-peak risk: does the board survive if both "
             "transient peaks coincide?",
    )

    # power_energy: device-aggregate power and energy
    ws_pe = wb.create_sheet("power_energy")
    _write_metric_sheet(
        ws_pe,
        cell_keys, mode_order, data,
        submetric_groups=[
            ("Pair power W (device-aggregate)",
             [("power W", "pair_power_w")]),
            ("Pair energy J (device-aggregate)",
             [("energy J", "pair_energy_j")]),
        ],
        note="Both metrics are device-aggregate from one shared INA3221 rail on the Jetson. "
             "They are NOT per-tenant. Do not sum across tenants or modes.",
    )

    # validity: overlap, calib_fallback, nvpmodel
    ws_val = wb.create_sheet("validity")
    _write_validity_sheet(ws_val, cell_keys, mode_order, data)

    wb.save(str(out_path))
    print(f"[report] wrote {out_path}")
    return True


def _write_validity_sheet(ws, cell_keys, modes, data):
    """Validity sheet: overlap fraction, calib_fallback, nvpmodel per cell per mode.

    Cells with timed_overlap_frac below OVERLAP_WARN are flagged with a red fill
    so a reader can identify suspect measurements at a glance.
    """
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")
    white = Font(bold=True, color="FFFFFF")
    warn_fill = PatternFill("solid", fgColor="FFCCCC")   # red tint for low overlap

    # Two-row header.
    ws.merge_cells("A1:A2")
    a = ws["A1"]
    a.value = "Cell"; a.font = white; a.alignment = center; a.border = border
    a.fill = PatternFill("solid", fgColor="404040")

    sub_labels = ["timed_overlap", "calib_fallback", "nvpmodel"]
    col = 2
    for mode in modes:
        hdr_fill, _ = _modeclr(mode)
        ws.merge_cells(start_row=1, start_column=col,
                       end_row=1, end_column=col + len(sub_labels) - 1)
        h = ws.cell(row=1, column=col)
        h.value = mode; h.font = white; h.alignment = center; h.border = border
        h.fill = PatternFill("solid", fgColor=hdr_fill)
        for sl in sub_labels:
            v = ws.cell(row=2, column=col)
            v.value = sl; v.font = white; v.alignment = center; v.border = border
            v.fill = PatternFill("solid", fgColor=hdr_fill)
            col += 1

    # Data rows.
    for ri, ck in enumerate(cell_keys):
        r = ri + 3
        a = ws.cell(row=r, column=1)
        a.value = ck
        a.font = Font(bold=True)
        a.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        a.border = border

        col = 2
        for mode in modes:
            _, body_fill = _modeclr(mode)
            row = data.get(mode, {}).get(ck)

            # timed_overlap_frac
            ov = _val(row, "timed_overlap_frac") if row else None
            c_ov = ws.cell(row=r, column=col)
            c_ov.value = ov
            c_ov.border = border; c_ov.alignment = center
            if isinstance(ov, float) and ov < OVERLAP_WARN:
                c_ov.fill = warn_fill
                c_ov.font = Font(bold=True, color="CC0000")
            else:
                c_ov.fill = PatternFill("solid", fgColor=body_fill)
            if isinstance(ov, float):
                c_ov.number_format = "0.000"

            # calib_fallback (per-tenant, concatenated)
            fb_vals = []
            if row:
                n = int(row.get("n_tenants") or 1)
                for i in range(n):
                    v = row.get(f"t{i}_calib_fallback", "")
                    fb_vals.append(str(v))
            c_fb = ws.cell(row=r, column=col + 1)
            c_fb.value = ", ".join(fb_vals) if fb_vals else None
            c_fb.border = border; c_fb.alignment = center
            c_fb.fill = PatternFill("solid", fgColor=body_fill)

            # nvpmodel (from t0_nvpmodel; should match the mode folder name)
            nv = row.get("t0_nvpmodel", "") if row else ""
            c_nv = ws.cell(row=r, column=col + 2)
            c_nv.value = nv if nv else None
            c_nv.border = border; c_nv.alignment = center
            c_nv.fill = PatternFill("solid", fgColor=body_fill)

            col += len(sub_labels)

    ws.column_dimensions["A"].width = 32
    for cc in range(2, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(cc)].width = 14
    ws.freeze_panes = "B3"
    ws.row_dimensions[1].height = 18
    ws.row_dimensions[2].height = 18

    # Append a note explaining the flag.
    note_row = len(cell_keys) + 4
    from openpyxl.styles import Font as _Font, Alignment as _Align
    c = ws.cell(row=note_row, column=1)
    c.value = (f"Red = timed_overlap_frac < {OVERLAP_WARN:.0%}. "
               "Little true co-location occurred; interference metrics are understated.")
    c.font = _Font(italic=True, color="CC0000")
    c.alignment = _Align(wrap_text=True)
    if ws.max_column > 1:
        ws.merge_cells(start_row=note_row, start_column=1,
                       end_row=note_row, end_column=ws.max_column)


# ---- selftest ----------------------------------------------------------------

def _selftest():
    """Build a workbook from a small synthetic in-memory CSV covering two modes."""
    import tempfile, os

    tmp = Path(tempfile.mkdtemp())

    # Two per-mode CSV files under tmp/result/multitenant/<mode>/
    header = (
        "tag,n_tenants,overlap_frac,timed_overlap_frac,"
        "t0,t0_slowdown,t0_p95_ratio,"
        "t1,t1_slowdown,t1_p95_ratio,"
        "stp,antt,throughput_gain,agg_throughput,agg_throughput_comparable,"
        "t0_gpu_mem_static_mb,t0_gpu_mem_dynamic_mb,t0_gpu_mem_peak_mb,"
        "t1_gpu_mem_static_mb,t1_gpu_mem_dynamic_mb,t1_gpu_mem_peak_mb,"
        "pair_gpu_mem_static_mb,pair_gpu_mem_peak_mb,"
        "pair_power_w,pair_energy_j,"
        "t0_calib_fallback,t1_calib_fallback,"
        "t0_nvpmodel,t1_nvpmodel"
    )
    row_25w = (
        "run25w,2,0.95,0.97,"
        "bert@0,1.15,1.20,"
        "yolo@3_P3,1.08,1.12,"
        "1.85,1.115,0.91,112.3,False,"
        "700.0,120.0,820.0,"
        "400.0,80.0,480.0,"
        "1100.0,1300.0,"
        "8.5,255.0,"
        "False,False,"
        "25W,25W"
    )
    row_15w = (
        "run15w,2,0.94,0.96,"
        "bert@0,1.25,1.31,"
        "yolo@3_P3,1.18,1.22,"
        "1.72,1.215,0.87,98.1,False,"
        "700.0,125.0,825.0,"
        "400.0,83.0,483.0,"
        "1100.0,1308.0,"
        "6.2,186.0,"
        "False,False,"
        "15W,15W"
    )
    # Low-overlap row to test validity flagging.
    row_15w_low = (
        "run15w_low,2,0.4,0.42,"
        "bert@12,1.35,1.45,"
        "yolo@5_P5,1.22,1.28,"
        "1.60,1.285,0.80,88.0,False,"
        "700.0,200.0,900.0,"
        "400.0,90.0,490.0,"
        "1100.0,1390.0,"
        "6.0,180.0,"
        "False,False,"
        "15W,15W"
    )

    for mode, row_data in [("25w", row_25w), ("15w", row_15w + "\n" + row_15w_low)]:
        mode_dir = tmp / "result" / "multitenant" / mode
        mode_dir.mkdir(parents=True)
        (mode_dir / "concurrent_slowdown.csv").write_text(
            header + "\n" + row_data, encoding="utf-8"
        )

    base = tmp / "result" / "multitenant"
    out = base / "multitenant_report.xlsx"
    ok = write_report(base_dir=base, out_path=out)
    assert ok, "write_report returned False"
    assert out.exists(), f"output file not found: {out}"

    # Verify sheet names.
    import openpyxl
    wb = openpyxl.load_workbook(str(out))
    expected_sheets = {"legend", "interference", "throughput", "memory",
                       "power_energy", "validity"}
    got = set(wb.sheetnames)
    missing = expected_sheets - got
    assert not missing, f"missing sheets: {missing}"
    print(f"[selftest] sheets present: {sorted(got)}")

    # Verify discovery.
    mode_order, data = load_all_modes(base)
    assert set(mode_order) == {"25W", "15W"}, f"unexpected modes: {mode_order}"
    assert "bert@0 + yolo@3_P3 (n=2)" in data["25W"], "cell key not found"
    assert "bert@0 + yolo@3_P3 (n=2)" in data["15W"], "cell key alignment failed"
    print(f"[selftest] discovery: modes={mode_order}  cell keys align: OK")

    # Verify validity sheet flags the low-overlap row.
    ws_v = wb["validity"]
    flag_found = False
    for row in ws_v.iter_rows(min_row=3):
        cell_label = row[0].value or ""
        if "bert@12" in cell_label:
            # The timed_overlap column for 15W should be red-flagged (< 0.9).
            # We check the numeric value rather than the fill (fill is an object).
            for c in row[1:]:
                if isinstance(c.value, float) and c.value < OVERLAP_WARN:
                    flag_found = True
    assert flag_found, "low-overlap cell not flagged in validity sheet"
    print("[selftest] low-overlap flag in validity sheet: OK")

    print(f"[selftest] workbook written to {out}  OK")


# ---- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--out", default=str(DEFAULT_OUT), metavar="PATH",
        help=f"output xlsx path (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--base", default=str(MT_DIR), metavar="DIR",
        help=f"directory to discover CSVs from (default: {MT_DIR})",
    )
    ap.add_argument(
        "--selftest", action="store_true",
        help="run offline smoke test with synthetic data and exit",
    )
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return

    ok = write_report(base_dir=Path(a.base), out_path=Path(a.out))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
