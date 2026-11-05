"""Export benchmark summaries using the UCM collector's existing XLSX package."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import statistics
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import collect_ucm_metrics as collector

FIELDS = (
    "run_id",
    "mode",
    "case",
    "routing",
    "dp",
    "input_len",
    "output_len",
    "target_hit_rate",
    "seed_tokens",
    "concurrency",
    "num_requests",
    "seed",
    "completed",
    "failed",
    "actual_input_min",
    "actual_input_max",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "tpot_mean_ms",
    "duration_s",
    "output_tokens_per_s",
    "total_tokens_per_s",
    "hbm_hit_rate",
    "external_hit_rate",
    "external_hit_rate_remaining",
    "host_ready_shard_fraction",
    "backend_load_shards",
    "posix_read_bytes",
    "status",
    "metrics_status",
)
RATE_FIELDS = {name for name in FIELDS if "rate" in name or "fraction" in name}
DEFINITIONS = (
    (
        "TPOT mean ms",
        "Mean per-request time per output token after the first token, in milliseconds.",
    ),
    (
        "Output tokens/s",
        "vLLM output_throughput: successful output tokens / benchmark duration.",
    ),
    (
        "Total tokens/s",
        "vLLM total_token_throughput: successful input plus output tokens / benchmark duration. Includes cached input tokens.",
    ),
    ("HBM hit rate", "HBM cached tokens / queried HBM prefix tokens."),
    ("External hit rate", "External cached tokens / queried HBM prefix tokens."),
    (
        "External hit rate remaining",
        "External cached tokens / externally queried tokens after HBM hits.",
    ),
    (
        "Host Ready shard fraction",
        "Ready Host Cache shards / successfully loaded shards. Includes all exported TP worker series.",
    ),
    (
        "Backend load shards",
        "Actual backend load submissions. Pending shared slots are not counted as Ready hits.",
    ),
    (
        "Posix read bytes",
        "Increase in posix_s2h_bytes_total, summed across exported workers.",
    ),
    (
        "Missing values",
        "Blank values indicate unavailable metrics. Ratios are calculated from summed counts.",
    ),
    (
        "Sources",
        "Each case_result.json records the bench JSON, measurement directory and metrics files.",
    ),
)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def performance(result: dict) -> dict:
    ttfts = result.get("ttfts", [])
    latencies = result.get("latencies", [])
    lengths = result.get("output_lens", [])
    tpots = [
        (latency - ttft) / (length - 1)
        for latency, ttft, length in zip(latencies, ttfts, lengths)
        if length > 1
    ]
    inputs = result.get("input_lens", [])
    return {
        "completed": result.get("completed"),
        "failed": result.get("failed", 0),
        "actual_input_min": min(inputs) if inputs else None,
        "actual_input_max": max(inputs) if inputs else None,
        "ttft_mean_ms": (
            statistics.mean(ttfts) * 1000 if ttfts else result.get("mean_ttft_ms")
        ),
        "ttft_p50_ms": (
            percentile(ttfts, 0.5) * 1000 if ttfts else result.get("median_ttft_ms")
        ),
        "ttft_p99_ms": (
            percentile(ttfts, 0.99) * 1000 if ttfts else result.get("p99_ttft_ms")
        ),
        "tpot_mean_ms": (
            statistics.mean(tpots) * 1000 if tpots else result.get("mean_tpot_ms")
        ),
        "duration_s": result.get("duration"),
        "output_tokens_per_s": result.get("output_throughput"),
        "total_tokens_per_s": result.get("total_token_throughput"),
    }


def column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def worksheet(headers: tuple, rows: list, rate_columns: set[int]) -> str:
    last_column = column_name(len(headers))
    pane = '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
    if len(headers) > 3:
        pane = '<pane xSplit="3" ySplit="1" topLeftCell="D2" activePane="bottomRight" state="frozen"/>'
    widths = []
    for index, header in enumerate(headers):
        longest = max([len(str(header)), *[len(str(row[index] or "")) for row in rows]])
        widths.append(min(48, max(14, longest + 2)))
    xml_rows = []
    for row_number, values in enumerate([headers, *rows], start=1):
        cells = []
        for index, value in enumerate(values, start=1):
            style = 3 if row_number == 1 else 4
            if row_number > 1 and isinstance(value, (float, int)):
                style = (
                    10
                    if index in rate_columns
                    else (6 if isinstance(value, int) else 5)
                )
            cells.append(
                collector._xlsx_cell(f"{column_name(index)}{row_number}", value, style)
            )
        lines = max(
            math.ceil(len(str(value or "")) / max(1, widths[index] - 2))
            for index, value in enumerate(values)
        )
        height = max(36 if row_number == 1 else 20, lines * 18)
        xml_rows.append(
            f'<row r="{row_number}" ht="{height}" customHeight="1">{"".join(cells)}</row>'
        )
    column_xml = [
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths, 1)
    ]
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{last_column}{len(rows) + 1}"/>'
        '<sheetViews><sheetView workbookViewId="0" showGridLines="0">'
        f"{pane}"
        '</sheetView></sheetViews><sheetFormatPr defaultRowHeight="20"/>'
        f'<cols>{"".join(column_xml)}</cols><sheetData>{"".join(xml_rows)}</sheetData>'
        f'<autoFilter ref="A1:{last_column}{len(rows) + 1}"/></worksheet>'
    )


def write_workbook(path: Path, headers: tuple, rows: list) -> None:
    # Reuse the user's collector for the workbook container and its styles.
    template = path.with_name(f".{path.stem}.{os.getpid()}.template.xlsx")
    pages = {key: ([], []) for key, _, _ in collector.REPORT_PAGES}
    try:
        collector.write_report_xlsx(template, pages, 0)
        with zipfile.ZipFile(template) as source:
            contents = {name: source.read(name) for name in source.namelist()}
        namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        ET.register_namespace("", namespace)
        ET.register_namespace(
            "r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        )
        styles = ET.fromstring(contents["xl/styles.xml"])
        num_formats = styles.find(f"{{{namespace}}}numFmts")
        ET.SubElement(
            num_formats, f"{{{namespace}}}numFmt", numFmtId="165", formatCode="0.0%"
        )
        num_formats.set("count", str(len(num_formats)))
        cell_formats = styles.find(f"{{{namespace}}}cellXfs")
        ET.SubElement(
            cell_formats,
            f"{{{namespace}}}xf",
            numFmtId="165",
            fontId="0",
            fillId="0",
            borderId="0",
            xfId="0",
            applyNumberFormat="1",
        )
        cell_formats.set("count", str(len(cell_formats)))
        for cell_format in cell_formats:
            alignment = cell_format.find(f"{{{namespace}}}alignment")
            if alignment is not None:
                alignment.set("wrapText", "1")
        contents["xl/styles.xml"] = ET.tostring(
            styles, encoding="utf-8", xml_declaration=True
        )
        workbook = ET.fromstring(contents["xl/workbook.xml"])
        sheets = workbook.find(f"{{{namespace}}}sheets")
        titles = ["Summary", "Definitions"]
        for sheet, title in zip(list(sheets), titles):
            sheet.set("name", title)
        sheets.remove(list(sheets)[2])
        contents["xl/workbook.xml"] = ET.tostring(
            workbook, encoding="utf-8", xml_declaration=True
        )
        for package_path in ("[Content_Types].xml", "xl/_rels/workbook.xml.rels"):
            element = ET.fromstring(contents[package_path])
            for child in list(element):
                target = child.get("PartName", child.get("Target", ""))
                if target.endswith("sheet3.xml"):
                    element.remove(child)
            ET.register_namespace("", element.tag.split("}")[0][1:])
            contents[package_path] = ET.tostring(
                element, encoding="utf-8", xml_declaration=True
            )
        contents.pop("xl/worksheets/sheet3.xml")
        rate_columns = {
            index for index, name in enumerate(headers, 1) if name in RATE_FIELDS
        }
        contents["xl/worksheets/sheet1.xml"] = worksheet(
            headers, rows, rate_columns
        ).encode()
        definitions = [*DEFINITIONS, ("Results directory", str(path.parent.resolve()))]
        contents["xl/worksheets/sheet2.xml"] = worksheet(
            ("Metric", "Definition"), definitions, set()
        ).encode()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as output:
            for name, data in contents.items():
                output.writestr(name, data)
        output_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        output_path.write_bytes(buffer.getvalue())
        output_path.replace(path)
    finally:
        template.unlink(missing_ok=True)


def write_summary(directory: Path) -> list[dict]:
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.rglob("case_result.json"))
    ]
    # Older case summaries omitted total throughput; recover it from bench JSON.
    for result in results:
        missing = [
            field
            for field in ("tpot_mean_ms", "output_tokens_per_s", "total_tokens_per_s")
            if result.get(field) is None
        ]
        source = result.get("bench_result_path")
        if missing and source and Path(source).is_file():
            recovered = performance(
                json.loads(Path(source).read_text(encoding="utf-8"))
            )
            for field in missing:
                result[field] = recovered.get(field)
    rows = [[result.get(field) for field in FIELDS] for result in results]
    with (directory / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        writer = csv.writer(output)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    write_workbook(directory / "summary.xlsx", FIELDS, rows)
    headers = (
        "Mode",
        "Case",
        "Target hit",
        "HBM hit",
        "External hit",
        "Host Ready",
        "Posix bytes",
        "TTFT ms",
        "TPOT ms",
        "Output tok/s",
        "Total tok/s",
        "Status",
        "Metrics",
    )
    table = []
    for result in results:
        ttft = result.get("ttft_mean_ms")
        tpot = result.get("tpot_mean_ms")
        output_throughput = result.get("output_tokens_per_s")
        total_throughput = result.get("total_tokens_per_s")
        values = [
            result.get(field)
            for field in (
                "target_hit_rate",
                "hbm_hit_rate",
                "external_hit_rate",
                "host_ready_shard_fraction",
            )
        ]
        rates = [f"{value:.1%}" if value is not None else "n/a" for value in values]
        posix_bytes = result.get("posix_read_bytes")
        table.append(
            (
                result["mode"],
                result["case"],
                *rates,
                f"{posix_bytes:,.0f}" if posix_bytes is not None else "n/a",
                f"{ttft:.2f}" if ttft is not None else "n/a",
                f"{tpot:.2f}" if tpot is not None else "n/a",
                f"{output_throughput:.2f}" if output_throughput is not None else "n/a",
                f"{total_throughput:.2f}" if total_throughput is not None else "n/a",
                result["status"],
                result.get("metrics_status", "unavailable"),
            )
        )
    widths = [
        max([len(header), *[len(row[index]) for row in table]])
        for index, header in enumerate(headers)
    ]
    for row in [headers, *table]:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))
    print(
        f"Summary saved to {directory / 'summary.csv'} and {directory / 'summary.xlsx'}"
    )
    return results
