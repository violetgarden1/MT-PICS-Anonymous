"""Rebuild the archived defense pools from pinned BEEAR reference workbooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from beear_artifacts import ANCHOR_POOLS, ROOT, encode, read_json, sha256, token_identity


def selected_rows(workbook, kind, row_numbers):
    from openpyxl import load_workbook

    expected = set(row_numbers)
    if len(expected) != len(row_numbers) or any(row < 2 for row in expected):
        raise ValueError(f"Invalid selection manifest: {kind}")
    selected = {}
    book = load_workbook(workbook, read_only=True, data_only=True)
    try:
        rows = book.active.iter_rows(values_only=True)
        header = list(next(rows))
        query_index, answer_index = header.index("query"), header.index("answer")
        for row_number, row in enumerate(rows, start=2):
            if row_number not in expected:
                continue
            query, answer = row[query_index], row[answer_index]
            if any(not isinstance(value, str) or not value.strip() for value in (query, answer)):
                raise ValueError(f"Empty reference field in {workbook.name}, row {row_number}")
            selected[row_number] = {
                "id": f"{kind}:{row_number}",
                "source_row": row_number,
                "query": query.strip(),
                "answer": answer.strip(),
            }
            if len(selected) == len(expected):
                break
    finally:
        book.close()
    if selected.keys() != expected:
        raise ValueError(f"Missing selected rows in {workbook.name}")
    return [selected[number] for number in row_numbers]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory with the three original BEEAR workbooks")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "defense")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    plan = read_json(ROOT / "anchor_selection.json")
    reference_path = ROOT / "anchor_manifest.json"
    reference = read_json(reference_path)
    source = args.source_dir.expanduser().resolve()
    for kind in ANCHOR_POOLS:
        workbook = source / plan["splits"][kind]["source_filename"]
        if sha256(workbook) != reference["source_sha256"][kind]:
            raise ValueError(f"Reference workbook hash mismatch: {workbook.name}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=False, local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if token_identity(tokenizer) != plan["tokenizer"]:
        raise ValueError("Tokenizer does not match the archived defense experiment")

    payloads = {}
    counts = {}
    for kind in ANCHOR_POOLS:
        selection = plan["splits"][kind]
        rows = selected_rows(source / selection["source_filename"], kind, selection["source_rows"])
        for row in rows:
            length = encode(tokenizer, row["query"], row["answer"])["length"]
            if length > plan["max_seq_length"]:
                raise ValueError(f"Overlong reference row: {row['id']}")
        newline = {"lf": "\n", "crlf": "\r\n"}[selection["newline"]]
        payload = "".join(json.dumps(row, ensure_ascii=False) + newline for row in rows).encode("utf-8")
        filename = f"{kind}.jsonl"
        if hashlib.sha256(payload).hexdigest() != reference["outputs"][filename]:
            raise ValueError(f"Rebuilt pool differs from the archived bytes: {filename}")
        payloads[filename] = payload
        counts[kind] = len(rows)

    output.mkdir(parents=True, exist_ok=True)
    for filename, payload in payloads.items():
        with (output / filename).open("xb") as stream:
            stream.write(payload)
    with (output / "manifest.json").open("xb") as stream:
        stream.write(reference_path.read_bytes())
    print(f"Verified and reconstructed defense reference pools at {output}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
