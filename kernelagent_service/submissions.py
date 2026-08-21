"""Normalize SOL-style uploaded kernel submissions into task input files."""

from __future__ import annotations

import ast
import base64
import binascii
import io
import json
import re
import tarfile
import zipfile
from pathlib import PurePosixPath

from kernelagent_service.models import InputFile

SUPPORTED_SUFFIXES = (
    ".py",
    ".cu",
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".json",
    ".zip",
    ".tar.gz",
    ".tgz",
)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_EXTRACTED_BYTES = 100 * 1024 * 1024
MAX_FILES = 64


class SubmissionError(ValueError):
    """A safe-to-display submission validation error."""


def submission_suffix(filename: str) -> str:
    lower = filename.lower()
    for suffix in (".tar.gz", ".tgz", ".cpp", ".cxx", ".json", ".zip", ".py", ".cu", ".cc", ".c"):
        if lower.endswith(suffix):
            return suffix
    return PurePosixPath(lower).suffix or "unknown"


def normalize_submission(filename: str, encoded_content: str) -> tuple[list[InputFile], str, str]:
    """Return materialized files, candidate entry point, and detected format."""
    name = PurePosixPath(filename).name
    suffix = submission_suffix(name)
    if suffix not in SUPPORTED_SUFFIXES:
        raise SubmissionError(f"{suffix} format is currently not supported")
    try:
        raw = base64.b64decode(encoded_content, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SubmissionError("uploaded file content is not valid base64") from exc
    if len(raw) > MAX_UPLOAD_BYTES:
        raise SubmissionError("uploaded file exceeds the 10 MB limit")

    if suffix == ".json":
        return _normalize_json(raw)
    if suffix in {".zip", ".tar.gz", ".tgz"}:
        files = _read_archive(raw, suffix)
        entrypoint = _find_archive_entrypoint(files)
        return _prefix(files), f"candidate/{entrypoint}", "archive"

    text = _decode_source(raw, name)
    if suffix == ".py":
        _validate_python_run(text, name)
        detected = "python"
    elif suffix in {".cpp", ".cc", ".cxx", ".c"}:
        _validate_host_entrypoint(text, name)
        detected = "cuda_cpp"
    else:
        # A standalone .cu is useful to the repo's parser/generator even though
        # SOL evaluation normally pairs it with a host binding in an archive.
        detected = "cuda"
    path = f"candidate/{name}"
    return [InputFile(path=path, content=text)], path, detected


def _normalize_json(raw: bytes) -> tuple[list[InputFile], str, str]:
    text = _decode_source(raw, "solution.json")
    try:
        solution = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SubmissionError(f"invalid JSON solution at line {exc.lineno}") from exc
    spec = solution.get("spec")
    sources = solution.get("sources")
    if not isinstance(spec, dict) or not isinstance(sources, list):
        raise SubmissionError("JSON solution requires top-level spec and sources fields")
    languages = spec.get("languages")
    entry = spec.get("entry_point")
    if (
        not isinstance(languages, list)
        or not languages
        or not all(isinstance(x, str) for x in languages)
    ):
        raise SubmissionError("JSON solution spec.languages must be a non-empty string array")
    if not isinstance(entry, str) or "::" not in entry:
        raise SubmissionError("JSON solution spec.entry_point must use filename::function format")
    entry_path, function = entry.rsplit("::", 1)
    if not entry_path or not function:
        raise SubmissionError("JSON solution spec.entry_point must use filename::function format")
    files: list[InputFile] = []
    for source in sources:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("path"), str)
            or not isinstance(source.get("content"), str)
        ):
            raise SubmissionError("each JSON source requires string path and content fields")
        files.append(InputFile(path=source["path"], content=source["content"]))
    _validate_file_set(files)
    if entry_path not in {item.path for item in files}:
        raise SubmissionError("JSON solution entry point is missing from sources")
    manifest = InputFile(path="solution.json", content=text)
    _validate_file_set([manifest, *files])
    return _prefix([manifest, *files]), f"candidate/{entry_path}", "+".join(languages)


def _read_archive(raw: bytes, suffix: str) -> list[InputFile]:
    files: list[InputFile] = []
    total = 0
    try:
        if suffix == ".zip":
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    mode = info.external_attr >> 16
                    if mode & 0o170000 == 0o120000:
                        raise SubmissionError("archive symlinks are not allowed")
                    total += info.file_size
                    _check_archive_limits(files, total)
                    files.append(
                        InputFile(
                            path=_safe_path(info.filename),
                            content=_decode_source(
                                archive.read(info), info.filename
                            ),
                        )
                    )
        else:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as archive:
                for member in archive.getmembers():
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise SubmissionError("archive links and special files are not allowed")
                    total += member.size
                    _check_archive_limits(files, total)
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise SubmissionError(f"cannot read archive member: {member.name}")
                    files.append(
                        InputFile(
                            path=_safe_path(member.name),
                            content=_decode_source(handle.read(), member.name),
                        )
                    )
    except (
        zipfile.BadZipFile,
        tarfile.TarError,
        RuntimeError,
        OSError,
        EOFError,
    ) as exc:
        raise SubmissionError("uploaded archive is invalid or corrupted") from exc
    if not files:
        raise SubmissionError("uploaded archive contains no source files")
    _validate_file_set(files)
    return files


def _find_archive_entrypoint(files: list[InputFile]) -> str:
    python_entries = [item for item in files if PurePosixPath(item.path).name == "submission.py"]
    if len(python_entries) == 1:
        _validate_python_run(python_entries[0].content, python_entries[0].path)
        return python_entries[0].path
    host_entries = [
        item
        for item in files
        if submission_suffix(item.path) in {".cpp", ".cc", ".cxx", ".c"}
        and "PYBIND11_MODULE" in item.content
    ]
    if len(host_entries) == 1:
        _validate_host_entrypoint(host_entries[0].content, host_entries[0].path)
        return host_entries[0].path
    if len(python_entries) > 1 or len(host_entries) > 1:
        raise SubmissionError("archive contains multiple possible entry points")
    raise SubmissionError(
        "archive requires submission.py with run() or one C/C++ file with PYBIND11_MODULE"
    )


def _validate_python_run(content: str, name: str) -> None:
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        raise SubmissionError(f"{name} is not valid Python at line {exc.lineno}") from exc
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run"
        for node in tree.body
    ):
        raise SubmissionError(f"{name} must define a top-level run() function")


def _validate_host_entrypoint(content: str, name: str) -> None:
    if "PYBIND11_MODULE" not in content:
        raise SubmissionError(f"{name} must contain a PYBIND11_MODULE binding")
    if not re.search(r"\brun\s*\(", content):
        raise SubmissionError(f"{name} must define a run() function")


def _decode_source(raw: bytes, name: str) -> str:
    if b"\0" in raw:
        raise SubmissionError(f"{name} contains null bytes")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SubmissionError(f"{name} must contain UTF-8 source text") from exc


def _safe_path(value: str) -> str:
    if "\0" in value or "\\" in value:
        raise SubmissionError("archive contains an unsafe path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SubmissionError("archive contains an unsafe path")
    return path.as_posix()


def _check_archive_limits(files: list[InputFile], total: int) -> None:
    if len(files) >= MAX_FILES:
        raise SubmissionError(f"archive contains more than {MAX_FILES} files")
    if total > MAX_EXTRACTED_BYTES:
        raise SubmissionError("archive expands beyond the 100 MB limit")


def _validate_file_set(files: list[InputFile]) -> None:
    paths = [item.path for item in files]
    if len(paths) != len(set(paths)):
        raise SubmissionError("submission contains duplicate source paths")
    if len(files) > MAX_FILES:
        raise SubmissionError(f"submission contains more than {MAX_FILES} files")


def _prefix(files: list[InputFile]) -> list[InputFile]:
    return [InputFile(path=f"candidate/{item.path}", content=item.content) for item in files]
