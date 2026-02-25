from __future__ import annotations

import argparse
import mimetypes
import re
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable

import requests

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _enable_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


class BackendApi:
    def __init__(self, base_url: str, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        r = requests.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
        if not r.ok:
            detail = r.text
            try:
                data = r.json()
                detail = data.get("detail", data)
            except Exception:
                pass
            raise RuntimeError(f"{method} {path} failed ({r.status_code}): {detail}")
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return r.text

    def health(self) -> dict[str, Any]:
        return self._json("GET", "/health")

    def list_employees(self) -> list[dict[str, Any]]:
        return self._json("GET", "/employees")

    def create_employee(self, full_name: str, employee_code: str, status: str = "active") -> dict[str, Any]:
        payload = {"full_name": full_name, "employee_code": employee_code, "status": status}
        return self._json("POST", "/employees", json=payload)

    def enroll_images(self, employee_id: int, kind: str, image_paths: list[Path]) -> dict[str, Any]:
        endpoint = f"/employees/{employee_id}/enroll/{kind}"
        with ExitStack() as stack:
            files: list[tuple[str, tuple[str, Any, str]]] = []
            for p in image_paths:
                fh = stack.enter_context(open(p, "rb"))
                mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                files.append(("files", (p.name, fh, mime)))
            return self._json("POST", endpoint, files=files)


class CodeGenerator:
    def __init__(self, existing_codes: Iterable[str], prefix: str = "EMP", width: int = 4):
        self.prefix = prefix
        self.width = width
        self.used = {str(c) for c in existing_codes}
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        max_n = 0
        for code in self.used:
            m = pattern.match(code)
            if m:
                max_n = max(max_n, int(m.group(1)))
        self.next_n = max_n + 1 if max_n > 0 else 1

    def next_code(self) -> str:
        while True:
            code = f"{self.prefix}{self.next_n:0{self.width}d}"
            self.next_n += 1
            if code not in self.used:
                self.used.add(code)
                return code


def collect_images(folder: Path, recursive: bool) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    images = [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    images.sort(key=lambda p: p.name.lower())
    return images


def infer_image_kind(path: Path, default_kind: str) -> str:
    # Naming convention override: files starting with "re_" (case-insensitive) are treated as ReID images.
    if path.name.lower().startswith("re_"):
        return "reid"
    return default_kind


def split_images_by_kind(images: list[Path], default_kind: str) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {"face": [], "reid": []}
    for img in images:
        kind = infer_image_kind(img, default_kind=default_kind)
        groups[kind].append(img)
    return groups


def chunked(items: list[Path], size: int) -> list[list[Path]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Bulk import employees from folders. Each subfolder under --root is one person; "
            "folder name becomes full_name (Unicode supported, including Asian names)."
        )
    )
    p.add_argument("--root", required=True, help="Root folder containing one subfolder per person")
    p.add_argument("--backend", default="http://127.0.0.1:8000", help="Backend base URL")
    p.add_argument(
        "--kind",
        choices=["face", "reid"],
        default="face",
        help="Default enrollment type. Files prefixed with re_ (case-insensitive) are always uploaded as ReID.",
    )
    p.add_argument("--status", choices=["active", "inactive"], default="active")
    p.add_argument("--code-prefix", default="EMP")
    p.add_argument("--code-width", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=20, help="Images per upload request")
    p.add_argument("--recursive", action="store_true", help="Search images recursively inside each person folder")
    p.add_argument(
        "--existing",
        choices=["use", "skip", "error"],
        default="use",
        help="What to do if an employee with the same full_name already exists",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions without creating/uploading")
    return p.parse_args()


def main() -> int:
    _enable_utf8_stdio()
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"[error] root folder not found: {root}", file=sys.stderr)
        return 2

    api = BackendApi(args.backend)
    try:
        health = api.health()
        print(f"[backend] {args.backend} status={health.get('status')} db={health.get('db')}")
    except Exception as exc:
        print(f"[error] backend unavailable: {exc}", file=sys.stderr)
        return 2

    try:
        employees = api.list_employees()
    except Exception as exc:
        print(f"[error] failed to fetch employees: {exc}", file=sys.stderr)
        return 2

    by_name: dict[str, dict[str, Any]] = {}
    for e in employees:
        # If duplicates already exist, keep earliest ID deterministically.
        existing = by_name.get(e["full_name"])
        if existing is None or int(e["id"]) < int(existing["id"]):
            by_name[e["full_name"]] = e

    code_gen = CodeGenerator((e.get("employee_code", "") for e in employees), prefix=args.code_prefix, width=args.code_width)

    person_folders = [p for p in root.iterdir() if p.is_dir()]
    person_folders.sort(key=lambda p: p.name.casefold())
    if not person_folders:
        print(f"[warn] no person folders found under: {root}")
        return 0

    created_count = 0
    used_existing_count = 0
    skipped_count = 0
    upload_ok_people = 0
    upload_fail_people = 0
    total_images_attempted = 0

    for folder in person_folders:
        full_name = folder.name.strip()
        if not full_name:
            print(f"[skip] empty folder name: {folder}")
            skipped_count += 1
            continue

        images = collect_images(folder, recursive=args.recursive)
        if not images:
            print(f"[skip] {full_name} (no images in {folder})")
            skipped_count += 1
            continue

        total_images_attempted += len(images)
        emp = by_name.get(full_name)
        if emp is not None:
            if args.existing == "skip":
                print(f"[skip] {full_name} (employee exists id={emp['id']})")
                skipped_count += 1
                continue
            if args.existing == "error":
                print(f"[error] duplicate full_name exists in DB: {full_name} (id={emp['id']})", file=sys.stderr)
                return 3
            used_existing_count += 1
            employee_id = int(emp["id"])
            print(f"[use ] {full_name} -> employee_id={employee_id}, images={len(images)}")
        else:
            new_code = code_gen.next_code()
            print(f"[new ] {full_name} -> code={new_code}, images={len(images)}")
            if args.dry_run:
                employee_id = -1
            else:
                try:
                    created = api.create_employee(full_name=full_name, employee_code=new_code, status=args.status)
                except Exception as exc:
                    print(f"[fail] create employee for {full_name}: {exc}", file=sys.stderr)
                    upload_fail_people += 1
                    continue
                emp = created
                by_name[full_name] = emp
                employee_id = int(emp["id"])
                created_count += 1

        grouped = split_images_by_kind(images, default_kind=args.kind)
        face_count = len(grouped["face"])
        reid_count = len(grouped["reid"])
        if reid_count > 0:
            print(f"      detected prefix override: face={face_count}, reid={reid_count} (re_* -> reid)")
        else:
            print(f"      classified images: {args.kind}={len(images)}")

        if args.dry_run:
            for kind in ("face", "reid"):
                for chunk in chunked(grouped[kind], args.chunk_size):
                    print(f"      would upload {len(chunk)} {kind} image(s) for '{full_name}'")
            upload_ok_people += 1
            continue

        all_ok = True
        for kind in ("face", "reid"):
            for chunk_idx, img_chunk in enumerate(chunked(grouped[kind], args.chunk_size), start=1):
                try:
                    resp = api.enroll_images(employee_id=employee_id, kind=kind, image_paths=img_chunk)
                    inserted = resp.get("inserted_embeddings")
                    if inserted is None:
                        # Backend returns different keys for face vs reid endpoints
                        inserted = resp.get("inserted_embeddings", resp.get("total_reid_embeddings", "n/a"))
                    print(f"      {kind} chunk {chunk_idx}: uploaded={len(img_chunk)} inserted={inserted}")
                except Exception as exc:
                    all_ok = False
                    print(f"[fail] enroll {kind} for {full_name} (chunk {chunk_idx}): {exc}", file=sys.stderr)
        if all_ok:
            upload_ok_people += 1
        else:
            upload_fail_people += 1

    print("\nSummary")
    print(f"- root: {root}")
    print(f"- people folders: {len(person_folders)}")
    print(f"- images attempted: {total_images_attempted}")
    print(f"- created employees: {created_count}")
    print(f"- used existing employees: {used_existing_count}")
    print(f"- skipped folders: {skipped_count}")
    print(f"- successful people uploads: {upload_ok_people}")
    print(f"- failed people uploads: {upload_fail_people}")
    print(f"- name source: folder name (Unicode preserved)")
    print(f"- filename override: re_* (case-insensitive) => ReID enrollment")
    return 0 if upload_fail_people == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
