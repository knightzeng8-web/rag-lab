"""把任意文档目录转换成 rag-lab 的规范语料。

为什么需要这个脚本：真实项目里语料是"一堆乱七八糟的文件"（PDF、Word、Markdown、
导出的网页、几万字的教材），而索引质量的第一步就是**把它们变成结构一致、doc_id 稳定、
可追溯来源的语料**。这个脚本负责这一步，且不引入任何第三方依赖。

它做四件事：
1. **读取**：复用 `rag.ingest.loader`，支持 md / txt / json / jsonl / csv / docx / pdf；
2. **规范化**：压缩空白、去零宽字符；
3. **拆分**（可选）：把一本"大书"按指定层级的标题拆成多篇文档，让 doc_id 的粒度更合理；
4. **去重与登记**：按内容哈希去重、按 token Jaccard 过滤近重复，并把来源/许可/指纹写进
   `data/manifests/corpus.json`（**不写进正文**，避免污染检索与生成）。

用法：
    python scripts/build_corpus.py <输入目录或文件…> --out data/corpus
    python scripts/build_corpus.py .\\我的资料 --out data/corpus --split-level 2 --dry-run
    python scripts/build_corpus.py paper.pdf --out data/corpus --prefix ext
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag.ingest.loader import iter_files, load_one  # noqa: E402
from rag.text import first_heading, normalize_ws, tokenize  # noqa: E402

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
ILLEGAL_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
NEAR_DUP_THRESHOLD = 0.90


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def slugify(title: str, fallback: str = "doc", limit: int = 48) -> str:
    """把标题变成安全的文件名主干。

    保留中文（Windows 文件名允许），只替换非法字符；标题过长时截断，
    避免生成 `xxxxx…xxx.md` 这种在资源管理器里无法辨认的名字。
    """
    name = ILLEGAL_NAME.sub("-", (title or "").strip())
    name = re.sub(r"\s+", "-", name).strip("-._")
    name = name[:limit].strip("-._")
    return name or fallback


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def near_duplicate(tokens: set, accepted: Sequence[set], threshold: float = NEAR_DUP_THRESHOLD) -> bool:
    """按 token Jaccard 判断近重复（只跟已接受的文档比，避免 O(n^2) 爆炸得太厉害）。"""
    if not tokens:
        return False
    for other in accepted:
        if not other:
            continue
        inter = len(tokens & other)
        if inter == 0:
            continue
        union = len(tokens | other)
        if union and inter / union >= threshold:
            return True
    return False


def split_by_heading(text: str, level: int) -> List[Tuple[str, str]]:
    """按 `level` 级标题把长文档拆成多节，返回 [(标题, 正文)]；level<=0 表示不拆。"""
    if level <= 0:
        return [(first_heading(text), text)]
    sections: List[Tuple[str, str]] = []
    current_title = ""
    buffer: List[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_title or first_heading(text), body))

    for line in text.split("\n"):
        match = HEADING_RE.match(line)
        if match and len(match.group(1)) == level:
            flush()
            current_title = match.group(2).strip()
            buffer = [line]
        else:
            buffer.append(line)
    flush()
    return sections or [(first_heading(text), text)]


def load_manifest(path: str) -> Dict[str, Dict[str, Any]]:
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return {item["doc_id"]: item for item in data.get("documents", [])}
        except Exception:
            return {}
    return {}


def save_manifest(path: str, entries: Dict[str, Dict[str, Any]], meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "generated_by": "scripts/build_corpus.py",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "documents": [entries[key] for key in sorted(entries)],
    }
    payload.update(meta)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_doc(path: str, title: str, body: str) -> None:
    """写入语料文件：正文一定带一级标题（loader 用它当 title，chunker 用它当 heading 路径）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    text = body.strip()
    if not text.startswith("#"):
        text = f"# {title}\n\n{text}" if title else text
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build(
    inputs: Sequence[str],
    out_dir: str,
    manifest_path: str,
    prefix: str = "",
    split_level: int = 0,
    min_chars: int = 200,
    max_chars: int = 400000,
    overwrite: bool = False,
    dry_run: bool = False,
    log=print,
) -> Dict[str, Any]:
    files = iter_files(inputs)
    if not files:
        raise SystemExit(f"没有找到可读文件：{list(inputs)}")

    entries = load_manifest(manifest_path)
    existing_doc_ids = set(entries)
    taken_names = {item.get("file", "") for item in entries.values()}
    accepted_tokens: List[set] = []
    seen_hashes: Dict[str, str] = {item.get("sha1", ""): key for key, item in entries.items() if item.get("sha1")}

    stats = {"files": 0, "written": 0, "skipped_empty": 0, "skipped_dup": 0, "skipped_exists": 0, "chars": 0}
    written: List[Dict[str, Any]] = []

    for path in files:
        stats["files"] += 1
        try:
            raw = normalize_ws(load_one(path))
        except Exception as exc:
            log(f"  [跳过] 无法解析 {os.path.basename(path)}：{exc}")
            stats["skipped_empty"] += 1
            continue
        if len(raw) < min_chars:
            stats["skipped_empty"] += 1
            continue

        sections = split_by_heading(raw, split_level)
        multi = len(sections) > 1
        for index, (title, body) in enumerate(sections, start=1):
            body = normalize_ws(body)
            if len(body) > max_chars:
                body = body[:max_chars]
            if len(body) < min_chars and multi:
                stats["skipped_empty"] += 1
                continue

            digest = content_hash(body)
            if digest in seen_hashes:
                stats["skipped_dup"] += 1
                continue
            tokens = set(tokenize(body)[:4000])
            if near_duplicate(tokens, accepted_tokens):
                stats["skipped_dup"] += 1
                continue

            base = slugify(title or first_heading(body), fallback=os.path.splitext(os.path.basename(path))[0])
            if multi:
                base = f"{base}-{index:02d}"
            if prefix:
                base = f"{prefix}-{base}"
            name = f"{base}.md"
            suffix = 2
            while name in taken_names and not overwrite:
                name = f"{base}-{suffix}.md"
                suffix += 1
            doc_id = os.path.splitext(name)[0]
            if doc_id in existing_doc_ids and not overwrite:
                stats["skipped_exists"] += 1
                continue

            target = os.path.join(out_dir, name)
            if not dry_run:
                write_doc(target, title or doc_id, body)
            accepted_tokens.append(tokens)
            seen_hashes[digest] = doc_id
            taken_names.add(name)
            stats["written"] += 1
            stats["chars"] += len(body)
            entry = {
                "doc_id": doc_id,
                "file": name,
                "title": title or doc_id,
                "source": os.path.abspath(path),
                "license": "见原始文件",
                "chars": len(body),
                "sha1": digest,
                "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "tool": "build_corpus",
            }
            entries[doc_id] = entry
            existing_doc_ids.add(doc_id)
            written.append(entry)

    if not dry_run and written:
        save_manifest(
            manifest_path,
            entries,
            {"last_run": {"inputs": [os.path.abspath(p) for p in inputs], "split_level": split_level}},
        )
    return {"stats": stats, "written": written, "manifest": manifest_path, "dry_run": dry_run}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="把文档目录转换成 rag-lab 规范语料")
    parser.add_argument("inputs", nargs="+", help="输入目录或文件（可多个）")
    parser.add_argument("--out", default="data/corpus", help="语料输出目录（默认 data/corpus）")
    parser.add_argument("--manifest", default="data/manifests/corpus.json", help="来源清单落盘位置")
    parser.add_argument("--prefix", default="", help="给生成的 doc_id 加前缀，便于与手写语料区分")
    parser.add_argument("--split-level", type=int, default=0, help="按第 N 级标题拆成多篇（0=不拆）")
    parser.add_argument("--min-chars", type=int, default=200, help="短于这个长度的片段丢弃")
    parser.add_argument("--max-chars", type=int, default=400000, help="单篇上限（超出截断）")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名 doc_id")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        try:
            print(message, flush=True)
        except UnicodeEncodeError:  # pragma: no cover - 极端控制台编码
            print(message.encode("utf-8", "ignore").decode("ascii", "ignore"), flush=True)

    log(f"输入：{len(args.inputs)} 个路径；输出：{args.out}；拆分层级：{args.split_level or '不拆'}")
    result = build(
        inputs=args.inputs,
        out_dir=args.out,
        manifest_path=args.manifest,
        prefix=args.prefix,
        split_level=args.split_level,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        log=log,
    )
    stats = result["stats"]
    log("")
    log("| 指标 | 值 |")
    log("| --- | --- |")
    log(f"| 扫描文件 | {stats['files']} |")
    log(f"| 新写入语料 | {stats['written']} |")
    log(f"| 写入总字数 | {stats['chars']} |")
    log(f"| 跳过：空/过短 | {stats['skipped_empty']} |")
    log(f"| 跳过：重复/近重复 | {stats['skipped_dup']} |")
    log(f"| 跳过：已存在 | {stats['skipped_exists']} |")
    if result["dry_run"]:
        log("\n（--dry-run：没有写任何文件）")
    log(f"\n来源清单：{args.manifest}")
    log("下一步：python -m rag.cli ingest data/corpus")
    return 0


if __name__ == "__main__":
    sys.exit(main())
