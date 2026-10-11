"""从公开来源抓取语料，扩充本土语料库（带来源与许可登记）。

为什么要专门写一个抓取器：`build_corpus.py` 解决"我手上已有的文件"，但项目起步时往往
**没有足够的语料**——5 篇手写笔记会把所有指标顶到天花板（实测 nDCG@10 = 1.0000），
根本测不出组件差异。这个脚本用来快速拿到一批**许可清晰、可追溯出处**的真实文档。

支持两个来源：
- `wikipedia`（默认，中文）：走 MediaWiki API 的 `list=search` + `prop=extracts`，
  许可 CC BY-SA 4.0；`explaintext` 返回的是 `== 小节 ==` 形式，脚本会转换成 Markdown 标题，
  这样 `chunker=strategy: structural` 才能按小节切分。
- `arxiv`：走公开 Atom API 取标题/摘要/作者/分类；`--fulltext` 时再下 PDF 并用 pypdf 抽正文
  （arXiv 上多数论文为 arXiv 非独占许可 / CC-BY，用于个人研究是可接受的，但**请勿再分发**）。

礼貌与稳健：自定义 User-Agent、请求之间 `--delay` 间隔、失败重试一次、单篇上限截断。
所有来源、许可、URL、抓取时间都写进 `data/manifests/corpus.json`（不写进正文，避免污染检索）。

用法：
    python scripts/fetch_public_corpus.py wikipedia --queries 检索增强生成 向量数据库 … --limit 40
    python scripts/fetch_public_corpus.py arxiv --query 'abs:"retrieval augmented generation"' --limit 25
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_corpus import content_hash, load_manifest, save_manifest, slugify, write_doc  # noqa: E402

USER_AGENT = "rag-lab/0.1 (personal study project; local use; contact: local user)"
WIKI_API = "https://zh.wikipedia.org/w/api.php"
ARXIV_API = "http://export.arxiv.org/api/query"
WIKI_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


# --------------------------------------------------------------------------- #
def http_get(url: str, timeout: int = 20, retries: int = 1) -> bytes:
    last: Optional[Exception] = None
    for _ in range(retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (403, 429):  # 被限流时稍等再试
                time.sleep(2.0)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0)
    raise RuntimeError(f"请求失败：{url}（{last}）")


def wiki_headings_to_markdown(text: str) -> str:
    """把 MediaWiki 纯文本里的 `== 小节 ==` 转成 Markdown 标题，供结构感知分块使用。"""
    def repl(match: re.Match) -> str:
        level = max(2, min(6, len(match.group(1))))
        return "#" * level + " " + match.group(2).strip()

    return WIKI_HEADING_RE.sub(repl, text)


def wiki_search(query: str, limit: int) -> List[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "format": "json",
        "formatversion": "2",
    }
    data = json.loads(http_get(f"{WIKI_API}?{urllib.parse.urlencode(params)}").decode("utf-8"))
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def wiki_extracts(titles: Sequence[str], delay: float = 0.8) -> List[Dict[str, Any]]:
    """逐条取整篇正文。

    这里有个必须踩过才知道的限制：**整篇正文提取（`explaintext=1`，不带 `exintro`）
    每次请求只能要 1 个标题**——多标题时 API 会把 `exlimit` 强制降到 1 并只返回第一页，
    表现为"请求 20 条只取回 1 篇"。所以这里改成一次一个标题，用间隔换稳定；
    如果想要批量，只能取"仅导言"（`exintro=1`），但那样每篇只有 1~3KB。

    另外 MediaWiki 会对高频请求返回 429，所以每次请求之间要 sleep，并发失败要退避重试。
    """
    out: List[Dict[str, Any]] = []
    total = len(titles)
    for index, title in enumerate(titles, start=1):
        params = {
            "action": "query",
            "prop": "extracts|info",
            "explaintext": "1",
            "exlimit": "1",
            "redirects": "1",
            "inprop": "url",
            "format": "json",
            "formatversion": "2",
            "titles": title,
        }
        try:
            data = json.loads(http_get(f"{WIKI_API}?{urllib.parse.urlencode(params)}", retries=2).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{index}/{total}] 「{title}」失败：{exc}")
            time.sleep(delay * 2)
            continue
        if data.get("warnings"):
            print(f"  [API 警告] {str(data['warnings'])[:160]}")
        pages = data.get("query", {}).get("pages", [])
        for page in pages:
            if page.get("missing") or not page.get("extract"):
                print(f"  [{index}/{total}] 「{title}」跳过（页面不存在或无正文）")
                continue
            text = page.get("extract", "")
            print(f"  [{index}/{total}] 「{page.get('title')}」{len(text)} 字")
            out.append(
                {
                    "title": page.get("title", ""),
                    "text": text,
                    "url": page.get("fullurl", ""),
                    "revision": page.get("lastrevid"),
                }
            )
        time.sleep(delay)
    return out


def arxiv_search(query: str, limit: int, fulltext: bool, max_chars: int) -> List[Dict[str, Any]]:
    params = {
        "search_query": query,
        "start": "0",
        "max_results": str(limit),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    raw = http_get(f"{ARXIV_API}?{urllib.parse.urlencode(params)}")
    root = ET.fromstring(raw)
    out: List[Dict[str, Any]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = " ".join((entry.findtext("atom:title", "", ATOM_NS) or "").split())
        summary = (entry.findtext("atom:summary", "", ATOM_NS) or "").strip()
        url = entry.findtext("atom:id", "", ATOM_NS) or ""
        published = entry.findtext("atom:published", "", ATOM_NS) or ""
        category = entry.findtext("arxiv:primary_category", "", ATOM_NS)
        if category is None:
            node = entry.find("arxiv:primary_category", ATOM_NS)
            category = node.get("term") if node is not None else ""
        authors = [
            (a.findtext("atom:name", "", ATOM_NS) or "").strip()
            for a in entry.findall("atom:author", ATOM_NS)
        ]
        body = f"## 摘要\n\n{summary}"
        if fulltext:
            pdf_url = ""
            for link in entry.findall("atom:link", ATOM_NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
            if pdf_url:
                try:
                    text = pdf_to_text(pdf_url)
                    if len(text) > 800:
                        body = f"## 正文\n\n{text[:max_chars]}"
                except Exception as exc:  # noqa: BLE001
                    print(f"    [警告] PDF 解析失败（{exc}），退回摘要")
        out.append(
            {
                "title": title,
                "text": body,
                "url": url,
                "license": "arXiv（非独占许可/CC 系列，个人研究用途）",
                "meta": {"authors": authors, "published": published, "category": category},
            }
        )
    return out


def pdf_to_text(url: str) -> str:
    """下载 PDF 并用 pypdf 抽正文（pypdf 是可选的，没装就抛错由上层退回摘要）。"""
    raw = http_get(url, timeout=60)
    import io
    import tempfile  # noqa: F401  # 仅用于类型提示，不创建文件

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


# --------------------------------------------------------------------------- #
def store(
    docs: Sequence[Dict[str, Any]],
    out_dir: str,
    manifest_path: str,
    prefix: str,
    max_chars: int,
    dry_run: bool,
    log,
) -> Dict[str, int]:
    entries = load_manifest(manifest_path)
    taken = {item.get("file", "") for item in entries.values()}
    seen = {item.get("sha1", "") for item in entries.values()}
    stats = {"written": 0, "skipped_dup": 0, "skipped_short": 0, "chars": 0}

    for doc in docs:
        text = wiki_headings_to_markdown(doc["text"]) if doc.get("wiki") else doc["text"]
        body = text.strip()[:max_chars]
        if len(body) < 200:
            stats["skipped_short"] += 1
            continue
        digest = content_hash(body)
        if digest in seen:
            stats["skipped_dup"] += 1
            continue

        base = slugify(doc["title"])
        if prefix:
            base = f"{prefix}-{base}"
        name = f"{base}.md"
        suffix = 2
        while name in taken:
            name = f"{base}-{suffix}.md"
            suffix += 1
        doc_id = os.path.splitext(name)[0]

        if not dry_run:
            write_doc(os.path.join(out_dir, name), doc["title"], body)
        seen.add(digest)
        taken.add(name)
        stats["written"] += 1
        stats["chars"] += len(body)
        entries[doc_id] = {
            "doc_id": doc_id,
            "file": name,
            "title": doc["title"],
            "source": doc.get("url", ""),
            "license": doc.get("license", "见来源页面"),
            "chars": len(body),
            "sha1": digest,
            "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tool": "fetch_public_corpus",
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "extra": doc.get("meta", {}),
        }
        log(f"  + {name}（{len(body)} 字）")

    if not dry_run and stats["written"]:
        save_manifest(manifest_path, entries, {"last_fetch": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="从公开来源抓取语料（维基百科 / arXiv）")
    sub = parser.add_subparsers(dest="provider", required=True)

    p_wiki = sub.add_parser("wikipedia", help="中文维基百科（CC BY-SA 4.0）")
    p_wiki.add_argument("--queries", nargs="*", default=[], help="主题词，逐个搜索取前 N 条")
    p_wiki.add_argument(
        "--titles",
        nargs="*",
        default=[],
        help="直接指定条目标题（跳过搜索，规避 API 限流；标题必须真实存在）",
    )
    p_wiki.add_argument("--per-query", type=int, default=2, help="每个主题取几篇（默认 2）")
    p_wiki.add_argument("--delay", type=float, default=0.8, help="请求间隔秒数（默认 0.8，防 429）")

    p_arxiv = sub.add_parser("arxiv", help="arXiv 论文（摘要或全文）")
    p_arxiv.add_argument("--query", required=True, help="arXiv 查询表达式，如 abs:\"retrieval augmented generation\"")
    p_arxiv.add_argument("--fulltext", action="store_true", help="下载 PDF 并抽取正文（更慢更大）")

    for sub_parser in (p_wiki, p_arxiv):
        sub_parser.add_argument("--limit", type=int, default=40, help="最终写入的文档数上限")
        sub_parser.add_argument("--out", default="data/corpus")
        sub_parser.add_argument("--manifest", default="data/manifests/corpus.json")
        sub_parser.add_argument("--prefix", default="", help="给 doc_id 加前缀（如 wiki / arxiv）")
        sub_parser.add_argument("--max-chars", type=int, default=60000, help="单篇上限（超出截断）")
        sub_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    docs: List[Dict[str, Any]] = []
    if args.provider == "wikipedia":
        titles: List[str] = [t for t in (args.titles or []) if t]
        if titles:
            log(f"维基百科：直接用 {len(titles)} 个给定标题（跳过搜索）")
        else:
            log(f"维基百科：{len(args.queries)} 个主题，每主题取 {args.per_query} 篇（间隔 {args.delay}s）")
        for query in args.queries:
            if len(titles) >= args.limit:
                break
            try:
                found = wiki_search(query, args.per_query)
            except Exception as exc:  # noqa: BLE001
                log(f"  [警告] 搜索「{query}」失败：{exc}")
                continue
            for title in found:
                if title not in titles:
                    titles.append(title)
            log(f"  「{query}」-> {found}")
            time.sleep(args.delay)
        if not titles:
            raise SystemExit("没有拿到任何标题：请检查 --queries / --titles 参数")
        titles = titles[: args.limit]
        log(f"共 {len(titles)} 个条目，开始抓正文…")
        for item in wiki_extracts(titles, delay=args.delay):
            item["wiki"] = True
            item["license"] = "CC BY-SA 4.0（维基百科，需保留署名与相同方式共享）"
            docs.append(item)
    else:
        log(f"arXiv 查询：{args.query}（fulltext={args.fulltext}）")
        docs = arxiv_search(args.query, args.limit, args.fulltext, args.max_chars)

    log(f"\n写入 {args.out}（dry-run={args.dry_run}）")
    stats = store(docs, args.out, args.manifest, args.prefix, args.max_chars, args.dry_run, log)
    log("")
    log("| 指标 | 值 |")
    log("| --- | --- |")
    log(f"| 抓到文档 | {len(docs)} |")
    log(f"| 写入语料 | {stats['written']} |")
    log(f"| 写入总字数 | {stats['chars']} |")
    log(f"| 跳过：重复 | {stats['skipped_dup']} |")
    log(f"| 跳过：过短 | {stats['skipped_short']} |")
    log(f"\n来源与许可清单：{args.manifest}")
    log("下一步：python -m rag.cli ingest data/corpus")
    return 0


if __name__ == "__main__":
    sys.exit(main())
