"""命令行入口：``python -m rag.cli <命令>``。

    python -m rag.cli ingest data/corpus            # 建索引
    python -m rag.cli ask "混合检索为什么要用 RRF？"   # 单次问答（带引用与耗时）
    python -m rag.cli eval --tag baseline          # 跑评测集，落盘报告 + 追加 ablation.csv
    python -m rag.cli serve --port 8000            # 起 HTTP 服务 + 打开 web/index.html
    python -m rag.cli info                         # 打印生效配置与组件状态

任何命令都支持 ``--set a.b.c=value`` 临时覆盖配置，这是做消融实验的核心开关，
例如 ``--set retrieval.rerank.enabled=true --set retrieval.rewrite.strategy=hyde``。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from .config import load_config, parse_scalar
from .obs import configure_logging
from .pipeline import RAGPipeline

__all__ = ["main", "build_parser"]


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="配置文件路径（默认 configs/default.yaml）")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖配置项，可重复，如 --set retrieval.rerank.enabled=true",
    )
    parser.add_argument("--root", default=None, help="项目根目录（默认为包含 pyproject.toml 的目录）")
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def _overrides(pairs: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set 需要 KEY=VALUE 形式，收到：{pair}")
        key, value = pair.split("=", 1)
        out[key.strip()] = parse_scalar(value)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="rag-lab：领域知识库问答（混合检索 + 重排 + 引用溯源 + 可复现评测）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="加载语料、分块、建向量与 BM25 索引（全量重建）")
    _add_common(p_ingest)
    p_ingest.add_argument("corpus", nargs="*", help="语料文件或目录（默认取 paths.corpus）")

    p_ask = sub.add_parser("ask", help="问一个问题，输出答案、引用与阶段耗时")
    _add_common(p_ask)
    p_ask.add_argument("question", nargs="+", help="问题文本（可带空格，无需引号）")
    p_ask.add_argument("--top-k", type=int, default=None, help="最终送给生成器的片段数")
    p_ask.add_argument("--json", action="store_true", help="输出完整 JSON（含调试信息）")
    p_ask.add_argument("--debug", action="store_true", help="额外打印各通道候选")

    p_eval = sub.add_parser("eval", help="在评测集上跑一遍，输出指标并落盘")
    _add_common(p_eval)
    p_eval.add_argument("--dataset", default=None, help="评测集路径（默认 paths.eval）")
    p_eval.add_argument("--tag", default=None, help="本次实验标签，写入 ablation.csv 的 tag 列")
    p_eval.add_argument("--full", action="store_true", help="打印完整 Markdown 报告")
    p_eval.add_argument("--json", action="store_true", help="打印汇总指标的 JSON")

    p_serve = sub.add_parser("serve", help="启动 HTTP 服务并托管 web/index.html（需手动打开提示的地址）")
    _add_common(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--server", choices=["auto", "simple", "fastapi"], default="auto")

    p_info = sub.add_parser("info", help="打印生效配置与组件状态")
    _add_common(p_info)
    p_info.add_argument("--no-build", action="store_true", help="不构建索引，只看配置")
    p_info.add_argument("--config-dump", action="store_true", help="额外打印完整生效配置")

    return parser


# ---- 各命令实现 -------------------------------------------------------------
def cmd_ingest(args, cfg, log) -> int:
    pipeline = RAGPipeline.build(config=cfg)
    inputs = args.corpus or None
    report = pipeline.ingest(inputs)
    pipeline.ensure_ready()
    print(report.summary())
    status = pipeline.status()
    print(f"向量后端：{status['index']['backend']}（维度 {status['index']['dim']}）")
    print(f"向量模型：{status['components']['embedder']['name']}")
    print(f"BM25：{int(status['bm25'].get('vocabulary', 0))} 个词项，avgdl={status['bm25'].get('avgdl')}")
    return 0


def cmd_ask(args, cfg, log) -> int:
    question = " ".join(args.question).strip()
    pipeline = RAGPipeline.build(config=cfg)
    answer = pipeline.ask(question, top_k=args.top_k)
    if args.json:
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f"问题：{question}")
    print("")
    print("答案：")
    print(answer.text)
    print("")
    if answer.citations:
        print(f"引用（{len(answer.citations)} 条）：")
        for citation in answer.citations:
            print(f"  [{citation.index}] {citation.doc_id} / {citation.chunk_id}  score={citation.score:.4f}")
            print(f"      {citation.quote}")
    else:
        print("引用：无（答案未标注来源，可能是拒答或模型未遵守引用约束）")
    print("")
    latency = answer.latency_ms
    print(
        "耗时：rewrite {:.1f}ms | retrieve {:.1f}ms | rerank {:.1f}ms | generate {:.1f}ms | total {:.1f}ms".format(
            latency.get("rewrite", 0.0),
            latency.get("retrieve", 0.0),
            latency.get("rerank", 0.0),
            latency.get("generate", 0.0),
            latency.get("total", 0.0),
        )
    )
    components = answer.debug.get("components", {})
    print(
        "组件：embedder={} | reranker={} | generator={} | rewrite={}".format(
            components.get("embedder", {}).get("name"),
            components.get("reranker", {}).get("name"),
            components.get("generator", {}).get("name"),
            (answer.debug.get("rewrite") or {}).get("strategy"),
        )
    )
    if args.debug:
        print("")
        print("调试信息：")
        for key in ("queries", "rewrite", "multihop", "generator"):
            print(f"  {key}: {json.dumps(answer.debug.get(key), ensure_ascii=False)}")
        for key in ("vector_top", "bm25_top", "fused_top"):
            print(f"  {key}:")
            for item in (answer.debug.get(key) or [])[:5]:
                print(f"    {item['chunk_id']}  score={item['score']}")
    return 0


def cmd_eval(args, cfg, log) -> int:
    from .eval.run_eval import format_markdown, run

    report, paths = run(cfg, dataset_path=args.dataset, tag=args.tag, log=log)
    if args.json:
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    elif args.full:
        print(format_markdown(report))
    else:
        print(f"评测完成：{report['tag']}（{report['metrics'].get('items')} 题）")
        print("")
        print("| 指标 | 值 |")
        print("| --- | --- |")
        for key, value in report["metrics"].items():
            print(f"| {key} | {value} |")
    print("")
    print(f"报告：{paths['markdown']}")
    print(f"JSON：{paths['json']}")
    print(f"消融表（追加一行）：{paths['csv']}")
    return 0


def cmd_serve(args, cfg, log) -> int:
    from .api import serve as simple_serve

    use_fastapi = args.server == "fastapi"
    if args.server == "auto":
        try:
            import fastapi  # type: ignore  # noqa: F401
            import uvicorn  # type: ignore  # noqa: F401

            use_fastapi = True
        except ImportError:
            use_fastapi = False

    if use_fastapi:
        from .api.main import run as fastapi_run

        fastapi_run(cfg, host=args.host, port=args.port, log=log)
    else:
        simple_serve(cfg, host=args.host, port=args.port, log=log)
    return 0


def cmd_info(args, cfg, log) -> int:
    pipeline = RAGPipeline.build(config=cfg)
    if not args.no_build:
        pipeline.ensure_ready()
    if args.config_dump:
        print(cfg.describe())
        print("")
    print(json.dumps(pipeline.status(), ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = _overrides(args.set)
    if getattr(args, "log_level", None):
        overrides["logging.level"] = args.log_level

    cfg = load_config(path=args.config, overrides=overrides, root=args.root)
    log = configure_logging(str(cfg.get("logging.level", "INFO")))

    handlers = {
        "ingest": cmd_ingest,
        "ask": cmd_ask,
        "eval": cmd_eval,
        "serve": cmd_serve,
        "info": cmd_info,
    }
    handler = handlers[args.command]
    try:
        return int(handler(args, cfg, log) or 0)
    except KeyboardInterrupt:
        log.warning("已中断")
        return 130
    except Exception as exc:
        log.error("%s", exc)
        if str(cfg.get("logging.level", "INFO")).upper() == "DEBUG":
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
