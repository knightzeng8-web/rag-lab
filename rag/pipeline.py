"""流水线编排：把「加载 -> 分块 -> 建索引」和「改写 -> 混合检索 -> 重排 -> 生成」串起来。

``RAGPipeline`` 是 CLI、API、评测脚本共用的唯一入口，责任只有三件事：

1. **索引生命周期**：存在且与当前配置/语料匹配就直接加载，否则重建。索引元数据里记录
   ``embedder.name``、分块参数、**语料来源与内容指纹**，因此「换了向量模型却复用旧索引」
   「语料改了还用老索引」这两类最隐蔽的错误都会被自动发现。
2. **组件装配**：embedder / reranker / LLM / generator 都在这里按配置构建，并自动降级。
3. **一次问答的编排**：翻译成 ``Answer``（答案 + 引用 + 阶段耗时 + 调试信息）。
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Config, load_config
from .device import device_report
from .generate.answer import build_generator, limit_contexts, parse_citations
from .generate.llm import build_llm, llm_status
from .generate.prompt import is_refusal
from .index.bm25 import BM25Index
from .index.embedder import Embedder, build_embedder
from .index.store import BaseVectorStore, build_store, load_store
from .ingest.chunker import Chunker
from .ingest.loader import load_documents
from .obs import get_logger
from .retrieve.hybrid import HybridRetriever
from .retrieve.rerank import build_reranker
from .types import Answer, IngestReport, RetrieveResult

__all__ = ["RAGPipeline"]


class RAGPipeline:
    def __init__(self, cfg: Config, log=None):
        self.cfg = cfg
        self.log = log or get_logger(level=str(cfg.get("logging.level", "INFO")))
        self.index_dir = cfg.path_of("index")
        self.chunker = Chunker.from_config(cfg)
        self.embedder: Optional[Embedder] = None
        self.store: Optional[BaseVectorStore] = None
        self.bm25: Optional[BM25Index] = None
        self.retriever: Optional[HybridRetriever] = None
        self.generator = None
        self.reranker = None
        self.llm = None
        self.meta: Dict[str, Any] = {}
        self._built = False
        self._corpus_signature: Optional[List[str]] = None
        self._corpus_signature_key: Optional[Tuple[str, ...]] = None

    # ---- 构建 ---------------------------------------------------------------
    @classmethod
    def build(
        cls,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        root: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> "RAGPipeline":
        cfg = config or load_config(path=config_path, overrides=overrides, root=root)
        return cls(cfg)

    # ---- 语料指纹 -----------------------------------------------------------
    def corpus_inputs(self) -> List[str]:
        """当前索引所依据的语料来源：优先用索引里记录过的，其次用配置里的 paths.corpus。"""
        recorded = self.meta.get("corpus_inputs")
        if recorded:
            return [str(p) for p in recorded]
        return [self.cfg.path_of("corpus")]

    def corpus_signature(
        self,
        inputs: Optional[Sequence[str]] = None,
        refresh: bool = False,
    ) -> List[str]:
        """语料指纹：``绝对路径:内容哈希`` 排序后的列表，用于判断索引是否过期。"""
        if inputs is None:
            inputs = self.corpus_inputs()
        key = tuple(str(p) for p in inputs)
        if not refresh and self._corpus_signature is not None and self._corpus_signature_key == key:
            return self._corpus_signature
        docs = load_documents([str(p) for p in inputs], log=None)
        signature = sorted(
            f"{d.path or d.doc_id}:{d.meta.get('fingerprint', '')}" for d in docs
        )
        self._corpus_signature = signature
        self._corpus_signature_key = key
        return signature

    def _rebuild_inputs(self) -> Optional[List[str]]:
        """重建索引时应当使用的语料来源（记录过且仍存在，否则回落到配置默认）。"""
        recorded = [str(p) for p in (self.meta.get("corpus_inputs") or [])]
        if recorded:
            if all(os.path.exists(p) for p in recorded):
                return recorded
            self.log.warning(
                "索引记录的语料来源已不可用（%s），改用配置中的 paths.corpus 重建", recorded
            )
        return None

    # ---- 索引 ---------------------------------------------------------------
    def _meta_path(self) -> str:
        return os.path.join(self.index_dir, "meta.json")

    def _index_exists(self) -> bool:
        return all(
            os.path.isfile(os.path.join(self.index_dir, name))
            for name in ("meta.json", "vectors.npy", "chunks.jsonl", "bm25.json")
        )

    def _index_matches(self) -> bool:
        """检查索引是否与当前配置/语料一致；不一致就必须重建。"""
        if not self.meta:
            return False
        stored_embedder = (self.meta.get("embedder") or {}).get("name")
        if stored_embedder and self.embedder and stored_embedder != self.embedder.name:
            self.log.warning(
                "索引使用的向量模型 %s 与当前配置 %s 不一致，将重建索引",
                stored_embedder,
                self.embedder.name,
            )
            return False
        stored_chunking = self.meta.get("chunking") or {}
        current_chunking = {
            "strategy": self.chunker.strategy,
            "max_chars": self.chunker.max_chars,
            "overlap_chars": self.chunker.overlap_chars,
        }
        for key, value in current_chunking.items():
            if key in stored_chunking and stored_chunking[key] != value:
                self.log.warning("分块参数 %s 变化（%s -> %s），将重建索引", key, stored_chunking[key], value)
                return False
        # 向量后端也算索引身份的一部分：显式指定 numpy/faiss 时要与磁盘上的一致，
        # 否则 --set index.backend=faiss 会静默复用旧后端（对比实验就变成了空跑）。
        # 注意 `auto` 不参与判断：它表示"用现有索引，没有才新建"。
        configured_backend = str(self.cfg.get("index.backend", "auto")).lower()
        stored_backend = str(self.meta.get("backend", "")).lower()
        if configured_backend in ("numpy", "faiss") and stored_backend and stored_backend != configured_backend:
            self.log.warning(
                "索引后端不一致（磁盘 %s / 配置 %s），将重建索引", stored_backend, configured_backend
            )
            return False
        if configured_backend == "auto" and stored_backend:
            self.log.info("索引后端 %s（index.backend=auto 不做迁移）", stored_backend)
        recorded_inputs = [str(p) for p in (self.meta.get("corpus_inputs") or [])]
        if recorded_inputs and recorded_inputs != [self.cfg.path_of("corpus")]:
            self.log.info(
                "当前索引来自 %s（配置里的 paths.corpus 是 %s）；如需切换语料请重新运行 ingest",
                recorded_inputs,
                self.cfg.path_of("corpus"),
            )
        if bool(self.cfg.get("index.auto_rebuild", True)):
            stored_sig = self.meta.get("corpus_signature")
            if stored_sig is not None and list(stored_sig) != self.corpus_signature():
                self.log.warning("语料内容发生变化，将重建索引")
                return False
        return True

    def _load_index(self) -> None:
        self.store = load_store(self.index_dir, log=self.log)
        self.meta = dict(getattr(self.store, "meta", {}) or {})
        self.bm25 = BM25Index.load(self.index_dir)
        self.log.info(
            "已加载索引：%d 个分块，后端 %s，维度 %s",
            len(self.store),
            self.store.backend,
            self.store.dim,
        )

    def ingest(self, inputs: Optional[Sequence[str]] = None) -> IngestReport:
        """全量重建索引（本项目不做增量索引；元数据里已记录 fingerprint，接口留给进阶项）。"""
        started = time.perf_counter()
        if inputs is None:
            inputs = [self.cfg.path_of("corpus")]
        inputs = [str(p) for p in inputs]
        docs = load_documents(inputs, log=self.log)
        if not docs:
            raise RuntimeError(f"没有加载到任何文档，请检查语料路径：{inputs}")
        chunks, chunk_stats = self.chunker.chunk_documents(docs)
        if not chunks:
            raise RuntimeError("分块结果为空，请检查 chunking 配置")
        self.log.info(
            "分块完成：策略 %s，平均 %.0f 字/块", chunk_stats.strategy, chunk_stats.avg_chars
        )

        if self.embedder is None:
            self.embedder = build_embedder(self.cfg, log=self.log)
        texts = [c.text for c in chunks]
        vectors = self.embedder.encode(texts)
        if vectors.size == 0:
            raise RuntimeError("向量化结果为空")

        store = build_store(self.cfg, dim=int(vectors.shape[1]), log=self.log)
        store.add(chunks, vectors)
        corpus_inputs = [os.path.abspath(p) for p in inputs]
        meta_extra = {
            "embedder": self.embedder.status(),
            "chunking": {
                "strategy": self.chunker.strategy,
                "max_chars": self.chunker.max_chars,
                "overlap_chars": self.chunker.overlap_chars,
                "min_chars": self.chunker.min_chars,
            },
            "docs": len(docs),
            "doc_ids": [d.doc_id for d in docs],
            "corpus_inputs": corpus_inputs,
            "corpus_signature": self.corpus_signature(inputs=corpus_inputs, refresh=True),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        store.save(self.index_dir, extra=meta_extra)
        self.meta = dict(meta_extra)
        self.meta.update({"dim": store.dim, "backend": store.backend, "count": len(store)})

        bm25 = BM25Index.from_config(self.cfg).fit([c.chunk_id for c in chunks], texts)
        bm25.save(self.index_dir)

        self.store = store
        self.bm25 = bm25
        # 索引换了，已装配的 retriever 还持有旧的 store/chunk_map，必须让它重新装配
        self._built = False
        elapsed = (time.perf_counter() - started) * 1000.0
        report = IngestReport(
            docs=len(docs),
            chunks=len(chunks),
            index_dir=self.index_dir,
            elapsed_ms=elapsed,
            files=[os.path.basename(d.path) for d in docs],
        )
        self.log.info(report.summary())
        return report

    def ensure_ready(self, force_rebuild: bool = False) -> None:
        """构建/加载索引并装配所有组件；幂等，重复调用无额外开销。"""
        if self._built and not force_rebuild:
            return
        if self.embedder is None:
            self.embedder = build_embedder(self.cfg, log=self.log)
        if self.llm is None:
            self.llm = build_llm(self.cfg, log=self.log)

        needs_build = force_rebuild or not self._index_exists()
        if not needs_build:
            try:
                self._load_index()
                if not self._index_matches():
                    needs_build = True
            except Exception as exc:
                self.log.warning("索引加载失败（%s），将重建", exc)
                needs_build = True
        if needs_build:
            self.ingest(inputs=self._rebuild_inputs())

        # 词法重排需要 IDF 加权（否则中文常见字词会淹没真正的相关词），把 BM25 的 idf 注入进去
        self.reranker = build_reranker(
            self.cfg,
            llm=self.llm,
            log=self.log,
            idf=(self.bm25.idf if self.bm25 is not None else None),
        )
        self.generator = build_generator(self.cfg, llm=self.llm, log=self.log)
        self.retriever = HybridRetriever(
            self.cfg,
            store=self.store,
            bm25=self.bm25,
            embedder=self.embedder,
            reranker=self.reranker,
            llm=self.llm,
            log=self.log,
        )
        self._built = True

    # ---- 问答 ---------------------------------------------------------------
    def retrieve(self, question: str, top_k: Optional[int] = None) -> RetrieveResult:
        self.ensure_ready()
        assert self.retriever is not None
        return self.retriever.retrieve(question, top_k=top_k)

    def ask(self, question: str, top_k: Optional[int] = None) -> Answer:
        question = (question or "").strip()
        started = time.perf_counter()
        if not question:
            return Answer(question=question, text="请输入问题。", refused=True, latency_ms={})

        result = self.retrieve(question, top_k=top_k)
        contexts = limit_contexts(result.ranked, self.cfg)

        generate_started = time.perf_counter()
        text, generate_debug = self.generator.generate(question, contexts)
        generate_ms = (time.perf_counter() - generate_started) * 1000.0

        citations = parse_citations(
            text, contexts, quote_chars=int(self.cfg.get("generation.quote_chars", 240))
        )
        refused = is_refusal(text) or generate_debug.get("mode") == "no_context"

        rewrite_ms = float(result.latency_ms.get("rewrite", 0.0))
        retrieve_ms = max(0.0, float(result.latency_ms.get("retrieve", 0.0)) - rewrite_ms)
        rerank_ms = float(result.latency_ms.get("rerank", 0.0))
        total_ms = (time.perf_counter() - started) * 1000.0

        status = self.status(include_index=False)
        debug: Dict[str, Any] = {
            "queries": result.queries,
            "components": status.get("components", {}),
            "vector_top": [s.to_dict(with_text=False) for s in result.vector_hits[:10]],
            "bm25_top": [s.to_dict(with_text=False) for s in result.bm25_hits[:10]],
            "fused_top": [s.to_dict(with_text=False) for s in result.fused[:10]],
            # 未截断的排序结果，供评测脚本计算 Recall@k / MRR / nDCG@k
            "ranked": [s.to_dict(with_text=False) for s in result.ranked],
            "ranked_docs": [s.doc_id for s in result.ranked],
            "rewrite": result.debug.get("rewrite"),
            "multihop": result.debug.get("multihop"),
            "rrf_k": result.debug.get("rrf_k"),
            "weights": result.debug.get("weights"),
            "reranker": result.debug.get("reranker"),
            "generator": generate_debug,
            "citation_coverage": round(len(citations) / max(1, len(contexts)), 4),
            "contexts": [s.to_dict(with_text=False) for s in contexts],
        }
        return Answer(
            question=question,
            text=text,
            citations=citations,
            refused=bool(refused),
            contexts=contexts,
            latency_ms={
                "rewrite": rewrite_ms,
                "retrieve": retrieve_ms,
                "rerank": rerank_ms,
                "generate": generate_ms,
                "total": total_ms,
            },
            debug=debug,
        )

    # ---- 状态 ---------------------------------------------------------------
    def status(self, include_index: bool = True) -> Dict[str, Any]:
        components = {
            "embedder": (self.embedder.status() if self.embedder else {"name": "unloaded"}),
            "reranker": (self.reranker.status() if self.reranker else {"name": "unloaded"}),
            "llm": llm_status(self.llm),
            "generator": (self.generator.status() if self.generator else {"name": "unloaded"}),
            "chunking": {
                "strategy": self.chunker.strategy,
                "max_chars": self.chunker.max_chars,
                "overlap_chars": self.chunker.overlap_chars,
            },
            "rewrite": {
                "enabled": bool(self.cfg.get("retrieval.rewrite.enabled", True)),
                "strategy": str(self.cfg.get("retrieval.rewrite.strategy", "auto")),
            },
            "multihop": {
                "enabled": bool(self.cfg.get("multihop.enabled", False)),
                "max_rounds": int(self.cfg.get("multihop.max_rounds", 2)),
            },
            "fusion": {
                "rrf_k": int(self.cfg.get("retrieval.rrf_k", 60)),
                "weights": self.cfg.section("retrieval").get("weights", {}),
            },
        }
        payload: Dict[str, Any] = {
            "project": str(self.cfg.get("project", "rag-lab")),
            "config_path": self.cfg.path,
            "root": self.cfg.root,
            # 现场可验证的"跑在哪块设备上"：没装 torch 时也会安全返回 cpu
            "device": device_report(),
            "components": components,
        }
        if include_index:
            payload["index"] = {
                "dir": self.index_dir,
                "exists": self._index_exists(),
                "chunks": len(self.store) if self.store else 0,
                "docs": self.meta.get("docs", 0),
                "backend": (self.store.backend if self.store else None),
                "dim": (self.store.dim if self.store else None),
                "built_at": self.meta.get("built_at"),
                "doc_ids": self.meta.get("doc_ids", []),
                "corpus_inputs": self.meta.get("corpus_inputs", []),
            }
            payload["bm25"] = self.bm25.stats() if self.bm25 else {}
        return payload

    def export_state(self) -> str:
        return json.dumps(self.status(), ensure_ascii=False, indent=2)
