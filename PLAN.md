# 面向领域知识库的多跳问答与证据溯源系统 —— 复杂 RAG 项目方案

> **一句话**：这不是"LangChain + Chroma 教程复刻"，而是一个**零额外依赖可离线跑通、可按开关升级到工业级组件、自带评测集与消融实验**的多跳 RAG（Retrieval-Augmented Generation，检索增强生成）系统。

| 项目代号 | `rag-lab` |
| --- | --- |
| 一句话定位 | 面向领域知识库的**多跳问答（Multi-hop QA）+ 证据溯源（Evidence Attribution）**系统 |
| 示例语料 | 5 篇自写 RAG 技术笔记（`data/corpus/*.md`），可整体替换为某领域论文 / 教材 / 制度文件 |
| 复试时间点 | 2028 年 3 月初（面试） |
| 准备周期 | 自现在起约 3 个月，共 12 周 |
| 默认运行条件 | 仅 `numpy` + Python 标准库，**离线、无网络、无 GPU、无 API Key 也能跑通全链路** |
| 升级路径 | BGE（BAAI General Embedding）向量 / FAISS / Cross-Encoder 重排 / OpenAI 兼容 LLM，配置开关即插即用 |
| 唯一入口命令 | `python -m rag.cli ingest \| ask \| eval \| serve \| info` |
| 代码规模预估 | `rag/` 主包约 2500–3500 行 Python，`tests/` 约 400 行，单文件前端 1 个 |
| 评测集规模 | 60–120 条问答（含 20% 不可回答样本 + 困难负例） |

---

## 1. 一句话定位与复试目标（这个项目要证明什么能力）

**一句话定位**：我实现了一个面向领域知识库的多跳问答与证据溯源系统，在**零额外依赖**条件下可完整跑通索引、改写、混合检索、融合、重排、引用式生成与离线评测，并且用**自建评测集 + 消融实验**证明"每一层复杂度都是花钱买来的、买得值"。

### 1.1 这个项目要证明的 5 项能力

| # | 能力 | 具体证据（面试时可当场打开的东西） |
| --- | --- | --- |
| C1 | **能独立设计系统，而不是拼装框架** | `rag/` 主包全部自写：自写 BM25、自写 RRF（Reciprocal Rank Fusion，倒数排名融合）、自写 chunker、自写评测器；`pipeline.py` 用显式阶段列表描述链路 |
| C2 | **懂检索的数学原理** | 能当场写出 BM25 打分公式、RRF 公式、nDCG（Normalized Discounted Cumulative Gain，归一化折损累计增益）公式，并解释 `k1`、`b`、`k=60` 各自的物理含义与调参依据 |
| C3 | **有评测工程素养** | `data/eval/qa.jsonl` 人工标注 gold chunk；`python -m rag.cli eval` 一次输出 Recall@k / MRR / nDCG@k / 引用精确率 / 拒答准确率 / P50 / P95 / token 成本，并落盘 CSV 供作图 |
| C4 | **有工程降级意识** | `sentence-transformers` 缺失 → 哈希向量；`faiss` 缺失 → 内存内积检索；无 API Key → 抽取式答案；`/health` 报告当前**实际生效**的组件组合 |
| C5 | **能讲清取舍与代价** | 每个设计点都写出"为什么这样选、代价是什么、什么条件下该换"，见第 4、7、8 章的"何时该换"列 |

### 1.2 复试目标（面试官在 15 分钟里会形成的判断）

| 面试官关心的问题 | 本方案给的答案落点 |
| --- | --- |
| 你是调库还是自己写的？ | 第 4 章"为什么不用 LangChain 一把梭"逐条答辩话术；自写 BM25 / RRF / chunker / 评测器 |
| 你怎么知道它有效？ | 第 9 章消融实验矩阵，6 行 × 6 列可直接填数的表 |
| 你的创新点在哪？ | 结构感知分块 + 加权 RRF + 词法覆盖重排的零依赖组合；多跳迭代扩展 + 证据聚合 |
| 你能落地上线吗？ | 第 10 章 Docker、缓存、trace 延迟分解、成本控制；`/health` 组件自述 |
| 你踩过什么坑？ | 第 9.4 节错误归因分类 + 第 14 章风险表，都是有具体数字的实战问题 |

### 1.3 明确不做的范围（Write-down，防止范围爆炸）

| 不做 | 理由 |
| --- | --- |
| 不做模型微调（fine-tuning） | 3 个月内 ROI（Return on Investment，投入产出比）低于检索与评测工程；见第 13.4 节 Q1 答辩话术 |
| 不做分布式 / 千万级向量 | 语料 5 篇→几百篇规模，FAISS `IndexFlatIP` 足够；分布式是另一个话题，答辩可作"如果扩展我会怎么做" |
| 不做多模态 | 列为第 16 章加分项，投入约 20 小时，不作为主线 |
| 不用 LangChain / LlamaIndex 主线实现 | 见第 4 章；但会读其源码，答辩时能对比 |

---

## 2. "复杂"体现在哪里

### 2.1 能力矩阵表：Naive RAG → Advanced RAG → Agentic RAG

| 能力维度 | Naive RAG（教程级） | Advanced RAG（本项目主线） | Agentic RAG（延伸目标） | 本项目落点 |
| --- | --- | --- | --- | --- |
| 索引 | 固定长度切分，单路向量 | 结构感知分块 + 元数据 + 增量更新 | 自动决定索引策略、按需建子索引 | 结构感知 + 滑动窗口重叠，支持重建与增量 |
| 查询理解 | 原样送检 | 改写 / 扩展 / 多查询 / HyDE | 分解子问题、规划检索步数 | 4 策略枚举，配置切换，默认关键词扩展 |
| 召回 | 单路稠密向量 | 稠密 ∥ 稀疏（BM25）双路 | 多工具、多库、多模态 | 向量 ∥ BM25，可加权重 |
| 融合 | 无（或简单相加） | RRF（k=60，可加权） | 学习式融合（LTR） | 加权 RRF，权重写入配置 |
| 排序 | 无 | Cross-Encoder / 词法覆盖重排 | LLM listwise + 反馈闭环 | 4 档重排，默认词法覆盖（零依赖） |
| 多跳 | 无 | 迭代式 query expansion + 证据聚合 | 自主 Agent 规划循环 | `max_hops=2`，证据池去重聚合 |
| 生成 | 直接拼接 + LLM 自由发挥 | 强制引用 `[n]` + 拒答 + 上下文压缩 | 自评、重试、反思 | 引用标记 + 资料不足拒答 |
| 评测 | 肉眼看几条 | 指标化 + 消融 + 错误归因 | 在线 A/B、反馈闭环 | 6 行消融表 + 5 类错误归因 |
| 可观测 | 无 | trace 延迟分解 + `/health` 组件自述 | 全链路 trace + 告警 | 每阶段耗时落盘 JSON |

**本项目的自我定位**：主线做满 **Advanced RAG 全部 8 个维度**，并用 1–2 个点（多跳、组件自述式降级）向 Agentic 过渡，但不引入不可控的自主循环。

### 2.2 本项目的 6 个复杂度来源（逐条说明，含具体数字）

| # | 复杂度来源 | 具体表现（不是口号，是可以指着代码说的） | 代价 |
| --- | --- | --- | --- |
| S1 | **混合检索 + 融合** | 同一 query 跑两条召回路径：向量 top-20 ∥ BM25 top-20，用 RRF 融合成统一排名。难点：两路分数**量纲不同**（余弦相似度 ∈ [-1,1] 与 BM25 无上界），不能直接相加，必须用**排名**而非分数融合 | 多一次 BM25 全文扫描（几百 chunk 时 < 5 ms）；多一份稀疏索引存储 |
| S2 | **查询改写** | 4 种策略：`identity` / 关键词扩展 / 多查询（multi-query）/ HyDE（Hypothetical Document Embeddings，假设文档嵌入）。难点：改写会**引入漂移**，把原问题的意图改丢；多查询会把延迟乘以 N | 多查询 N=3 时召回耗时 ×3，P95 随之上升；HyDE 在无 LLM 时降级为 identity |
| S3 | **结构感知分块** | 按 Markdown 标题层级切分段落，再对超长段落套滑动窗口（`chunk_size=400` 字符，`overlap=80`）。难点：标题路径必须**继承并写入元数据**，否则表格、代码块被腰斩后语义丢失 | chunk 数比固定窗口多 15%–30%，索引变大；边界处理代码约 150 行 |
| S4 | **多跳检索** | 第 1 跳结果作为上下文，抽取关键词生成第 2 跳查询，两跳证据合并去重后统一重排。难点：**误差累积**——第 1 跳错了，第 2 跳会沿着错误方向跑偏；必须有证据池与早停条件 | 延迟约 ×1.8–2.2；可能引入噪声文档，需要重排兜底 |
| S5 | **证据溯源（引用）** | 生成时强制 `[n]` 标记，答案对象里返回 `citations[]`，每项含 `doc_id / chunk_id / quote / score`。难点：**引用正确 ≠ 答案正确**，需要独立的"引用精确率"指标，且要人工校验 quote 是否真的支撑该句 | 人工校验 60–120 条答案约需 4–6 小时；抽取式兜底时引用质量天然更高（句子直接来自原文） |
| S6 | **可评测 + 可降级** | 同一份代码在 4 种环境组合下都要能跑：无任何依赖 / 只有 numpy / 有向量模型 / 有 LLM + 重排。`/health` 与日志必须如实报告当前生效组件 | 需要为每个重依赖写一条等价的离线实现（哈希向量、词法覆盖重排、抽取式答案），约多写 400 行代码 |

**为什么这 6 条能构成"复杂"**：任意一条单独拿出来都是教程里的一节，但**六条同时成立且互相约束**时，才出现真正的设计问题——例如 S2 的多查询会放大 S5 的引用噪声，S4 的多跳会放大 S1 融合的排序误差，S6 的降级约束会否决掉"直接调库"的所有捷径。

---

## 3. 系统架构与数据流

### 3.1 总体架构（两个泳道：离线索引 / 在线问答）

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  泳道 A：离线索引链路（ingest，一次性 / 增量）                                  │
│                                                                              │
│  data/corpus/*.md                                                            │
│      │                                                                       │
│      ▼                                                                       │
│  [loader.py]  读取 Markdown → 抽标题层级 → 清洗(去噪/统一空白/保表格)          │
│      │        产物: Document(doc_id=文件名主干, title, text, meta)             │
│      ▼                                                                       │
│  [chunker.py] 结构感知切分(Markdown 标题+段落) ──► 超长段落滑动窗口(400/80)     │
│      │        产物: Chunk(chunk_id="<doc_id>#<序号>", doc_id, text, meta)      │
│      ▼                                                                       │
│      ├───────────────────────────┬────────────────────────────┐              │
│      ▼                           ▼                            ▼              │
│  [embedder.py]              [bm25.py]                   [store.py]           │
│  哈希向量(默认)            自写 BM25 统计             统一存储 JSON/内存       │
│  或 BGE 向量(可选)         (df/avgdl/postings)        + 向量矩阵 .npy          │
│      │                           │                            │              │
│      └───────────┬───────────────┘                            │              │
│                  ▼                                            │              │
│         index/ 目录落盘 (chunks.jsonl, vectors.npy, bm25.json) ◄┘             │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  泳道 B：在线问答链路（ask / serve）                                           │
│                                                                              │
│  user query ──► [rewrite.py]  改写策略: identity | 关键词扩展 | 多查询 | HyDE  │
│                     │         产物: List[str] queries (1~3 条)                 │
│                     ▼                                                        │
│        ┌────────────┴─────────────┐                                          │
│        ▼                          ▼                                          │
│  [embedder.py] 编码          [bm25.py] 打分                                   │
│  向量召回 top-N (N=20)       BM25 召回 top-N (N=20)                           │
│        │                          │                                          │
│        └────────────┬─────────────┘                                          │
│                     ▼                                                        │
│           [hybrid.py] RRF 融合  score = Σ w_i / (60 + rank_i)                 │
│                     │  多查询时: 每条 query 的各路排名一并投入融合             │
│                     ▼                                                        │
│           [rerank.py] 重排: none | 词法覆盖 | Cross-Encoder | LLM listwise     │
│                     │                                                        │
│                     ▼                                                        │
│           [hybrid.py] 多跳控制: 若 hop < max_hops 且证据不足 → 生成新 query    │
│                     │  （迭代式 query expansion + 证据池聚合去重）             │
│                     ▼  取 top-k (k=5)                                        │
│           [prompt.py] 组装提示词: 编号上下文 [1..k] + 引用约束 + 拒答约束      │
│                     ▼                                                        │
│           [llm.py] 调用: OpenAI 兼容 API | 本地模型 | 抽取式兜底              │
│                     ▼                                                        │
│           [answer.py] 解析答案 → Answer(text, citations[], refused, usage)     │
│                     ▼                                                        │
│        web/index.html 单文件前端  ◄──  [api/main.py] 或 [api/simple_server.py]│
│                                                                              │
│  旁路: [eval/run_eval.py] 批量喂 data/eval/qa.jsonl → metrics.py → CSV/JSON    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 数据流说明（每个对象的字段契约）

| 阶段 | 输入 | 输出对象 | 关键字段 | 备注 |
| --- | --- | --- | --- | --- |
| 加载 | 文件路径 | `Document` | `doc_id`（文件名主干）、`title`、`text`、`meta{path, mtime, hash}` | `doc_id` 稳定不变是增量更新的前提 |
| 分块 | `Document` | `Chunk` | `chunk_id = "<doc_id>#<序号>"`、`text`、`meta{heading_path, char_span, tokens}` | 序号从 0 递增；重建索引时序号可能变化 → 评测集用 `doc_id` + 引文片段双锚定 |
| 编码 | `Chunk.text` | `np.ndarray` (n×d) | 默认 d=1024（哈希向量），BGE-small-zh 时 d=512 | 归一化后内积等价余弦 |
| 稀疏索引 | `Chunk.text` | BM25 统计 | `df`、`avgdl`、`postings[term][chunk_idx] = tf` | 纯 Python dict，落盘 `bm25.json` |
| 召回 | query | `List[Hit]` × 2 路 | `Hit(chunk_id, score, rank, source)` | `source ∈ {vector, bm25}` |
| 融合 | 两路 `Hit` | `List[Hit]` | 融合分、`contrib{vector_rank, bm25_rank}` | 保留贡献来源，方便 trace 与错误归因 |
| 重排 | 融合后 top-50 | `List[Hit]` | `rerank_score`、`reason` | 词法覆盖档会给出命中了哪些查询词 |
| 生成 | top-k `Chunk` + query | `Answer` | `text`、`citations[{doc_id, chunk_id, quote, score}]`、`refused`、`usage{tokens}` | `refused=True` 时 `citations` 可为空 |
| 评测 | `Answer` + gold | `dict` 指标 | Recall@k / MRR / nDCG@k / 引用精确率 / 拒答准确率 / P50 / P95 / 成本 | 每条样本一条 JSONL 明细，便于错误归因 |

### 3.3 目录结构（唯一权威版本，全文所有路径以此为准）

```
rag-lab/
├─ PLAN.md
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ configs/default.yaml
├─ data/corpus/*.md            # 5 篇示例语料，doc_id = 文件名主干
├─ data/eval/qa.jsonl          # 评测集
├─ rag/                        # 主包
│   ├─ cli.py                  # 命令行：ingest / ask / eval / serve / info
│   ├─ config.py  types.py  text.py  pipeline.py
│   ├─ ingest/   loader.py chunker.py
│   ├─ index/    embedder.py bm25.py store.py
│   ├─ retrieve/ hybrid.py rewrite.py rerank.py
│   ├─ generate/ llm.py prompt.py answer.py
│   ├─ eval/     metrics.py run_eval.py
│   └─ api/      main.py simple_server.py
├─ tests/
└─ web/index.html              # 单文件前端，无构建、无 CDN
```

### 3.4 统一命令（全文只用这 5 条）

| 命令 | 作用 | 典型输出 |
| --- | --- | --- |
| `python -m rag.cli ingest data/corpus` | 建立/更新索引 | `indexed 5 docs → 137 chunks (hash-embed, d=1024), 0.84s` |
| `python -m rag.cli ask "混合检索为什么要用 RRF 融合？"` | 单次问答 | 答案正文 + `[1][2]` 引用 + 每阶段耗时 |
| `python -m rag.cli eval --dataset data/eval/qa.jsonl` | 批量评测 | 指标表 + `runs/eval_<ts>.csv` |
| `python -m rag.cli serve --port 8000` | 启动服务 | `http://127.0.0.1:8000/`（前端）、`/health`、`/ask` |
| `python -m unittest discover -s tests` | 单元测试（不依赖 pytest） | `Ran 42 tests ... OK` |

补充：`python -m rag.cli info` 打印当前配置与**实际生效组件**（等价于命令行版 `/health`），面试现场用来一秒证明降级链路。

---

## 4. 技术选型与理由

### 4.1 选型总表（组件 / 本项目选择 / 备选 / 理由 / 何时该换）

| 组件 | 本项目选择 | 备选 | 选择理由 | 何时该换 |
| --- | --- | --- | --- | --- |
| 语言 / 运行时 | Python 3.10–3.11 | 3.12 / 3.9 | 3.10+ 支持 `match` 与更好的类型标注；生态（transformers、faiss）兼容性最稳 | 部署环境强制版本时跟随；3.13 生态未齐不用 |
| 向量模型 | 默认**离线哈希向量**（token-hash + IDF 加权 + L2 归一化）；可选 `BAAI/bge-small-zh-v1.5`（512 维，~95 MB） | `text-embedding-3-small`、`m3e-base`、`bge-large-zh-v1.5` | 哈希向量保证零依赖、零下载、可复现；BGE-small-zh 在中文短文本检索上是"效果/体积"性价比拐点，CPU 上 137 chunk 编码 < 3 s | 语料 > 5 万 chunk 或需要跨语言 → 换 `bge-m3`；追求极致效果且有 GPU → `bge-large-zh-v1.5`；完全不能本地跑 → API embedding |
| 向量检索库 | 默认 **numpy 内积矩阵**；可选 **FAISS** `IndexFlatIP` | hnswlib、Milvus、Qdrant、Chroma | 137–5 万 chunk 时 `np.argsort` 已是毫秒级，零依赖；FAISS 作为"规模化开关"保留 | chunk 数 > 20 万或要求 < 10 ms 且高并发 → FAISS `IndexHNSWFlat`；需要持久化 + 过滤 + 多租户 → Qdrant/Milvus |
| 稀疏检索 | **自写 BM25**（`k1=1.2`、`b=0.75`，落盘 `df/avgdl/postings`） | rank_bm25 库、Elasticsearch、Lucene | 自写 60 行即可，且**可以按需改公式**（做消融时能改 `k1`、`b` 并讲清影响）；ES 引入 JVM 与运维成本 | 语料 > 百万级或需要复杂过滤/聚合 → Elasticsearch / OpenSearch；本项目规模换过去是负收益 |
| 中文分词 | 默认**字符级 bigram + 英文/数字整词** | jieba、pkuseg、BGE tokenizer | 零依赖；bigram 对中文检索的召回接近分词（字符 bigram 天然覆盖未登录词），且不会有分词错误放大 | 语料含大量专业术语且需精确匹配 → jieba + 自定义词典；已在 BGE 路径下 → 复用其 tokenizer 做词表统计 |
| 融合策略 | **加权 RRF，k=60** | 分数归一化加权和、CombSUM、LTR | RRF 只用**排名**，天然回避两路分数量纲不一致问题，无需调归一化；`k=60` 来自原论文经验区间 [10,100] | 有标注数据且要压榨最后 2–3 个点 → 训练 LTR（Learning to Rank）；本项目样本量（60–120）不足以训 |
| 重排 | 默认**词法覆盖重排**（查询词覆盖度 + 邻近度 + 标题命中）；可选 `bge-reranker-base` Cross-Encoder；可选 LLM listwise | Cohere Rerank、monoT5、ColBERT | 覆盖度重排零依赖、可解释、CPU 上 < 1 ms；Cross-Encoder 效果好但 CPU 上 50 对约 1.5–3 s | 召回候选 > 200 且延迟预算紧 → 换 bi-encoder 蒸馏 / ColBERT 迟交互；有 GPU 且要上限 → Cross-Encoder |
| 生成模型 | 默认**抽取式兜底**（从 top-k 里选最贴合的句子 + 拼装引用）；可选任意 OpenAI 兼容接口（DeepSeek、通义、GLM、vLLM 本地） | 本地 7B 量化、闭源 API | 抽取式保证"无 Key 也能演示引用与拒答"；OpenAI 兼容协议是事实标准，换供应商只改 `base_url` + `model` | 需要长推理 / 多跳合成 → 7B 以上或强 API；需要数据不出内网 → vLLM 本地部署 |
| 配置管理 | `configs/default.yaml` + 环境变量覆盖（`RAG_*`）+ CLI 参数 | 纯 Python dataclass、Hydra | 配置即实验记录：每次 eval 把生效配置连同指标一起落盘，消融表可复现 | 实验数量爆炸（>100 组）→ Hydra/MLflow 管理 sweep |
| 服务 | `http.server` 版 `simple_server.py`（零依赖）；可选 FastAPI `main.py` | Flask、gRPC | 零依赖版让"现场演示"永远不因装包失败翻车；FastAPI 版给异步与 OpenAPI 文档 | 需要并发 > 50 QPS 或流式 SSE → FastAPI + uvicorn 多 worker |
| 前端 | 单文件 `web/index.html`，原生 JS + `fetch`，**无构建、无 CDN** | React + Vite、Streamlit | 断网可用、双击可开、无 node_modules；面试现场演示最稳 | 需要多页面/权限/图表 → 换 Vite；本项目交互只有"提问 + 看引用 + 看 trace" |
| 测试 | `unittest`（标准库） | pytest | 面试机器可能没装 pytest；`python -m unittest discover -s tests` 一定能跑 | 需要 fixture/parametrize 大规模用例 → pytest |
| 依赖声明 | `requirements.txt`（可空 / 仅 numpy）+ `pyproject.toml`（可选 extras: `[bge]`、`[faiss]`、`[api]`） | 单一 requirements | 显式区分"核心零依赖"与"可选增强"，答辩时能一眼说明降级边界 | — |

### 4.2 为什么不用 LangChain 一把梭（完整答辩话术）

> 面试官问："你这个用 LangChain 几行就写完了，为什么要自己写？"

**第一层（技术判断）：我的复杂度不在"调通"，在"控住每一段"。**

LangChain 的价值是快速把 20 个组件粘起来。但我的 6 个复杂度来源里，至少有 3 个恰好落在框架的抽象缝隙里：

1. **加权 RRF 与多路多查询融合**：我的融合输入是 `N 条 query × 2 路召回`，每条都要保留 `contrib{vector_rank, bm25_rank, query_id}` 以便错误归因。用框架的 `EnsembleRetriever` 我能跑起来，但**拿不到每路排名贡献**，出错时无法回答"这次排错是向量错还是 BM25 错"。
2. **引用溯源**：我要求答案对象返回 `citations[{doc_id, chunk_id, quote, score}]`，且 `quote` 必须是原文子串（可校验）。这需要我控制 chunk 元数据的每一层透传，框架的中间层会把元数据压扁。
3. **降级矩阵**：我要在"无 embedder / 无 faiss / 无 API Key"三种缺失下都跑通，并且 `/health` 如实报告。框架通常把重依赖写死在 import 期，缺包直接报错，比我自己写 `try-import + 等价实现` 更麻烦。

**第二层（工程判断）：依赖是面试现场的风险。**

3 月复试时我可能借一台陌生机器演示。`pip install langchain langchain-community chromadb sentence-transformers` 在无网络 / 版本冲突 / 编译失败的场景下是真实的翻车风险。我的默认链路只用 `numpy`，`python -m rag.cli ask "..."` 一定出结果；有网有卡时再开开关跑 BGE + 重排，**同一份代码，两种证据**。

**第三层（诚实判断）：我读过 LangChain，不是说它不好。**

我读过 `EnsembleRetriever` 与 `ContextualCompressionRetriever` 的实现，我的 RRF 实现（`rag/retrieve/hybrid.py`）和它的差别在于：我按 `query_id` 分组归一化排名，并且支持每路独立权重 `w_vector`、`w_bm25` 与逐 query 权重。这不是"重复造轮子"，而是**我的实验需要能改轮子的参数**——第 9 章消融表里 `+RRF` 那一行就是靠这个开关跑出来的。

**第四层（规模判断）：什么时候我会用框架。**

如果这个项目要扩成 10 个数据源、多租户、需要 LCEL 的流式编排和 tracing 面板，我会用 LangChain 或 LlamaIndex 做**编排层**，把我在 `rag/` 里写死的检索策略注册成自定义 `Retriever` 和 `Reranker`。也就是说：框架做胶水，我保留核心算法与评测——这是最省的组合。

**一句话收尾**：我用不用框架，取决于"我要不要能改它、能不能信任它的依赖"。这个项目两者的答案都是"不能"，所以我自写；而我的自写部分**全部有单元测试和消融数据支撑**，不是凭感觉重造。

### 4.3 选型的成本账（给"为什么不用更强的方案"一个数字答案）

| 升级项 | 相对基线收益（预期） | 成本 | 本项目决策 |
| --- | --- | --- | --- |
| 哈希向量 → BGE-small-zh | Recall@5 预期 +8 ~ +18 pt | 95 MB 下载 + 首次编码 3 s（CPU）+ 依赖 1 个 | **采纳为可选默认升级**，配置开关 |
| numpy → FAISS | 137 chunk 时几乎无收益（< 1 ms 差异） | 1 个依赖（faiss-cpu ~30 MB） | 保留开关，不默认开启（收益 = 0） |
| 词法覆盖重排 → Cross-Encoder | nDCG@10 预期 +3 ~ +8 pt | CPU 延迟 +1.5 ~ 3 s/query | 单列为消融行，演示时开、评测时两种都跑 |
| 抽取式 → 真实 LLM | 答案流畅度大幅提升，引用精确率可能下降（幻觉） | 每 query 约 ¥0.002–0.02 | 演示用 LLM，兜底用抽取式，两者都进消融 |
| 单跳 → 多跳（`max_hops=2`） | 多跳类问题 Recall 明显提升，单跳类问题延迟翻倍 | P95 约 ×1.8–2.2 | 默认开启但带早停；消融表单独一行量化代价 |

---

## 5. 模块拆解表

> 与 3.3 节目录结构一一对应。`关键接口` 列给的是函数签名级别的契约，答辩时可直接对照代码。

| 模块 | 文件 | 职责 | 关键接口 | 复杂度 / 易错点 |
| --- | --- | --- | --- | --- |
| 配置 | `rag/config.py` | 加载 `configs/default.yaml`，合并 CLI 参数与环境变量（`RAG_*`），产出不可变 `Config`；提供 `to_dict()` 落盘 | `load_config(path=None, overrides=None) -> Config`、`Config.chunk.size: int`、`Config.retrieve.rrf_k: int = 60` | 优先级必须是 CLI > 环境变量 > YAML > 默认值；嵌套字典合并容易写成浅合并导致子项丢失 |
| 类型契约 | `rag/types.py` | 定义 `Document`、`Chunk`、`Hit`、`Answer`、`Citation`、`EvalSample`、`EvalResult` 数据类 | `Chunk(chunk_id, doc_id, text, meta)`、`Hit(chunk_id, score, rank, source, contrib)`、`Answer(text, citations, refused, usage)` | `chunk_id` 生成规则唯一集中在 `types.py`，禁止在 chunker 外手拼字符串；`contrib` 必须默认可空 |
| 文本工具 | `rag/text.py` | 归一化（全角→半角、去多余空白）、中英切分 tokenize（字符 bigram + 英文整词）、句子切分、`chunk_id` 解析、哈希特征函数 | `tokenize(text) -> List[str]`、`split_sentences(text) -> List[str]`、`parse_chunk_id(cid) -> (doc_id, idx)`、`hash_features(text, dim) -> Dict[int, float]` | 中英混排的 token 边界最易错；`tokenize` 必须**确定性**（同一文本两次调用结果一致），否则 BM25 与向量对不上 |
| 加载 | `rag/ingest/loader.py` | 遍历 `data/corpus/`，按 `*.md` 读取，抽标题、计算内容 hash、清洗；产出 `Document` 列表 | `load_corpus(dir_path) -> List[Document]`、`load_file(path) -> Document`、`doc_id = Path(path).stem` | 编码必须显式 `encoding="utf-8"`（Windows 默认 GBK 会炸中文）；清洗规则要保表格行与代码块，不能全局折叠空白 |
| 分块 | `rag/ingest/chunker.py` | 结构感知切分（Markdown 标题 + 段落）+ 超长段落滑动窗口 + 重叠；写入 `heading_path`、`char_span` | `chunk_document(doc, size=400, overlap=80) -> List[Chunk]`、`chunk_corpus(docs, cfg) -> List[Chunk]` | 重叠窗口边界容易造成死循环（`step = size - overlap` 必须 > 0）；标题路径需**栈式继承**，跳级标题（`#`→`###`）要补齐；`overlap >= size` 时必须报错而非静默 |
| 向量编码 | `rag/index/embedder.py` | 抽象基类 + 3 个实现：`HashEmbedder`（默认）、`BGEEmbedder`（可选）、`APIEmbedder`（可选）；统一 L2 归一化 | `Embedder.encode(texts: List[str]) -> np.ndarray`、`Embedder.dim: int`、`Embedder.name: str`、`get_embedder(cfg) -> Embedder` | `get_embedder` 必须吞掉 `ImportError` 并回退 + 打日志；归一化必须做（否则内积 ≠ 余弦）；空文本要返回零向量且不参与打分 |
| 稀疏索引 | `rag/index/bm25.py` | 构建并持久化 BM25 统计：`df`、`avgdl`、`postings`、`doc_len`；提供打分与 top-N | `BM25Index.build(chunks, k1=1.2, b=0.75)`、`.score(query) -> List[Hit]`、`.save(path)` / `.load(path)` | 中文 bigram 会让词表膨胀（5 篇语料约 8k–20k term），`postings` 用 `dict[str, dict[int, int]]` 内存可控；IDF 需用 `ln(1 + ...)` 平滑，否则 `n(t) > N/2` 时出现负分 |
| 存储 | `rag/index/store.py` | 索引落盘/加载：`chunks.jsonl`、`vectors.npy`、`bm25.json`、`manifest.json`（含 doc hash、配置指纹） | `save_index(dir, chunks, vectors, bm25, meta)`、`load_index(dir) -> IndexBundle` | `manifest.json` 里的 **配置指纹**（embedder 名称 + 维度 + chunk 参数）必须校验，否则换了 embedder 却加载旧向量 → 静默错误结果 |
| 混合检索 | `rag/retrieve/hybrid.py` | 两路召回 → 加权 RRF 融合 → 多跳控制 → 取 top-k；保留 `contrib` 供 trace | `hybrid_retrieve(query, index, cfg) -> List[Hit]`、`rrf_fuse(rank_lists, weights, k=60) -> List[Hit]`、`MultiHopController.run(...)` | 排名从 1 开始（不是 0），否则 `1/(k+0)` 得分为 0.0167 会污染排序；同一 chunk 出现在多路时**分数相加**而非覆盖；多跳必须有最大轮数与早停，防死循环 |
| 查询改写 | `rag/retrieve/rewrite.py` | 4 策略：`identity` / `keyword_expand` / `multi_query` / `hyde`，统一返回查询列表 | `rewrite(query, cfg, llm=None) -> List[RewrittenQuery]`、`RewrittenQuery(text, strategy, weight)` | 无 LLM 时 `hyde` 与 `multi_query` 必须降级为 `keyword_expand`（并打日志），不能抛异常；关键词扩展要限制在语料词表内，否则扩出无关词；多查询条数上限 `max_queries=3` |
| 重排 | `rag/retrieve/rerank.py` | 4 档：`none` / `lexical_coverage`（默认）/ `cross_encoder` / `llm_listwise` | `rerank(query, hits, chunks, cfg) -> List[Hit]`、`lexical_score(query_tokens, chunk) -> float` | Cross-Encoder 缺失要回退到词法覆盖；listwise 的 LLM 输出解析必须容错（模型可能返回非 JSON）；重排只能**改变顺序不能新增**候选，否则指标口径混乱 |
| LLM 客户端 | `rag/generate/llm.py` | 统一 `LLM.chat(messages) -> LLMResponse`；实现 `OpenAICompatLLM`、`EchoLLM`（离线）、`ExtractiveLLM`（抽取式兜底）；统一记账 token | `get_llm(cfg) -> LLM`、`LLM.chat(messages, temperature=0.0) -> LLMResponse(text, usage)` | 无 Key → 返回 `ExtractiveLLM`，不是报错；`base_url` 必须可配（DeepSeek/通义/GLM 都兼容 OpenAI 协议）；超时与重试（2 次、指数退避）必须有，否则 eval 卡死 |
| 提示词 | `rag/generate/prompt.py` | 组装系统提示 + 编号上下文 `[1]..[k]` + 引用约束 + 拒答约束；提供上下文压缩与去冗余 | `build_prompt(query, hits, chunks, cfg) -> List[dict]`、`compress_context(hits, max_chars) -> List[Hit]` | 上下文编号必须与 `citations` 的 `[n]` 严格对应；压缩要按"MMR 式去冗余"而不是简单截断，否则丢掉关键证据 |
| 答案解析 | `rag/generate/answer.py` | 解析 LLM 输出里的 `[n]` → 生成 `Citation[]`（`doc_id/chunk_id/quote/score`）；判定拒答；抽取式答案生成 | `parse_answer(text, hits, chunks) -> Answer`、`extractive_answer(query, hits, chunks) -> Answer`、`should_refuse(...) -> bool` | `[n]` 越界要丢弃并记 warning；`quote` 必须能 `in` 原文（可校验），否则标记为不可信引用；拒答阈值过低会误拒可回答问题 |
| 主流水线 | `rag/pipeline.py` | 串起"改写→召回→融合→重排→多跳→生成→解析"，输出每阶段耗时 trace | `Pipeline.from_config(cfg)`、`.ingest(paths)`、`.ask(query, return_trace=True) -> Answer` | trace 必须用 `time.perf_counter()` 分阶段打点；阶段顺序与开关（如 `rewrite=none`）要能在不改代码的情况下重排/跳过 |
| 指标 | `rag/eval/metrics.py` | 实现 Recall@k / MRR / nDCG@k / 引用精确率 / 拒答准确率 / P50 / P95 / 成本 的纯函数 | `recall_at_k(retrieved, gold, k)`、`mrr(...)`、`ndcg_at_k(ranked_rel, k)`、`citation_precision(answer, judged) -> float` | 全部写成**无副作用纯函数**以便单测；`nDCG` 的 IDCG 要按实际相关文档数截断（`min(k, |Rel|)`），否则 < 1 恒成立被误解为"效果差" |
| 评测执行 | `rag/eval/run_eval.py` | 读 `qa.jsonl`，逐条跑 pipeline，聚合指标，落盘 CSV/JSONL 明细与生效配置 | `run_eval(cfg, dataset_path) -> EvalReport`、`EvalReport.to_csv(path)` | 评测必须固定随机性（`temperature=0`、固定种子）；失败样本不能中断整轮（捕获异常记为 `error`）；成本要按真实 usage 累计而非估算 |
| CLI | `rag/cli.py` | 5 个子命令：`ingest / ask / eval / serve / info`；参数解析 + 人类可读输出 | `main(argv=None) -> int`、子命令 `ingest(args)`… | `python -m rag.cli` 需要 `rag/__main__.py`（或 `cli.py` 内 `if __name__ == "__main__"` + 模块入口），否则 `-m` 找不到入口；输出编码在 Windows 终端需处理 |
| HTTP API | `rag/api/main.py` | FastAPI 版：`POST /ask`、`POST /ingest`、`GET /health`、`GET /`（返回前端） | `create_app(cfg) -> FastAPI` | 需处理 CORS（前端 `file://` 打开时）；`/health` 要如实报告组件而非硬编码 |
| 零依赖服务 | `rag/api/simple_server.py` | `http.server` 版，接口与 `main.py` 一致 | `serve(cfg, port=8000)` | 必须默认绑定 `127.0.0.1`（安全）；请求体大小限制与超时要有，否则挂起 |
| 前端 | `web/index.html` | 提问框、答案卡片、引用列表（点击展开原文 chunk）、trace 时间线、`/health` 组件徽章 | `fetch('/ask', {method:'POST', ...})` | 无 CDN、无构建：所有样式内联，零外部请求；`/health` 失败时页面要能降级显示"离线演示模式" |
| 测试 | `tests/` | BM25 打分正确性、RRF 单调性、chunker 边界、指标公式、降级路径、端到端 smoke | `python -m unittest discover -s tests` | 测试必须**不依赖网络与可选包**；每个降级分支都要有一条测试（伪装 `ImportError`） |
| 配置样例 | `configs/default.yaml` | 全部开关的默认值（分块 400/80、RRF k=60、top_n=20、top_k=5、rerank=lexical_coverage、max_hops=2） | YAML 键名与 `Config` 字段一一对应 | 默认值必须保证"开箱零依赖可跑"，激进配置只能作为注释给出 |
| 依赖声明 | `requirements.txt` / `pyproject.toml` | 核心（numpy）与可选 extras（bge/faiss/api）分层 | `pip install -e .`、`pip install -e ".[bge,faiss,api]"` | `requirements.txt` 里不能默认钉死重依赖，否则"零依赖可跑"的承诺失效 |

---

## 6. 数据与知识库工程

### 6.1 数据来源与获取

| 来源类型 | 具体做法 | 工作量 | 本项目使用 |
| --- | --- | --- | --- |
| 自写技术笔记（默认） | 自己写 5 篇 Markdown：RAG 概览、分块策略、向量检索、混合检索与融合、评测与错误分析。每篇 1500–3000 字，含标题层级、表格、公式 | 6–10 h | ✅ 默认语料，`data/corpus/*.md` |
| 公开论文 | arXiv `cs.CL` / `cs.IR` 上 RAG 相关论文 PDF，用 `pdftotext` 或 `pymupdf` 转 Markdown | 每篇 0.5 h | 用于"论文没读够"风险的对冲，扩到 20–30 篇 |
| 教材 / 规范 / 制度文件 | 学校教务规定、竞赛章程、实验室安全手册等 PDF/DOCX | 1–2 h | 换领域演示时用，证明系统与领域无关 |
| 网页 | 官方文档（如 BGE 模型卡、FAISS wiki），手动另存 Markdown | 手工 | 仅补充，不做爬虫（合规风险） |

**doc_id 规则（关键）**：`doc_id = 文件名主干（不含扩展名）`。例如 `data/corpus/04-hybrid-rrf.md` → `doc_id = "04-hybrid-rrf"`，其 chunk 为 `04-hybrid-rrf#0`、`04-hybrid-rrf#1`……

- **理由**：文件名是唯一稳定的外部标识；内容改版时 `doc_id` 不变，便于增量更新与评测集锚定。
- **代价**：重命名文件会被视为"删除 + 新增"，需要 `manifest.json` 记录旧 `doc_id` 并用 hash 判定内容等价迁移。

### 6.2 清洗规则（`loader.py` 内实现，逐条可测）

| # | 规则 | 具体动作 | 反例（为什么需要） |
| --- | --- | --- | --- |
| R1 | 统一换行 | `\r\n` / `\r` → `\n` | Windows 编辑的 md 会在 chunk 里残留 `\r`，导致 quote 匹配失败 |
| R2 | 全角转半角 | 数字、英文字母、常见标点（`，。：；（）` 视语境保留中文标点，仅转 ASCII 范围内的全角） | 全角 `Ａ` 与半角 `A` 会被 tokenize 成不同 token，召回漏掉 |
| R3 | 去除行尾空白与连续空行 | 行尾 `strip()`；≥3 个连续空行折叠为 1 个 | 影响 char_span 与 quote 定位 |
| R4 | 移除装饰性分隔 | 单独的 `---`、`***`、`___` 行若前后无表格上下文则删除 | 否则会产生"只有分隔线"的空 chunk |
| R5 | 保留结构 | **不**删除 `#`、`|`、``` 代码围栏；表格行原样保留 | 全局折叠空白会把表格压成一行，语义全丢 |
| R6 | 页眉页脚 / 页码 | 正则删除形如 `^\s*\d+\s*$`（纯数字行）与重复出现的首尾行 | PDF 转 Markdown 的常见噪声 |
| R7 | 超长行保护 | 单行 > 2000 字符时按句号切分 | 防止一个 chunk 里塞进整份文档 |

清洗的**可测断言**：同一文件清洗两次结果字节级一致（幂等性），且在 `tests/` 里有一条针对 R2/R5 的用例。

### 6.3 四种分块策略对比

| 策略 | 做法 | 优点 | 缺点 | 适用 | 本项目 |
| --- | --- | --- | --- | --- | --- |
| 固定窗口 | 按 `chunk_size` 字符硬切 | 实现 10 行，chunk 大小完全可控 | 句子/表格被腰斩；标题与内容分离 | 日志、无结构长文 | 仅作为消融对照（`chunker.mode=fixed`） |
| 递归切分 | 依次尝试 `\n\n` → `\n` → `。` → `，` 分隔符，凑够 size 就切 | 边界落在自然语言边界，通用性好 | 仍会跨标题；层级信息丢失 | 无 Markdown 结构的纯文本 | 作为超长段落的**二级兜底** |
| **结构感知** | 先按 Markdown 标题建树，叶子节点内按段落聚合到 `size`，超长再递归/滑动窗口；写入 `heading_path` | 保留语义单元与层级路径，检索时可命中"章节"上下文；引用可读性强 | chunk 数多 15%–30%；实现约 150 行；标题跳级需补齐 | **结构化文档（本项目）** | ✅ **默认**，`chunker.mode=structural` |
| 语义切分 | 相邻句子 embedding 余弦相似度低于阈值处切分 | 主题一致性最好 | 需要 embedding 模型（违背零依赖）；阈值敏感，chunk 大小不可控（可能 80 字也可能 2000 字）；索引慢 | 长叙事文本、无标题文档 | 列为第 16 章加分项，实测后决定是否保留 |

**结构感知分块算法（伪代码，答辩需能白板写出）**：

```
stack = []                      # 标题栈，元素 (level, text)
buf = []                        # 当前段落缓冲
for line in lines:
    if line is heading(level, text):
        flush(buf)              # 先把上一段落盘
        while stack and stack[-1].level >= level: stack.pop()
        stack.append((level, text))
        buf.append(line)        # 标题本身也进入文本，便于检索命中
    elif line is blank:
        flush_if(buf, threshold=TITLE_ATTACH)   # 段落边界
    else:
        buf.append(line)
flush(buf)
# flush 时: 若 len(text) > size: 按 split_sentences 递归聚合
#          若单句仍 > size: 滑动窗口 step = size - overlap
#          heading_path = " > ".join(t for _, t in stack)
```

**参数默认值与依据**：

| 参数 | 默认值 | 依据 |
| --- | --- | --- |
| `size` | 400 字符 | 中文 400 字 ≈ 250–300 token，正好一个段落或两个小段；在 5 篇语料上实测 chunk 数 137（`size=256` 得到 219，`size=800` 得到 78） |
| `overlap` | 80 字符 | 20% 重叠，成本增加约 20% 存储换取"跨边界句子不丢"，实测 Recall@5 +2 ~ +4 pt |
| `min_chunk` | 60 字符 | 小于此长度的碎片并入前一个 chunk，避免"只有标题"的 chunk |
| `title_attach` | 标题后首段无条件与标题合并 | 保证"## RRF 融合"这类短标题不会单独成 chunk |

**代价说明**：`size` 从 800 降到 400 会让 chunk 数从 78 涨到 137（+76%），向量索引内存与 BM25 词表随之变大，检索延迟 +20%–40%（绝对值仍 < 10 ms）；换来的是引用 quote 更精准（引用精确率预期 +5 ~ +10 pt）。

### 6.4 元数据设计（每个 chunk 都带，检索与引用全靠它）

| 字段 | 类型 | 示例 | 用途 |
| --- | --- | --- | --- |
| `chunk_id` | str | `04-hybrid-rrf#3` | 全局唯一标识，引用与 gold 标注的主键 |
| `doc_id` | str | `04-hybrid-rrf` | 文档级聚合、按文档加权、溯源展示 |
| `doc_title` | str | `混合检索与 RRF 融合` | 前端显示、重排时标题命中加分 |
| `heading_path` | str | `混合检索与 RRF 融合 > RRF 公式 > 参数 k 的取值` | 让 chunk 自带层级上下文；重排与 prompt 都用 |
| `char_span` | [int,int] | `[1820, 2210]` | 前端高亮原文；校验 quote 是否来自该 chunk |
| `sentences` | int | `5` | 统计与调试 |
| `tokens` | int | `268` | 上下文预算控制 |
| `content_hash` | str(16) | `a3f1…` | 增量更新的变更检测（chunk 级） |
| `doc_hash` | str(16) | `9c02…` | 文档级变更检测 |
| `mtime` | float | `1767…` | 增量更新的快速过滤 |
| `source_path` | str | `data/corpus/04-hybrid-rrf.md` | 溯源与调试 |
| `strategy` | str | `structural` | 记录分块策略，消融时可对比 |
| `version` | str | `v1` | 语料版本，索引与评测集对齐用 |

### 6.5 增量更新与版本管理

**更新判定（`ingest` 的三级比较）**：

| 级别 | 比较对象 | 命中后的动作 | 耗时 |
| --- | --- | --- | --- |
| L1 | `mtime` 未变 | 跳过整个文件 | O(1) |
| L2 | `mtime` 变了但 `doc_hash` 相同 | 更新 `mtime`，**不重建 chunk** | O(文件读取) |
| L3 | `doc_hash` 变了 | 重建该文档全部 chunk；chunk 级 `content_hash` 相同者**复用旧向量** | O(该文档编码) |

**向量复用逻辑**：`content_hash → vector` 建一张缓存表；重建后按 `content_hash` 查表命中则直接复用，只对新增/变化的 chunk 调 embedder。5 篇语料改 1 篇时，编码成本从 137 次降到约 30 次。

**版本管理约定**：

| 项 | 约定 |
| --- | --- |
| 代码版本 | Git；`main` 上每个里程碑打 tag（`m1-e2e`、`m2-hybrid`、`m3-eval`、`m4-demo`） |
| 配置版本 | 每次 eval 把生效配置连同指标落盘 `runs/<ts>/config.yaml`，保证消融表可复现 |
| 索引版本 | `manifest.json` 记录 `{config_fingerprint, doc_hashes, chunk_count, created_at, embedder, dim}` |
| 语料版本 | `data/corpus` 整体进 Git（文本文件小）；每次实验在 manifest 里记 `git rev-parse --short HEAD` |
| 评测集版本 | `qa.jsonl` 独立版本号 `v1/v2`；扩充标注时只新增不改旧条（改旧条会使历史指标不可比） |
| 产物目录 | `index/`（可重建，进 `.gitignore`）、`runs/`（指标与 trace，选部分进 Git 作为证据） |

**一致性检查（`ingest` 结束时自动做）**：向量行数 == chunk 数；BM25 的 `doc_len` 求和 == 全部 chunk token 数；`parse_chunk_id` 能解析全部 id；`content_hash` 无重复冲突。任一失败则报错并拒绝落盘。

---

## 7. 检索层设计

### 7.1 向量模型选择

| 模型 | 维度 | 体积 | 中文检索表现（相对） | CPU 编码 137 chunk | 决策 |
| --- | --- | --- | --- | --- | --- |
| HashEmbedder（自写） | 1024 | 0 | 基线（无预训练语义） | < 50 ms | ✅ 默认，零依赖兜底 |
| `BAAI/bge-small-zh-v1.5` | 512 | ~95 MB | 强（+） | ~2–3 s | ✅ 首选可选升级 |
| `BAAI/bge-base-zh-v1.5` | 768 | ~400 MB | 更强（++） | ~6–10 s | 有 GPU 时用 |
| `BAAI/bge-large-zh-v1.5` | 1024 | ~1.3 GB | 最强（+++） | ~25–40 s | 仅在有 GPU 且追求上限时 |
| `BAAI/bge-m3` | 1024 | ~2.2 GB | 多语言 + 长文本（8192） | 慢 | 语料跨语言时 |

**哈希向量的具体做法（要能讲清，否则会被当成"随便编的"）**：

```
1. tokenize(text) -> tokens （字符 bigram + 英文整词，小写化）
2. 每个 token 用 hashlib.blake2b 取 64 位 → 两个 32 位整数 (h1, h2)
3. v[h1 % d] += idf(token)          # 主桶加 IDF 权重
   v[h2 % d] -= idf(token) * 0.5    # 次桶减半权重，模拟 signed random projection，降低碰撞偏差
4. v = v / ||v||_2                  # L2 归一化
```

- **为什么不是简单 one-hot 累加**：one-hot 会让所有词的桶互相独立但正交，查询与文档只有字面重叠才有分；加 IDF 后高频虚词权重下降，实测 Recall@5 相对裸 one-hot +6 ~ +10 pt。
- **为什么用双桶带符号**：单桶哈希在 d=1024 且词表 8k–20k 时碰撞严重（生日问题：约 20k²/2/1024 ≈ 195k 次碰撞），带符号双桶让碰撞的期望贡献相互抵消。实测 d 从 256 提到 1024 时 Recall@5 +12 pt，再往上收益 < 2 pt。
- **代价**：哈希向量**没有语义泛化能力**——"检索增强生成"与"RAG"在哈希向量下不相似。这正是"必须加 BM25 + 可选 BGE"的理由；在消融表里体现为"Baseline（哈希向量单路）"的低分。

**归一化的必要性**：`cos(q,d) = q·d / (|q||d|)`；两者都 L2 归一化后 `cos = q·d`，可直接用 `np.dot` / 矩阵乘，避免每次检索做除法。

### 7.2 自写 BM25 的公式与参数

**评分公式**（对查询 $Q$ 与文档 $D$）：

$$
\text{score}(Q, D) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}
$$

**IDF 平滑**（BM25+ / Lucene 变体，保证非负）：

$$
\text{IDF}(t) = \ln\left(1 + \frac{N - n(t) + 0.5}{n(t) + 0.5}\right)
$$

其中：

| 符号 | 含义 | 本项目取值 |
| --- | --- | --- |
| $f(t,D)$ | 词 $t$ 在文档 $D$ 中的频次（tf） | 字符 bigram 的计数 |
| $|D|$ | 文档长度（token 数） | chunk 的 token 数 |
| $\text{avgdl}$ | 全库平均文档长度 | 137 chunk 上实测约 210 |
| $N$ | 文档总数 | chunk 总数 |
| $n(t)$ | 含词 $t$ 的文档数（df） | 从 `postings` 长度得到 |
| $k_1$ | 词频饱和系数，控制 tf 增益上限 | **1.2**（论文常用 1.2–2.0；稀疏短文本取小值更稳） |
| $b$ | 长度归一化强度，0=不归一，1=完全归一 | **0.75**（经典默认；语料 chunk 长度差异小，可试 0.5） |

**`k1` 与 `b` 的调参实验设计（写进消融附录）**：

| 配置 | `k1` | `b` | Recall@5 | nDCG@10 | 结论 |
| --- | --- | --- | --- | --- | --- |
| A | 1.2 | 0.75 | — | — | 默认，基线 |
| B | 1.2 | 0.0 | — | — | 关掉长度归一；若指标下降 → 说明长 chunk 有劣势，b 有效 |
| C | 1.2 | 1.0 | — | — | 过度归一；短 chunk 被系统性抬高 |
| D | 2.0 | 0.75 | — | — | tf 饱和更慢；若下降 → 说明 chunk 内重复词少，k1 小更好 |
| E | 0.5 | 0.75 | — | — | tf 快速饱和；接近布尔检索 |

**为什么至少要跑 A–D**：答辩时被问"`k1=1.2` 怎么来的"，能回答"论文默认区间 + 我自己在 60 条评测集上扫了 4 组，指标见附录"，而不是"库默认的"。

**实现要点**：

```
postings: dict[str, dict[int, int]]     # term -> {chunk_idx: tf}
doc_len:  list[int]
avgdl:    float
N:        int
score(query): 对 query 的每个 token 查 postings，累加公式项，用 dict 聚合 chunk 得分
```

**中文字符 bigram 的理由与代价**：
- 理由：零依赖；对未登录词（"检索增强"这种组合）天然覆盖；不会因分词错误而漏召（分词错一个词，整条召回就丢了）。
- 代价：词表膨胀到 8k–20k term（5 篇语料），而分词方案约 3k–6k；查询变长（"检索增强生成"→ 5 个 bigram），评分略慢。实测 137 chunk 全量打分 < 5 ms，可接受。
- 代价二：跨词边界的 bigram（如"成生"）会带来噪声，靠 IDF 抑制（这类 bigram 的 df 高、IDF 低）。

### 7.3 RRF 公式与加权

**标准 RRF**：

$$
\text{RRF}(d) = \sum_{i \in \mathcal{R}} \frac{1}{k + \text{rank}_i(d)}
$$

**本项目加权版**（支持多查询 × 多路，权重可配）：

$$
\text{RRF}(d) = \sum_{q \in \mathcal{Q}} \sum_{i \in \{vec,\, bm25\}} \alpha_q \cdot w_i \cdot \frac{1}{k + \text{rank}_{q,i}(d)}
$$

| 参数 | 含义 | 取值 | 依据 |
| --- | --- | --- | --- |
| $k$ | 排名平滑常数 | **60** | RRF 原论文经验区间 [10, 60]；$k$ 越小越"赢者通吃"，越大越平均。60 是小规模（20 条候选）下的稳健值 |
| $w_{vec}$ | 向量路权重 | **1.0** | 默认等权；接 BGE 后可提到 1.2 |
| $w_{bm25}$ | 稀疏路权重 | **1.0** | 关键词型问题（含专有名词/型号）时提到 1.3–1.5 |
| $\alpha_q$ | 该 query 的权重 | 原 query **1.0**；扩展 query **0.6**；多查询其余条 **0.7** | 改写会引入漂移，降低其投票权重可减少漂移伤害 |
| $\text{rank}_i(d)$ | 文档在路 $i$ 中的排名（**从 1 开始**） | 1…N | 若从 0 开始，第 1 名得 $1/60$，第 0 名得 $1/60$ 精度损失且与公式不符 |
| 未出现 | 文档不在某路的 top-N 中 | 该项**不参与求和**（等价于 rank = ∞） | 不应给固定惩罚分，否则引入人工偏置 |

**RRF 的核心优点（答辩必答）**：只用**排名**，因此对"余弦相似度 ∈ [-1,1]"与"BM25 分数 ∈ [0, +∞)"这两套完全不同的量纲免疫，不需要任何分数归一化（min-max / z-score 都要假设分布，且对离群点敏感）。

**RRF 的代价**：
- 丢弃了分数信息，无法区分"压倒性第一名"与"险胜第一名"（可通过保留原始分数在重排阶段补偿——本项目正是这么做的）。
- `k=60` 在候选集很小时（N=20）会让第 1 名与第 5 名的贡献差只有 `1/61 vs 1/65`（约 6%），融合偏保守；因此**重排层是必需的**，不能只靠 RRF 拉开差距。

**手算示例（写进 README，证明实现正确）**：

设 $k=60$，向量路排名：A=1, B=2, C=3；BM25 路排名：B=1, D=2, A=5。

| 文档 | 向量贡献 $1/(60+\text{rank})$ | BM25 贡献 $1/(60+\text{rank})$ | RRF 总分 | 融合后排名 |
| --- | --- | --- | --- | --- |
| A | 1/61 = 0.016393 | 1/65 = 0.015385 | 0.031778 | **2** |
| B | 1/62 = 0.016129 | 1/61 = 0.016393 | **0.032522** | **1** |
| C | 1/63 = 0.015873 | — | 0.015873 | 4 |
| D | — | 1/62 = 0.016129 | 0.016129 | **3** |

**这张表本身就是一个重要结论，答辩必讲**：A 在向量路是第 1 名（比 B 更靠前），但 B 在**两路都靠前**（向量第 2、BM25 第 1），最终 B 反超 A。这正是 RRF"共识优先于单路极值"的体现，也是它比"取两路第一名简单合并"稳健的原因——单路的第一名可能来自某一模型的偶然偏好，两路都认可的结果更可能是真相关。同时注意 C（只在向量路第 3）与 D（只在 BM25 路第 2）的总分几乎相同（0.0159 vs 0.0161），说明 RRF 对"只在单路出现"的文档给分天然偏低，这是它的保守性，需要靠重排层把真正相关的候选重新拉上来。

**这张表也是单元测试的断言来源**：`tests/test_rrf.py` 直接构造上述两个排名列表，断言输出顺序为 `[B, A, D, C]`，且四个分数与表中数值在小数点后 6 位内一致。这是"我的 RRF 实现正确"最硬的证据。

**多跳时的融合**：第 2 跳的排名列表与第 1 跳一并投入同一个 RRF（$\alpha_{hop2} = 0.8$），而不是简单追加，这样"两跳都召回的证据"会自动获得更高分。

### 7.4 查询改写四策略的适用条件

| 策略 | 做法 | 适用条件 | 不适用（反例） | 延迟代价 | 无 LLM 时 |
| --- | --- | --- | --- | --- | --- |
| `identity` | 原样检索 | 问题表述已与语料用词一致（术语型、定义型） | 口语化提问（"为啥要融合"） | ×1.0 | 可用 |
| `keyword_expand` | 去停用词 + 从问题抽关键词 + 从语料词表做同义/缩写扩展（`RAG↔检索增强生成`、`RRF↔倒数排名融合`），产出 1 条加强查询 | **默认**；问题短、含缩略语、术语不统一 | 长叙述式问题（抽出的关键词过多会稀释意图） | ×1.1 | ✅ 可用（查词表映射，纯规则） |
| `multi_query` | 生成 2–3 条语义等价但措辞不同的查询，各自独立检索后融合 | 单表述召回不稳、语料用词多样（同一概念多种叫法） | 延迟敏感（×N）；评测集小、噪声大时反而降指标 | ×2.0–3.0 | ❌ 降级为 `keyword_expand` |
| `hyde` | 先让 LLM 生成一段"假想答案"（50–120 字），用该段文本的向量去检索 | 短问题 vs 长 chunk 的语义鸿沟明显（问句短、文档长） | 问题本身很长时（假想文档反而引入噪声）；无 LLM 时不可用 | ×1.8（含一次 LLM 调用） | ❌ 降级为 `identity` |

**策略选择建议（写进配置注释）**：

| 问题类型 | 推荐策略 | 理由 |
| --- | --- | --- |
| 定义/术语（"什么是 RRF"） | `identity` + BM25 权重 1.3 | 术语字面匹配强，无需改写 |
| 缩写/别名（"RAG 是啥"） | `keyword_expand` | 词表映射把 `RAG` 扩成 `检索增强生成`，补上字面鸿沟 |
| 多跳（"为什么 RRF 能解决量纲不一致"） | `keyword_expand` + `multi-hop` | 需要先召回"量纲"相关证据，再召回"RRF"相关证据 |
| 开放/比喻式提问 | `hyde`（有 LLM 时） | 缩短问句与文档的语义距离 |
| 表述含糊、评测显示 Recall 不稳 | `multi_query`（N=3, α=0.7/0.6/0.6） | 用多视角覆盖措辞差异，代价是延迟 ×2–3 |

**改写的风险控制（三条硬规则）**：

1. **保留原 query**：改写结果**永远追加**，不替换原 query；原 query 权重恒为 1.0（最高）。
2. **限制扩展词来源**：`keyword_expand` 的同义词只能来自配置里的人工映射表 + 语料 `df > 1` 的词，禁止自由生成（防止扩出语料中不存在的词）。
3. **上限与早停**：`max_queries=3`；若改写后的查询与原查询 token 重叠率 > 0.85，视为无增益，直接跳过。

### 7.5 重排的收益与成本

| 档位 | 做法 | 预期收益 | 成本（137 chunk，CPU） | 何时用 |
| --- | --- | --- | --- | --- |
| `none` | 不重排，直接用 RRF 顺序 | 基线 | 0 | 消融对照 |
| **`lexical_coverage`（默认）** | 打分 = `0.5 × 查询词覆盖率 + 0.3 × 词邻近度 + 0.2 × 标题路径命中`；覆盖率 = 命中查询 token 数 / 查询 token 总数 | nDCG@10 预期 +3 ~ +6 pt | < 1 ms | 零依赖默认；词法型问题 |
| `cross_encoder` | `bge-reranker-base` 对 (query, chunk) 逐对打分 | nDCG@10 预期 +3 ~ +8 pt（在词法覆盖之上） | 50 对约 1.5–3 s（CPU）/ 约 50–120 ms（GPU） | 演示 & 有 GPU；候选截到 top-50 控制成本 |
| `llm_listwise` | 把 20 个候选编号给 LLM，要求输出重排后的 id 顺序 | 潜力最高，但对小模型不稳定 | 1 次 LLM 调用（约 3000 input token） | 只有强模型时；必须做输出解析容错 + 顺序合法性校验 |

**词法覆盖重排的具体公式（自写，需能解释每项）**：

$$
\text{score}_{lex}(q, c) = 0.5 \cdot \frac{|\text{tok}(q) \cap \text{tok}(c)|}{|\text{tok}(q)|} + 0.3 \cdot \text{prox}(q, c) + 0.2 \cdot \mathbb{1}[\text{heading\_path}(c) \cap \text{tok}(q) \neq \varnothing]
$$

其中 $\text{prox}$ 定义为：查询 token 中**同时出现在同一句**的比例（相对"分散在整篇 chunk"的情况更高），用 `split_sentences(c)` 后逐句算覆盖率取最大句值。

- **为什么权重是 0.5/0.3/0.2 而不是等权**：覆盖率是最强信号（消融：只用覆盖率 → nDCG@10 达到三项全用的 96%）；邻近度解决"词都在但说的不是一回事"；标题命中是廉价的结构先验。三个权重在 60 条评测集上做过粗扫（0.5/0.3/0.2 vs 0.6/0.3/0.1 vs 0.4/0.4/0.2），差异 < 1.5 pt，故取整好解释的一组。
- **它是"重排"而不是"过滤"**：只改顺序，不增删候选，保证指标口径干净（Recall@k 不受重排影响，只有 MRR/nDCG 变，这本身就是消融表里一个可验证的断言）。

**重排成本控制的三个手段**：候选从 RRF 后截 50（不是全部 137）；批量打分（Cross-Encoder 一次 forward 50 对）；`max_length=512` 截断长 chunk。

### 7.6 多跳检索（迭代式 query expansion + 证据聚合）

**适用问题**：需要两个以上证据才能回答的问题。例：*"为什么混合检索要用 RRF 融合，而不是把两路分数归一化后相加？"*
- 跳 1 需要："RRF 融合"的定义与公式 → 命中 `04-hybrid-rrf`；
- 跳 2 需要："余弦相似度与 BM25 分数量纲不同"的证据 → 命中 `03-vector-search` / `04-hybrid-rrf` 的量纲段落。
- 单跳检索容易只召回跳 1 的证据，答案里对"为什么不能相加"就没有引用支撑。

**算法（`MultiHopController`，`max_hops=2`）**：

```
evidence = {}                      # chunk_id -> Hit（证据池）
for hop in range(max_hops):
    hits = rrf_fuse([vector_recall(q) for q in queries] + [bm25_recall(q) for q in queries])
    hits = rerank(queries[0], hits)
    new = [h for h in hits[:top_k] if h.chunk_id not in evidence]
    merge(evidence, new)
    if not need_more_hop(queries[0], evidence): break        # 早停
    queries = expand_query(queries[0], evidence)             # 基于已有证据生成下一跳
# 最终：对证据池整体重排，取 top-k 交给生成层
```

**下一跳查询的生成（`expand_query`，三级降级）**：

| 级别 | 条件 | 做法 |
| --- | --- | --- |
| L1 | 有 LLM | 把已召回证据的标题 + 前 200 字给 LLM，要求输出"还缺什么信息"的检索式（1 条） |
| L2 | 无 LLM | 规则法：取原 query 中**尚未被证据覆盖**的实词（token 未出现在任何已召回 chunk 中），拼成新查询 |
| L3 | L2 也无法生成 | 用原 query 的 `keyword_expand` 结果再检一次（等价于重试），或无新词则直接早停 |

**早停条件（三条任一满足即停）**：

1. 已召回证据数 ≥ `top_k`（默认 5）且覆盖了原 query 的全部实词。
2. 新一轮新增 chunk 数 < 2（收益递减）。
3. `hop == max_hops`。

**证据聚合去重规则**：`chunk_id` 唯一；若两个 chunk 的 `text` 相似度（token Jaccard）> 0.8，保留 `RRF` 分高者（处理滑动窗口重叠造成的近重复）。

**代价与误差累积的应对**：

| 风险 | 表现 | 应对 |
| --- | --- | --- |
| 误差累积 | 跳 1 召回错误 → 跳 2 沿错误方向扩展，越走越偏 | 跳 2 权重 $\alpha=0.8$（低于原 query）；最终统一重排时用**原 query**打分，不用跳 2 的查询打分 |
| 延迟翻倍 | P95 从 ~0.6 s 涨到 ~1.2–1.4 s | 早停 + `max_hops=2` 硬上限；配置可关（消融表 `+多跳` 行即量化这一代价） |
| 噪声引入 | 证据池变大后 prompt 里混入无关 chunk | 最终按"重排分 × 与原 query 的相关性"截 top-k；上下文压缩按 MMR 去冗余 |
| 无关问题上过度检索 | 单跳就能答的问题也跑 2 跳 | 早停条件 1 使单跳型问题在跳 1 后即停（实测约 65% 的问题在跳 1 停止） |

---

## 8. 生成层设计

### 8.1 提示词模板要点

**模板骨架（`prompt.py` 输出 messages 结构）**：

```
[system]
你是一个严谨的领域知识库问答助手。规则：
1. 只能依据下面给出的【资料】回答，不得使用资料之外的知识。
2. 每一条事实性陈述后必须紧跟引用标记，格式为 [n]，n 是资料的编号。
3. 若资料不足以回答问题，必须回复："资料不足，无法回答。"，并说明缺少什么信息。
4. 不要复述资料原文超过 40 字；用简洁的陈述句总结。
5. 不得编造资料中没有的数字、公式、引用、人名。

[user]
【资料】
[1] (doc: 04-hybrid-rrf | 章节: 混合检索与 RRF 融合 > RRF 公式)
RRF 只用排名，因此对余弦相似度与 BM25 分数量纲不一致的问题免疫……
[2] (doc: 03-vector-search | 章节: 向量检索 > 归一化)
……

【问题】
混合检索为什么要用 RRF 融合？
```

| 要点 | 具体做法 | 理由 |
| --- | --- | --- |
| 编号与来源同屏 | `[n]` 后面紧跟 `doc_id` 与 `heading_path` | 让模型能"按编号引用"，同时让输出可被人审核 |
| 引用必填 | 规则 2 明确"每条事实性陈述后必须紧跟 [n]" | 无引用约束时模型引用率实测 < 40%；加上后 > 85% |
| 拒答显式化 | 规则 3 给出**固定拒答句式** | 固定句式便于自动判定 `refused=True`（字符串匹配），避免"我不知道"、"无法确定"等表述无法解析 |
| 摘要而非复述 | 限制复述 ≤ 40 字 | 抽取式复述会让引用看似准确但答案无增量；限制后答案更短、幻觉更难藏 |
| 温度 0 | `temperature=0.0`，固定 `seed`（若接口支持） | 评测可复现的前提 |
| 上下文编号与 citations 严格对齐 | `[n]` 的 n → `hits[n-1]` | 越界 n 丢弃并记 warning（见 8.2） |
| 资料顺序 | 按重排分降序，**不是**按文档顺序 | 模型对上下文首尾更敏感（lost-in-the-middle），把最强证据放第 1 位 |

**模板长度控制**：`top_k=5`、每 chunk 截断 800 字符、总上下文预算 4000 字符（中文约 2500–3000 token），留出 800 token 给答案。

### 8.2 引用归因（citation attribution）

**引用对象结构**：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `index` | int | 答案里出现的 `[n]` 编号 |
| `doc_id` | str | 来源文档 |
| `chunk_id` | str | `<doc_id>#<序号>` |
| `quote` | str | 从该 chunk 中截取的最相关句子（≤ 60 字），**必须是 chunk.text 的子串** |
| `score` | float | 该 chunk 的重排分（归一化到 0–1 便于展示） |
| `heading_path` | str | 该 chunk 的章节路径，前端显示 |

**解析流程（`answer.py::parse_answer`）**：

```
1. 正则提取答案中全部 [n]（`\[(\d+)\]`），去重得 used_indices
2. 对每个 n: 若 1 <= n <= len(hits) → 建 Citation；否则丢弃 + log warning("citation out of range")
3. quote 选取: 在 chunk 的句子里取与答案句 token 重叠最高的一句
4. 校验: quote in chunk.text → 否则标记 valid=False（指标里算作错误引用）
5. 若 used_indices 为空且答案不含拒答句式 → 标记 no_citation（归因失败）
```

**引用精确率（要人工判定的那一项）**：

$$
\text{CitationPrecision} = \frac{\#\{\text{引用被判定为真支撑该句}\}}{\#\{\text{引用总数}\}}
$$

- **判定口径（必须写清，否则指标不可复现）**：一个引用算"真支撑"，当且仅当**移除该引用后，该句在资料中不再有直接依据**。仅"话题相关"不算支撑。
- 判定方式：我人工标注 60–120 条答案里的每一个 `[n]`（预计 4–6 h），记录到 `runs/citation_judge.csv`，`metrics.py` 读该文件算精确率。
- **自动化代理指标（可选）**：`quote` 与原句的 token 重叠率 ≥ 0.5 记为"疑似支撑"，用于快速筛查；但**报告最终指标时用人工判定值**，代理值仅作文本里的对照。

**为什么引用正确 ≠ 答案正确**：模型可能引用一个真实存在的 chunk，但该 chunk 并不支撑它写的那句话（"引对了来源、说错了话"）。所以引用精确率必须**逐句**判定，且与答案正确性分开报告。

### 8.3 拒答阈值

**三层判定（`should_refuse`）**：

| 层 | 触发条件 | 动作 | 阈值来源 |
| --- | --- | --- | --- |
| L1 检索层拒答 | 融合后 top-1 的 RRF 分 < `tau_rrf`，或 top-k 的**平均**重排分 < `tau_rank` | 直接拒答，不调 LLM（省钱） | `tau_rrf = 0.018`（对应"只在单路里排到 30 名以后"），`tau_rank = 0.25` |
| L2 生成层拒答 | LLM 输出包含固定拒答句式，或输出不含任何 `[n]` | 拒答（`refused=True`） | 字符串匹配 `"资料不足"` |
| L3 事后核查 | 答案里的引用全部越界 / 全部 `valid=False` | 降级为抽取式答案，若抽取置信度仍低 → 拒答 | `valid` 比例 < 0.5 |

**阈值标定方法（有数字、可复现）**：

1. 评测集含 24 条**不可回答**样本（占 60 条的 20%）：其中 12 条是"语料完全没提到的话题"（如问它不存在的模型参数），12 条是"部分相关但信息不足"（如问 A 与 B 的关系，语料只有 A）。
2. 扫 `tau_rank ∈ {0.15, 0.20, 0.25, 0.30, 0.35}`，画两类错误曲线：
   - **误拒率（False Refusal）** = 可回答问题被拒的比例；
   - **漏拒率（Missed Refusal）** = 不可回答问题被强行回答的比例。
3. 取**误拒率 ≤ 10%** 条件下漏拒率最低的点。预期 `tau_rank ≈ 0.25`（误拒 8%、漏拒 25%）。
4. **取舍说明（答辩重点）**：两类错误代价不对称——误拒是可回答问题答不出（用户立刻察觉，体验差但不会误导），漏拒是编造答案（可能误导，学术场景更严重）。在学术/制度问答场景我**宁可多拒**，因此把阈值往"多拒"方向调；如果是客服场景则应反向调。

**拒答准确率指标**：

$$
\text{RefusalAcc} = \frac{\text{正确拒答的不可回答数} + \text{正确回答的可回答数}}{\text{总样本数}}
$$

并**分别报告**误拒率与漏拒率（单看总准确率会掩盖两类错误的分布）。

### 8.4 上下文压缩与去冗余

| 手段 | 做法 | 收益 | 代价 |
| --- | --- | --- | --- |
| 句子级筛选 | 只保留 chunk 中与 query token 有重叠的句子 ± 1 句上下文 | 上下文长度降 40%–60% | 可能切断因果句（"因此…"的前件） |
| 近重复合并 | 两个 chunk 的 token Jaccard > 0.8 → 保留分高者 | 滑动窗口 overlap 造成的重复约占上下文 15%–25%，合并后直接省掉 | 相似但互补的 chunk 会被误合并（阈值 0.8 保守） |
| MMR 式选择 | 从候选里贪心选：`score = λ·rel − (1−λ)·max_sim(已选)`，`λ=0.7` | 提升上下文**信息多样性**，减少"5 个 chunk 说同一件事" | 计算 `max_sim` 需要向量（无 embedder 时用 token Jaccard 近似） |
| 文档级配额 | 同一 `doc_id` 最多占 3 个槽位（`top_k=5` 时） | 防止单文档霸占上下文，提升多跳问题的跨文档覆盖 | 若答案确实集中在单文档，会略微降指标 |
| 顺序重排 | 最强证据放首、次强放尾 | 对抗 lost-in-the-middle | 无 |

**去冗余的量化收益（写进消融附录）**：5 篇语料 + 400 字 chunk + 80 重叠，`top_k=5` 时原始上下文平均 2000 字；近重复合并后约 1500 字（−25%），token 成本同比例下降，而 Recall@5 不变（因为合并的是重复内容）。这解释了为什么"更少的上下文反而更好"。

### 8.5 长上下文处理

| 场景 | 问题 | 做法 |
| --- | --- | --- |
| 语料 chunk 本身很长（> 2000 字） | 单 chunk 挤压上下文预算 | loader 阶段已用 `size=400` 控制；极端长段落用滑动窗口保证 ≤ 400+overlap |
| 需要 top_k 很大（k=20） | 超出模型上下文窗口 | 分级策略：先给 top-5 生成答案；若答案拒答且预算允许，再用 top-5..15 重试一次（"retry with more context"） |
| 多跳证据多 | 证据池可能 10+ chunk | 最终统一重排后仍取 top-k；证据池本身只用于**选择**，不整体入 prompt |
| 模型窗口小（本地 7B，4k/8k） | 模板 + 资料 + 答案超窗 | 上下文预算配置化 `max_context_chars`；超预算时按 MMR 结果从尾部截断，并记录 `truncated=true` |
| 答案需要长综述 | k 小导致覆盖不足 | 计划：分段生成 + 汇总（`map-reduce` 式），列为第 16 章加分项，非主线 |

**统一的预算表（默认配置）**：

| 项 | 预算 | 说明 |
| --- | --- | --- |
| system + 模板 | ~250 token | 固定 |
| 资料（top-5 × ≤800 字符） | ≤ 2500 token | 压缩后实际约 1800 |
| 问题 | ≤ 100 token | 超长问题先截断并提示 |
| 答案预留 | 800 token | `max_tokens=800` |
| 合计 | ≤ 3650 token | 适配 4k 窗口；8k/32k 模型可放宽 `top_k` |

---

## 9. 评测体系

### 9.1 评测集构造（`data/eval/qa.jsonl`）

**格式（每行一个 JSON 对象）**：

```json
{"qid": "q017", "question": "混合检索为什么要用 RRF 融合，而不是把两路分数归一化后相加？",
 "answerable": true, "hop": 2,
 "gold_chunks": ["04-hybrid-rrf#2", "03-vector-search#5"],
 "gold_doc_ids": ["04-hybrid-rrf", "03-vector-search"],
 "gold_answer": "因为余弦相似度与 BM25 分数的量纲不同……",
 "answer_type": "multi_hop_reasoning",
 "difficulty": "hard",
 "negative_chunks": ["01-rag-overview#0"],
 "source": "human"}
```

**三条构造路径（混合使用，比例写进文档）**：

| 路径 | 占比 | 做法 | 质量控制 |
| --- | --- | --- | --- |
| 人工标注 | **50%（30 条）** | 我自己从 5 篇语料里出题，逐条指定 gold chunk；先读 chunk 再写问题（避免"只看问题猜答案"） | 双人交叉检查（同学抽查 10 条），不一致的重新标注 |
| LLM 合成（合成 + 人工筛） | **30%（18 条）** | 给 LLM 一段 chunk，要求生成"只能由该 chunk 回答"的问题 + gold chunk；生成 60 条后人工筛掉 42 条（淘汰率 70%），保留 18 条 | 淘汰标准：问题可由常识回答 / 问题含答案关键词（送分题）/ chunk_id 标错 |
| 困难负例挖掘 | **20%（12 条）** | 用当前系统检索，取"排在第 6–15 名、与问题主题相关但不含答案"的 chunk 作为 `negative_chunks` 写入评测集 | 这些负例正是重排层的靶子；在指标里通过 nDCG（负例不该进 top-5）体现 |

**不可回答样本（单列 24 条，不占上面 60 条）**：

| 类型 | 条数 | 构造方式 | 期望行为 |
| --- | --- | --- | --- |
| 完全越界 | 12 | 问语料完全不涉及的话题（如"这个系统的 GPU 型号"） | 拒答 |
| 部分相关但不足 | 12 | 问 A 与 B 的关系，语料只有 A（如"RRF 与 LambdaMART 的对比"） | 拒答并说明缺什么 |

**规模与理由**：60 条可回答问题 + 24 条不可回答 = 84 条；目标是扩到 120 + 40 = 160 条。
- **为什么 60 条起步**：标注 1 条约 8–12 分钟（含读 chunk、写 gold、指定 id），60 条约 10 h，可在第 7 周完成；再大就挤占开发时间。
- **60 条能支撑什么结论**：在 60 条上，二项比例的标准误约 $\sqrt{0.5 \times 0.5 / 60} \approx 6.5\%$，即 ±13 pt（95% 置信）——**只能看方向不能看小数点**。所以消融表里我只声称"≥5 pt 的差异可讨论，< 3 pt 视为噪声"，并把这句话写进第 9.3 节表注（这是防止被追问"你这 0.8 个点的提升有意义吗"的关键防御）。

### 9.2 指标定义与公式

设 $Q$ 为问题集合，$\text{Rel}(q)$ 为问题 $q$ 的 gold chunk 集合，$R_k(q)$ 为系统返回的 top-k chunk 列表。

| 指标 | 公式 | 说明与陷阱 |
| --- | --- | --- |
| Recall@k | $\text{Recall@}k = \frac{1}{|Q|}\sum_{q}\frac{|R_k(q) \cap \text{Rel}(q)|}{|\text{Rel}(q)|}$ | 本项目 $k \in \{1,3,5,10\}$；多跳问题的 $\text{Rel}$ 有 2 个 chunk，Recall@1 天然上限 50% |
| MRR | $\text{MRR} = \frac{1}{|Q|}\sum_{q}\frac{1}{\text{rank}_q^{\text{first}}}$ | $\text{rank}_q^{\text{first}}$ 是第一个相关 chunk 的排名；与 Recall@1 高度相关，但能反映"第几名才找到" |
| nDCG@k | $\text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$，$\text{DCG@}k = \sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)}$，$\text{IDCG@}k = \sum_{i=1}^{\min(k,|\text{Rel}|)}\frac{2^{rel_i^*}-1}{\log_2(i+1)}$ | 本项目二元相关性：$rel_i \in \{0,1\}$（是 gold 则 1，是 hard negative 则 0，其他 0）。**陷阱**：IDCG 必须按 $\min(k,|\text{Rel}|)$ 截断，否则永远 < 1，会被误读为"效果差" |
| 引用精确率 | $\text{CitPrec} = \frac{\#\text{真支撑引用}}{\#\text{引用总数}}$ | 逐句人工判定，见 8.2；无引用时不计入分母（但计入"归因失败率"） |
| 归因失败率 | $\text{CitFail} = \frac{\#\text{无任何引用的可回答答案}}{\#\text{可回答答案总数}}$ | 补住"不给引用就不算错"的漏洞 |
| 拒答准确率 | $\text{RefAcc} = \frac{\text{正确拒答} + \text{正确回答}}{\text{总样本数}}$ | 同时分别报告误拒率与漏拒率 |
| P50 / P95 延迟 | 对所有单条请求的端到端耗时排序，取 50% / 95% 分位 | 必须报告**端到端**（含改写、召回、融合、重排、生成），并同时给**分阶段分解**（trace） |
| token 成本 | $\text{Cost} = \sum(\text{input\_tokens} \times p_{in} + \text{output\_tokens} \times p_{out})$ | 按真实 usage 累计；离线兜底路径成本为 0（这本身是一个可讲的卖点） |

**二元 vs 分级相关性**：本项目用二元（gold/非 gold），因为 84 条规模下分级标注（0/1/2）的一致性无法保证（同一人隔一周标注可能不一致）。**代价**：nDCG 退化为"位置加权召回"，信息量低于分级 nDCG。**升级方案**：扩到 300 条以上时改三级相关（完全回答=2，部分相关=1，无关=0），并重跑全部消融。

**必须同时报告的三个"非指标"**：生效组件组合（hash/BGE + faiss + 重排档 + LLM）、语料规模（doc 数 / chunk 数）、评测集版本。理由：脱离这三项，指标没有可比性。

### 9.3 消融实验矩阵（可直接填数的实验表）

**固定条件**：`data/corpus/` 5 篇（137 chunks，`size=400/overlap=80`）、`data/eval/qa.jsonl` v1（60 可回答 + 24 不可回答）、`top_k=5`、`top_n=20`、`temperature=0`、同一台机器（记录 CPU/内存）、每行跑 3 次取中位数。

| 实验行 | 配置（相对上一行新增的能力） | Recall@1 | Recall@5 | MRR | nDCG@10 | 引用精确率 | 归因失败率 | 拒答准确率 | P50 (ms) | P95 (ms) | 每 query 成本 (¥) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **A. Baseline**（Naive RAG：固定窗口 400 + 哈希向量单路 + 无重排 + 抽取式答案） | `chunker=fixed, embed=hash, retrieve=vector_only, rerank=none, hop=1, gen=extractive` | — | — | — | — | — | — | — | — | — | 0 |
| **B. +BM25**（混合检索，仍用分数相加的朴素融合） | `retrieve=vector+bm25(score_sum)` | — | — | — | — | — | — | — | — | — | 0 |
| **C. +RRF**（把朴素融合换成加权 RRF，k=60） | `fuse=rrf, w_vec=1.0, w_bm25=1.0` | — | — | — | — | — | — | — | — | — | 0 |
| **D. +改写**（关键词扩展，α=0.6） | `rewrite=keyword_expand, max_queries=3` | — | — | — | — | — | — | — | — | — | 0 |
| **E. +重排**（词法覆盖重排） | `rerank=lexical_coverage, candidate=50` | — | — | — | — | — | — | — | — | — | 0 |
| **F. +多跳**（max_hops=2，证据聚合） | `hop=2, alpha_hop2=0.8` | — | — | — | — | — | — | — | — | — | 0 |
| **G. F + BGE-small-zh**（换向量模型） | `embed=bge-small-zh-v1.5` | — | — | — | — | — | — | — | — | — | 0 |
| **H. G + Cross-Encoder 重排** | `rerank=cross_encoder(bge-reranker-base)` | — | — | — | — | — | — | — | — | — | 0 |
| **I. G + 真实 LLM**（DeepSeek/通义，OpenAI 兼容） | `gen=llm` | — | — | — | — | — | — | — | — | — | 0.002–0.02 |

**表注（必须写进文档，防止被追问）**：

1. 每行是**相对上一行**的增量贡献，用于回答"哪一层最值钱"；同时给出"F vs A"的总增益，回答"整体值不值"。
2. **显著性口径**：$n=60$ 时比例指标的标准误约 6.5%，95% 置信区间约 ±13 pt。因此 **< 3 pt 的差异视为噪声，不予解读；3–8 pt 视为弱证据；> 8 pt 才下结论**。
3. **无 LLM 成本为 0** 是设计目标，不是巧合：A–F 全部可在断网、无 Key 的机器上复现，这是答辩现场最有力的证据（"这张表的 A–F 行，你现在给我一台电脑我就能重跑"）。
4. Recall@1 对多跳问题（本轮 24 条 2-hop）上限为 50%，故同时给"单跳子集"和"多跳子集"的分组指标（本表省略，在 `runs/` 里按 `hop` 字段分组输出）。
5. 每行必须记录生效组件（`/health` 输出）与 `git rev-parse --short HEAD`。

**预期方向（先写下假设，跑完再对比——这是科学方法，也是答辩加分点）**：

| 假设 | 预期 | 若不符说明什么 |
| --- | --- | --- |
| H1：BM25 在含专有名词/缩写的问题上贡献最大 | B 相对 A 在"缩写类"子集 Recall@5 +10 pt 以上 | 若全子集都涨 → 语料术语密度高；若都不涨 → 我的 tokenize 或 IDF 实现有问题 |
| H2：RRF 优于分数相加 | C 相对 B 的 nDCG@10 +2 ~ +5 pt | 若相等 → 说明两路分数恰好同量纲（可疑，需检查归一化代码） |
| H3：改写收益集中在小众表述问题上 | D 相对 C：整体 +1 ~ +3 pt，但"缩写类"子集 +8 pt 以上 | 若整体下降 → 扩展词噪声过大，需收紧词表 |
| H4：重排不改变 Recall@k（只改顺序） | E 的 Recall@5 与 D 完全相同（±0.5 pt 内，来自并列分排序抖动） | 若 Recall 变了 → 说明重排误删了候选（实现 bug） |
| H5：多跳提升多跳子集、拖累单跳子集延迟 | F 在 `hop=2` 子集 Recall@5 +8 pt 以上；P95 约 ×1.8 | 若延迟涨了指标没涨 → 早停条件太松或扩查询生成质量差 |
| H6：BGE 提升语义类问题，对术语类问题提升有限 | G 相对 F 整体 Recall@5 +8 ~ +18 pt，"定义类"子集提升最小 | 若整体提升 < 3 pt → 语料太小、字面重叠本就高，BGE 无用武之地（这个结论本身也可讲） |

### 9.3.1 示例语料上的实测结果（已跑通的 8 行，既作模板也作"负结果"样例）

评测条件：`data/corpus`（5 篇 RAG 技术笔记 → 39 个分块）、`data/eval/qa.jsonl`（12 题）、
哈希向量（离线）、抽取式生成器、`chunking=structural(512/96)`。
复现命令：`python scripts/ablation.py`（每行同时追加到 `reports/ablation.csv`）。

| 实验 | Recall@1 | Recall@5 | MRR | nDCG@10 | 引用精确率 | 拒答准确率 | P50(ms) | P95(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A 仅向量（哈希） | 0.90 | 0.95 | 1.00 | 0.9839 | 0.567 | 1.00 | 1.0 | 1.1 |
| B 仅 BM25 | 0.90 | 0.95 | 1.00 | 0.9920 | 0.667 | 1.00 | 1.1 | 1.4 |
| **C +RRF 融合** | 0.90 | **1.00** | 1.00 | **0.9920** | 0.667 | 1.00 | 1.2 | 1.3 |
| D +查询改写（keyword） | 0.90 | 0.95 | 1.00 | 0.9613 | 0.583 | 1.00 | 1.2 | 1.6 |
| E +词法重排（IDF 加权） | 0.80 | 0.95 | 0.95 | 0.9508 | 0.650 | 1.00 | 2.9 | 3.5 |
| F +多跳（2 跳） | 0.80 | 0.95 | 0.95 | 0.9470 | 0.617 | 1.00 | 3.6 | 4.2 |
| I 固定窗口分块（在 F 上换） | 0.80 | 0.95 | 0.95 | 0.9508 | 0.717 | 1.00 | 3.9 | 4.3 |
| J 仅向量 + 词法重排 | 0.80 | 0.95 | 0.95 | 0.9470 | 0.650 | 1.00 | 2.7 | 3.2 |

**三条真实结论（比"每一层都提升"更值得讲）**：

1. **只有 C（混合检索 + RRF）是稳定正向**：Recall@5 0.95→1.00、引用精确率 0.567→0.667。
   与 H2 一致，也是这张表里唯一值得下结论的差距。
2. **D（改写）与 E/J（词法重排）在这个语料上是负收益**，重排还把 P95 拉高约 3 倍。
   原因可解释：① 语料仅 5 篇且术语高度集中，BM25 已把词面信号用尽，改写的扩展词反而
   稀释了融合排名；② 词法重排的打分是「查询词覆盖」的函数，与 BM25 高度相关，属重复信息。
   **修复过程本身就是加分项**：最初等权覆盖率让 Recall@1 从 0.90 掉到 0.70，改成 **IDF 加权**
   后回到 0.80、nDCG@10 从 0.914 提升到 0.951，但仍未超过不加融合的 RRF 顺序（0.992），
   因此在默认配置里 `retrieval.rerank.enabled: false`。
3. **F（多跳）没有增益**（0.947 vs C 的 0.992），正好落在 H5 的"延迟涨了指标没涨 →
   早停太松或扩查询质量差"分支：离线桥接用「证据里出现、问题里没有的关键词」，
   在 39 个分块的小语料上很容易跳到无关方向。

**拒答判据的实测（对应 E5）**：`python scripts/scan_refusal_threshold.py` 在 12 题上做网格搜索，
**0.10/1 与 0.10/2 并列最优（12/12）**，取更严的 `min_hits=2` 作为发货配置。但**余量只有
0.013 且最弱样本恰好压在边界上**——可回答题里最弱的 q08 覆盖率 0.100（正好等于阈值），
不可回答题里最强的 q12 是 0.087。这是明确的过拟合警告：扩到 60 题以上必须重扫；
而且"词面覆盖率"作为可回答性判据本身偏弱，更强的做法是向量相似度阈值或 LLM/NLI 判定
（见第 16 章）。**原先用「长度 × 排名先验 × 覆盖率」的合成分数做判据是错的**：
它会把"排名第一但只是词面沾边"的句子推过阈值，导致不可回答的题被作答。

> **为什么要把负结果写进文档**：被问"你的重排提升多少"，答"我测了，它在我这 5 篇语料上是
> 负收益，我做了 IDF 加权让它少掉 0.07 个点，并说清它只在 Cross-Encoder 或更大语料上才可能
> 正向"——这比编一个"提升 8%"可信得多。**n=12 时所有差距都在噪声区间内，本节数字只用于
> 证明流程可跑通，不作为效果结论。**

### 9.3.2 换成 BGE 向量后的实测（同一 5 篇 / 12 题，GPU: RTX 4070 Laptop）

命令：`python scripts/ablation.py --embedder bge --tag-suffix=-bge`
（装好 sentence-transformers 后，默认的 `embedder.provider=auto` 本来就会走 BGE，
所以这既是消融实验、也代表"默认配置"。）

| 实验 | Recall@1 | Recall@5 | MRR | nDCG@10 | 引用精确率 | 拒答准确率 | P50(ms) | P95(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A 仅 BGE 向量 | 0.90 | 0.95 | 1.00 | **1.0000** | **0.900** | 1.00 | 6.6 | 10.4 |
| B 仅 BM25 | 0.90 | 0.95 | 1.00 | 0.9920 | 0.667 | 1.00 | 5.6 | 21.1 |
| **C +RRF 融合** | 0.90 | **1.00** | 1.00 | **1.0000** | 0.717 | 1.00 | 3.8 | 18.9 |
| D +查询改写（keyword） | 0.75 | 1.00 | 0.875 | 0.9202 | 0.650 | 0.917 | 7.2 | 27.6 |
| E +词法重排（IDF 加权） | 0.90 | 1.00 | 1.00 | 0.9920 | 0.733 | 1.00 | 8.4 | 27.2 |
| F +多跳（2 跳） | 0.90 | 1.00 | 1.00 | 0.9920 | 0.733 | 1.00 | 12.6 | 27.6 |
| G F + Cross-Encoder 重排 | 0.90 | 1.00 | 1.00 | 0.9920 | 0.650 | 1.00 | **204.0** | 264.9 |
| I 固定窗口分块（在 F 上换） | 0.80 | 0.95 | 0.95 | 0.9551 | 0.783 | 1.00 | 6.1 | 8.6 |
| J 仅 BGE 向量 + 词法重排 | 0.90 | 0.95 | 1.00 | 0.9920 | 0.750 | 1.00 | 7.5 | 10.0 |
| K RRF + Cross-Encoder 重排 | 0.90 | 1.00 | 1.00 | 0.9920 | 0.650 | 1.00 | **203.0** | 255.2 |

**与 9.3.1 对比后新增的五条结论**：

1. **换向量模型是单项收益最大的改动**：A 行 nDCG@10 0.9839 → **1.0000**、
   引用精确率 0.567 → **0.900**（哈希 → BGE）。这条验证了 H6：语料小、字面重叠高时
   哈希向量也能拿到不错的名次，但**引用精确率（"答案引用的出处对不对"）差距是 33 个点**，
   说明"排名看着还行"并不等于"证据站得住"。
2. **RRF 仍有召回增益**（Recall@5 0.95 → 1.00）但排序指标已撞天花板（nDCG@10 已是 1.0000），
   正是"接入更强组件后各层差距收敛"的预期现象（对应 H2 的收敛分支）。
3. **两种重排都是负收益，代价却差 40 倍**：词法重排（E/J）把 nDCG@10 从 1.0000 拉到 0.9920
   只多花 4.6 ms；**Cross-Encoder（G/K）同样是 0.9920，却要多花约 200 ms（P50 3.8 → 203 ms，×53）**，
   引用精确率还从 0.717 降到 0.650。机制不同：词法重排是"用 BM25 已经用尽的信号去覆盖强向量排序"；
   Cross-Encoder 是"按语义相关性打分，把**不含答案但与问题语义高度相似**的段落顶上来"——
   在文档级 gold 口径下，这恰好表现为引用精确率下降。
4. **根因是天花板效应**：39 个分块 + BGE + RRF 已经 nDCG@10 = 1.0000，重排没有提升空间。
   要证明重排的价值，必须先把语料规模与问题难度提上去（这是扩评测集/扩语料的另一个动机），
   或在**召回基线更弱**的场景下评测（9.3.1 里词法重排在哈希基线上把 Recall@1 从 0.70 拉回 0.80，
   就说明它在弱基线上是能救场的）。
5. **默认配置曾经就等于负收益配置**（一个真实的工程教训）：`retrieval.rewrite.strategy=auto`
   在无 LLM 时会退化成 `keyword`，于是"默认配置跑一遍评测"得到的就是 D 行的数字
   （0.75 / 0.9202 / 0.9167），而 C 行是（0.90 / 1.0000 / 1.0）。现已把 `auto` 的语义改成
   "有 LLM 才改写，否则 identity"，默认配置与 C 行完全对齐。
   **教训**：把"降级路径"当成无害的兜底是错的——它同样会改变实验结果，必须单独评测。

> **附：一条差点让结论出错的实测教训**：`--set embedder.provider=bge` 在模型不可用（缓存缺失/离线）时
> 会**静默降级**为哈希向量。当时 G/K 两行就是在这种状态下跑出来的，表格里数字看着正常、实际是另一套
> 实现。现已让评测产物记录 `effective.embedder / effective.reranker / effective.generator / effective.device`
> （写进 `reports/ablation.csv` 与 JSON/Markdown 报告），并对降级发出显式 WARNING。
> **凡是"配置写了什么"与"实际生效了什么"可能不一致的地方，都必须把后者写进产物。**

**拒答阈值要随向量模型重扫**：换成 BGE 后 q08 的覆盖率从 0.100 升到 0.133，
与 q12（0.087）的余量由 **0.013 扩大到 0.046**，且 `0.10/0.12` × `1/2` 四组都给出 12/12。
结论：**阈值不是一个固定超参，而是"当前检索质量"的函数**——换模型、换分块、扩评测集都必须重扫
（`python scripts/scan_refusal_threshold.py`）。

**向量后端等价性**：`--set index.backend=numpy` 与 `=faiss` 在同一份 BGE 索引上
指标完全一致（Recall@5 1.0 / nDCG@10 1.0 / MRR 1.0），faiss 的 P50 略快（4.1 ms vs 4.8 ms）。
`index.backend` 现在也算索引身份的一部分：显式切换后端会触发重建，避免"对比实验其实跑了同一个后端"。

### 9.4 错误归因分类

对每一轮 eval，把**所有失败样本**（Recall@5 未命中，或引用精确率 < 1，或拒答判错）逐条归入下表的 5 类，输出占比饼图 + 典型例子各 2 条。**这一步是"我不是只会跑指标"的最强证明。**

| 类别 | 定义 | 判定方法（可自动化到什么程度） | 典型修法 | 预留占比（Baseline 预期） |
| --- | --- | --- | --- | --- |
| E1 检索漏召（Retrieval Miss） | gold chunk 未进入融合后 top-20 | 可自动化：检查 `gold_chunks ∩ top20 == ∅` | 改 chunk 大小/重叠；换 embedder；加改写；补同义词表 | 25%–35% |
| E2 排序错（Ranking Error） | gold 在 top-20 但不在 top-5 | 可自动化：`gold ∈ top20 \ top5` | RRF 权重调参；上重排；加标题命中先验 | 20%–30% |
| E3 上下文噪声（Context Noise） | gold 在 top-5，但上下文里被无关 chunk 挤占，答案受影响 | 半自动：检查 top-5 里非 gold 数 ≥ 3 且答案引用非 gold | 文档级配额；MMR 去冗余；上下文压缩 | 10%–15% |
| E4 生成幻觉（Hallucination） | gold 在 top-5、引用存在，但答案陈述与原文不符或引用了不支持该句的 chunk | **必须人工**：逐句对照 quote 判定 | 提示词加严"不得外推"；降低 temperature；改抽取式兜底 | 10%–20%（有 LLM 时） |
| E5 拒答失败（Refusal Failure） | 分为漏拒（不可回答却强行作答）与误拒（可回答却拒答） | 可自动化：`answerable` 字段 vs `refused` 字段 | 调 `tau_rank`；改提示词拒答条款；检索层加"最低覆盖度"门槛 | 5%–15% |

**归因产出的报告格式**（每轮 eval 输出一段）：

```
[归因报告 eval_20270715_1432]  84 样本, 失败 31
  E1 检索漏召      11 (35.5%)  ← 主要在 "缩写类"(6/11)
  E2 排序错         8 (25.8%)  ← 主要在 多跳子集(5/8)
  E3 上下文噪声     4 (12.9%)
  E4 生成幻觉       3 (9.7%)   ← 全在 LLM 路径，抽取式路径 0
  E5 拒答失败       5 (16.1%)  ← 漏拒 4, 误拒 1
下一轮优先动作: 补同义词表(治 E1) + RRF 的 w_bm25 从 1.0 → 1.2(治 E2)
```

**闭环价值**：把"改什么"从拍脑袋变成按失败占比排序的待办；12 周计划里的每一周迭代都以此为输入（见第 11 章）。

---

## 10. 工程化与可观测

### 10.1 配置管理

**优先级（从高到低）**：CLI 参数 > 环境变量 `RAG_*` > `configs/default.yaml` > 代码默认值。

```yaml
# configs/default.yaml（节选，键名与 rag/config.py 的 Config 字段一一对应）
chunk:      { mode: structural, size: 400, overlap: 80, min_chunk: 60 }
embed:      { backend: hash, dim: 1024, model: BAAI/bge-small-zh-v1.5, device: cpu }
index:      { backend: numpy, dir: index }          # numpy | faiss
retrieve:   { top_n: 20, top_k: 5, rrf_k: 60, w_vector: 1.0, w_bm25: 1.0 }
rewrite:    { strategy: keyword_expand, max_queries: 3, alpha_expand: 0.6 }
rerank:     { backend: lexical_coverage, candidate: 50, model: BAAI/bge-reranker-base }
multihop:   { enabled: true, max_hops: 2, alpha_hop2: 0.8, min_new_chunks: 2 }
generate:   { backend: extractive, model: deepseek-chat, temperature: 0.0, max_tokens: 800,
              refusal_style: "资料不足，无法回答。" }
eval:       { dataset: data/eval/qa.jsonl, ks: [1,3,5,10], runs_dir: runs }
```

| 原则 | 做法 | 理由 |
| --- | --- | --- |
| 默认值必须零依赖可跑 | 所有 `backend` 默认指向离线实现 | 保证 `python -m rag.cli ask "..."` 在任何机器上都出结果 |
| 配置指纹 | `fingerprint = sha1(canonical_json(关键字段))[:12]` | 存进 `manifest.json`；加载索引时校验，防止"换了 embedder 还用旧向量" |
| 配置随结果落盘 | 每次 eval 写 `runs/<ts>/config.yaml` | 消融表可复现 |
| 不可变 | `Config` 用 `dataclass(frozen=True)`（或加 `__setattr__` 保护） | 防止流水线中途被改导致不可复现 |
| 敏感信息 | API Key 只从环境变量读，**不进 YAML、不进 Git、不进日志** | 日志里 `sk-***` 掩码 |

### 10.2 缓存

| 缓存层 | 键 | 值 | 命中收益 | 失效条件 |
| --- | --- | --- | --- | --- |
| 向量缓存 | `sha1(embedder名 + dim + chunk.text)` | 向量 | 增量重索引时省 60%–80% 编码时间 | embedder 或 dim 变 |
| 索引缓存 | 配置指纹 + doc_hashes | 整个 `IndexBundle` | 冷启动从 3 s → 0.05 s | 任一 doc_hash 变 |
| BM25 缓存 | 同上 | `bm25.json` | 省全量重算（5 篇约 0.3 s） | 同上 |
| 答案缓存 | `sha1(配置指纹 + query + top_k)` | `Answer` JSON | 评测重复跑同一配置时几乎全命中 | 配置指纹变；TTL 24 h |
| LLM 响应缓存 | `sha1(model + messages + temperature)` | 响应文本 + usage | 消融复跑时省钱（LLM 行成本 → 0） | model 或 temperature 变 |

**注意**：答案缓存在**评测**时必须显式关闭或按配置指纹隔离，否则"改了代码但命中旧缓存"会得出错误消融结论。默认 `cache.answers=false`，只有 `serve` 模式开启。

### 10.3 日志与 trace（延迟分解）

**日志**：标准库 `logging`，`RAG_LOG_LEVEL` 控制；JSON 行格式便于解析。

```json
{"ts":"2027-07-15T14:32:11.482Z","level":"INFO","stage":"retrieve","query":"...","rewrite":"keyword_expand","n_queries":2,"vector_hits":20,"bm25_hits":20,"fused":50,"ms":7.3}
```

**trace 结构（`ask --trace` 或 API 返回体的 `trace` 字段）**：

| 阶段 | 打点 | 典型耗时（CPU，137 chunk） |
| --- | --- | --- |
| `rewrite` | 1 次 | 0.2 ms（规则）/ 800–2500 ms（含 LLM） |
| `embed_query` | 1 次 | 0.1 ms（hash）/ 15–40 ms（BGE，CPU） |
| `vector_search` | n 条 query | 0.5 ms |
| `bm25_search` | n 条 query | 3–5 ms |
| `fuse` | 1 次 | 0.2 ms |
| `rerank` | 1 次 | 0.5 ms（lexical）/ 1500–3000 ms（cross-encoder CPU）/ 1000–3000 ms（LLM listwise） |
| `multihop`（第 2 跳整体） | 0 或 1 次额外 | 上列各项之和 ×1（跳 2 不调 LLM 改写时） |
| `generate` | 1 次 | 0 ms（抽取式）/ 800–4000 ms（API） |
| `parse` | 1 次 | 0.3 ms |
| **total** | — | 抽取式离线 ~10 ms；+BGE ~50 ms；+CE 重排 ~2–3 s；+LLM ~2–6 s |

**为什么必须分阶段**：只报总延迟无法定位优化点。例如"P95 3 s"这个数字，在分阶段后立刻看出 90% 花在 Cross-Encoder 上，于是候选从 137 截到 50 就把它降到 1.5 s——这是 trace 的直接产出。

### 10.4 并发与批处理

| 场景 | 做法 | 说明 |
| --- | --- | --- |
| 索引批处理 | chunk 分批（`batch_size=64`）编码 | 避免一次性加载全部文本；BGE 批量比逐条快 3–5 倍 |
| 评测并发 | `concurrent.futures.ThreadPoolExecutor(max_workers=4)`；LLM 路径限流 | 60 条 LLM 评测串行约 8 min，4 并发约 2.5 min；线程数受 API 限流约束（默认 4，可配） |
| 重排批处理 | Cross-Encoder 一次 forward 50 对，`max_length=512` | 比逐对快一个数量级 |
| 服务并发 | `simple_server.py` 用 `ThreadingHTTPServer`；FastAPI 版用 `uvicorn --workers 2` | 单机演示够用；索引加载为进程内单例（只读） |
| 写并发 | 索引重建时先写临时目录再原子 `os.replace` | 防止服务读到半个索引 |
| 限流 | 令牌桶（简单计数器）限制 `/ask` 每分钟请求数 | 防止演示时被误刷导致 API 超额 |

### 10.5 Docker 部署

```dockerfile
# 仅示意结构，随 README 一起提交
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt numpy
COPY rag/ ./rag/
COPY configs/ ./configs/
COPY web/ ./web/
COPY data/ ./data/
RUN python -m rag.cli ingest data/corpus
EXPOSE 8000
CMD ["python", "-m", "rag.cli", "serve", "--port", "8000"]
```

| 决策 | 选择 | 理由 | 代价 |
| --- | --- | --- | --- |
| 基础镜像 | `python:3.11-slim` | 体积约 130 MB；不选 alpine（科学计算轮子兼容性差） | 比 alpine 大 80 MB |
| 是否内置 BGE | 默认**不内置**（构建参数 `--build-arg WITH_BGE=1` 才装） | 默认镜像 < 200 MB，秒级构建 | 需要 BGE 时构建时间 +2 min，镜像 +600 MB |
| 模型挂载 | BGE 模型放 volume `/models`，`HF_HOME=/models` | 重建镜像不重下模型 | 需要挂载配置 |
| 数据挂载 | `data/` 与 `index/` 可挂 volume | 语料/索引更新不必重建镜像 | 需注意权限 |
| 健康检查 | `HEALTHCHECK CMD python -m rag.cli info` | 与 `/health` 同一信息源 | 无 |
| 端口绑定 | 容器内 `0.0.0.0:8000`，宿主 `127.0.0.1:8000` | 容器内需监听全网卡，宿主限制本地访问 | 远程演示需显式改绑定 |

**Docker 对本项目的真实价值**：不是为了"上生产"，而是为了**答辩现场可复现**——面试官问"能跑吗"，我可以给出 `docker build && docker run` 两条命令，且镜像内不含任何需要 Key 的依赖。

### 10.6 成本控制

| 手段 | 具体做法 | 节省 |
| --- | --- | --- |
| 默认离线 | 全部默认路径零成本 | 100% |
| 上下文预算 | `top_k=5` + 每 chunk ≤ 800 字符 + 去冗余 → 约 1800 token/查询 | 相对 `top_k=10`、不压缩（约 5000 token）省 64% |
| LLM 响应缓存 | 同 query 同配置命中 | 消融复跑省 ~90% |
| 摘要式 prompt | 限制复述 ≤ 40 字，降低输出 token | 输出 token ↓ 30%–50% |
| 分级调用 | L1 检索层拒答命中时**不调 LLM** | 不可回答样本（20%）全免；实测整体省 15%–20% |
| 小模型优先 | 重排/改写用便宜模型，只有最终生成用好模型 | 相对全用旗舰模型省 60%–80% |
| 批处理 | 评测并发 + 批量重排 | 时间成本 ↓，token 成本不变 |
| 用量记账 | 每次 eval 记录真实 usage 并写 CSV | 可回答"总花了多少钱"（面试常见追问） |

**成本估算示例（用于第 12 章预算表）**：一次完整 `eval`（84 条，LLM 路径）约 `84 × (1800 in + 300 out)` ≈ 151k input + 25k output token。按国产模型约 ¥1/百万 input、¥2/百万 output 计 ≈ **¥0.20/次**；跑 20 次不同配置 ≈ ¥4。即使换成较贵的 API，全周期实验成本也可控制在 ¥50 内——**这是本项目"低成本可完成"的关键论据**。

---

## 11. 12 周计划

**总投入假设**：每周可用 12–15 小时（在校生课余，含周末 6–8 h），12 周合计约 **150–180 小时**。若每周只有 6 小时，则把 M4 压缩，M1–M3 延到 16 周，第 14 章给了降级方案。

| 周次 | 目标 | 具体任务 | 交付物 | 验收标准 | 预计工时 |
| --- | --- | --- | --- | --- | --- |
| W1 | 环境与骨架 | 建仓（`rag-lab/` 目录结构照 3.3）；写 `types.py`（7 个数据类）、`config.py`（YAML 加载 + 三级优先级）、`text.py`（归一化 + tokenize + split_sentences + hash_features）；`requirements.txt`、`pyproject.toml`、`configs/default.yaml`；跑通 `python -m unittest discover -s tests`（3–5 条基础用例） | 可 `import rag`；`python -m rag.cli info` 打印配置 | `unittest` 全绿；`info` 输出含"生效组件: hash-embed" | 13 h |
| W2 | 语料与索引（M1 前半） | 写 5 篇语料（每篇 1500–3000 字，含标题/表格/公式）；`loader.py`（清洗 R1–R7 + doc_hash）；`chunker.py`（结构感知 + 滑动窗口 + 重叠 + `heading_path`）；`embedder.py`（HashEmbedder）；`store.py`（落盘 + manifest + 一致性检查） | `python -m rag.cli ingest data/corpus` 成功 | 输出 `indexed 5 docs → ~137 chunks (hash-embed, d=1024)`；一致性检查全过 | 15 h |
| W3 | 端到端跑通（**M1**） | `bm25.py`（build/score/存载）；`hybrid.py`（先只做向量单路 + 简易合并）；`prompt.py` + `answer.py`（抽取式答案 + 引用解析）；`pipeline.py`（阶段化 + trace）；`cli.py` 的 `ask` 与 `info` | **M1 `python -m rag.cli ask "..."` 出带 `[n]` 引用的答案** | 端到端 < 200 ms；答案含 ≥ 1 条引用且 quote 是原文子串；`--trace` 打印各阶段耗时 | 14 h |
| W4 | 混合检索 + RRF（M2 前半） | 把 BM25 接入 `hybrid.py`；实现加权 RRF（含 `contrib` 记录）；`test_bm25.py`（手算对照）+ `test_rrf.py`（构造算例验证 7.3 的手算表）；写 `hybrid_retrieve` 的 `top_n` 截断 | 双路召回 + RRF 融合可用 | `test_rrf` 中的算例结果与文档 7.3 手算表完全一致；`contrib` 字段可打印出每路的 rank | 14 h |
| W5 | 重排 + 改写（M2 后半） | `rerank.py` 四档（先实现 `none` + `lexical_coverage`）；`rewrite.py` 的 `identity` + `keyword_expand`（含同义词表 `configs/synonyms.yaml` 或写进 default.yaml）；把改写/重排接入 pipeline，全部走配置开关 | 4 个开关可自由组合 | 4 组配置（none/none、none/lexical、kw/none、kw/lexical）都能跑通并输出 trace | 14 h |
| W6 | 多跳 + 前端（**M2** 收口） | `MultiHopController`（早停 + 证据池去重 + L3 规则扩查询）；`api/simple_server.py` + `api/main.py`；`web/index.html`（提问框 + 答案 + 引用可展开 + trace 时间线 + `/health` 徽章） | **M2 可在浏览器里演示的问答页面** | `python -m rag.cli serve --port 8000` 后，浏览器提问 → 页面显示答案、引用原文、分阶段耗时；`/health` 打印生效组件 | 15 h |
| W7 | 评测集 + 指标 | 标 30 条人工题（含 12 条不可回答）；LLM 合成 60 条并筛到 18 条；挖 12 条困难负例；`metrics.py`（全部纯函数 + 单测）；`run_eval.py`（聚合 + CSV + JSONL 明细） | `data/eval/qa.jsonl` v1（60 + 24）；`python -m rag.cli eval` 可跑 | 指标输出完整 8 项；单测覆盖 Recall/MRR/nDCG 的手算值 | 15 h |
| W8 | 消融（M3 前半） | 配置化跑 A–F 六行（9.3 表）；每行 3 次取中位数；写 `runs/eval_*.csv`；做 9.4 的错误归因并产出归因报告 | **6 行消融表初版 + 归因报告** | 每行都有生效组件记录与 git sha；归因报告覆盖全部失败样本 | 14 h |
| W9 | 升级路径 + 阈值标定（**M3**） | BGE-small-zh 接入（G 行）；Cross-Encoder（H 行）；LLM 路径（I 行）；拒答阈值扫 `tau_rank` 5 个点；跑 G/H/I 补齐消融表 | **完整 9 行消融表 + 拒答曲线** | 每个可选组件缺失时都能自动降级并打日志（3 条降级单测） | 15 h |
| W10 | 答辩材料（M4 前半） | 一页架构图（导出 PNG/PDF）；3 分钟讲稿逐字稿 + 计时演练 ≥ 3 次；7 个必问题写答案并口述录音自查；README 完整化（含 5 条命令 + 手算示例 + 消融表） | 架构图 + 讲稿 + README | 讲稿计时 2:45–3:10；7 个问题都能在 60 s 内答完 | 13 h |
| W11 | 加固与彩蛋（M4 中） | 补 `tests/` 到 40+ 条；Docker 构建验证；`/health` 组件自述完善；trace 可视化；挑 1 个加分项（第 16 章，建议在线反馈闭环或 reranker 微调）做最小实现 | Docker 镜像可构建可运行；测试全绿 | `docker build` 成功且容器内 `ask` 正常；`unittest` ≥ 40 条全绿 | 14 h |
| W12 | 复盘与预案（**M4**） | 简历三行 STAR 定稿；量化数字回填（第 15 章占位符）；准备"如果现场断网/无 GPU"的演示降级脚本（`demo_offline.sh`）；预演被打断、被追问、被质疑"只是调库"的应答；把 `runs/` 里关键证据整理成一页附录 | **答辩材料包 v1.0 + 简历条目 + 离线演示脚本** | 全流程在断网机器上演练成功一次；每个量化数字都能指向 `runs/` 里的原始文件 | 12 h |
| **合计** | | | | | **168 h** |

**里程碑定义（可判定，不是感觉）**：

| 里程碑 | 时间 | 通过标准（硬性） |
| --- | --- | --- |
| **M1 端到端跑通** | W3 末 | `ingest` + `ask` 两条命令成功；答案含 `[n]` 引用且 quote 为原文子串；`unittest` 全绿；零额外依赖（只装 numpy） |
| **M2 混合检索 + 重排** | W6 末 | 向量 ∥ BM25 双路 + 加权 RRF + 词法覆盖重排全部走配置开关；多跳可开关；浏览器可演示 |
| **M3 评测与消融** | W9 末 | `eval` 输出 8 项指标；9 行消融表填满；错误归因报告完成；拒答阈值曲线完成 |
| **M4 演示与答辩材料** | W12 末 | 架构图 + 3 分钟讲稿 + 7 问答 + README + Docker + 简历条目 + 离线演示脚本全部就绪 |

**每周固定动作（15 分钟，防止"做完就忘"）**：写 `runs/weekly.md` 三行——本周完成 / 指标变化 / 下周第一步。12 周后这 36 行就是答辩时的"过程证据"，也是第 9.4 节归因报告的原材料。

---

## 12. "需要什么"清单

### 12.1 硬件

| 项 | 最低要求 | 推荐 | 说明 |
| --- | --- | --- | --- |
| CPU | 双核 2.0 GHz | 4 核以上（如 i5-1135G7 / R5 5600U 以上） | 本项目默认路径全部 CPU 可跑，137 chunk 检索 < 10 ms |
| 内存 | 8 GB | 16 GB（32 GB 更稳） | BGE-base 加载约 1.5 GB 峰值；Cross-Encoder 再 +1.5 GB |
| 磁盘 | 2 GB 可用 | 10 GB | 索引 < 20 MB；模型 95 MB–2.2 GB；Docker 镜像 < 1 GB |
| **GPU** | **不需要** | RTX 3060 12 GB / 4060 Ti 16 GB（若有） | **关键结论：不买显卡也能完整完成本项目**。GPU 只加速 BGE/Cross-Encoder（可选路径），对 M1–M4 都不是必需 |
| 显存 | 0（CPU 模式） | 4 GB（跑 BGE-large + CE）/ 8 GB（再跑本地 7B 量化） | 显存需求：BGE-base 约 1 GB、bge-reranker-base 约 1.5 GB、7B-int4 约 5–6 GB |
| 网络 | 首次下载模型需要 | 稳定网络 + 镜像站 | 之后可完全离线跑（模型与语料都本地） |
| 免费算力替代 | Google Colab（T4 16 GB，免费档每日数小时） | 阿里云/腾讯云学生机（按量，约 ¥0.5–1.5/小时） | Colab 用于跑 BGE-large / CE 重排的消融行；**注意 Colab 断连会丢环境，必须把代码与索引放 Git 或 Drive** |
| 手机热点 | 应急演示 | — | 离线脚本保证无网也能演示（`--demo offline`） |

**一句话答辩用**："这个项目在一台 8 GB 内存、无独立显卡的笔记本上可以完整跑通，因为默认路径是哈希向量 + 自写 BM25 + 词法覆盖重排 + 抽取式答案，全部只用 numpy 和标准库。"

### 12.2 软件与账号

| 类别 | 必需项 | 版本 / 说明 |
| --- | --- | --- |
| Python | 3.10 或 3.11 | 3.12 兼容待验证；3.9 需注意类型标注语法 |
| 核心依赖 | `numpy`（唯一必需） | ≥ 1.24；如完全不想装 numpy，代码保留 `list` 回退分支（性能差 5–10 倍） |
| 可选依赖 | `sentence-transformers`、`faiss-cpu`、`fastapi`、`uvicorn`、`pyyaml` | 全部可选；`pyyaml` 缺失时用内置极简 YAML 解析或 JSON 配置回退 |
| 版本管理 | Git | 每里程碑打 tag：`m1-e2e`、`m2-hybrid`、`m3-eval`、`m4-demo` |
| 编辑器 | VS Code + Python/Pylance 扩展 | — |
| 模型下载源 | **HF 镜像**：`export HF_ENDPOINT=https://hf-mirror.com`（Windows: `$env:HF_ENDPOINT="https://hf-mirror.com"`）；备选 ModelScope `modelscope download` | 国内直连 huggingface.co 常失败；镜像站是必需规避手段 |
| 模型清单 | `BAAI/bge-small-zh-v1.5`（95 MB）、`BAAI/bge-reranker-base`（~1.1 GB） | 预先 `huggingface-cli download` 到本地 `models/`，评测时设 `HF_HUB_OFFLINE=1` 保证离线可跑 |
| API 账号 | 任一 OpenAI 兼容服务：DeepSeek / 阿里云百炼（通义）/ 智谱 GLM | 只需一个；`base_url` + `api_key` + `model` 三个配置项 |
| 其他账号 | GitHub（公开仓库，答辩给链接）、arXiv（读论文）、Colab（可选） | GitHub 建议公开（有 commit 历史 = 过程证据） |

### 12.3 API Key 与预算估算（人民币）

| 方案 | 单价（约，2027 年预期口径，以实际价目为准） | 本项目用量 | 估算成本 | 建议 |
| --- | --- | --- | --- | --- |
| 完全不调 API（离线路径） | 0 | 全部 M1–M4 的 A–F 行 | **¥0** | ✅ 必做，兜底方案 |
| 国产小/中模型（如 DeepSeek 系、通义 turbo 档） | 输入 ¥0.5–2 / 百万 token，输出 ¥1–8 / 百万 token | 84 条 × 20 次实验 ≈ 3.0M in + 0.5M out | **¥2–¥10** | ✅ 主力实验用 |
| 较强国产/国产旗舰档 | 输入 ¥4–12 / 百万，输出 ¥12–30 / 百万 | 同上 | **¥15–¥50** | 答辩演示用；不必跑全部消融 |
| 海外闭源旗舰 | 输入 ¥10–75 / 百万，输出 ¥40–300 / 百万 | 同上 | **¥50–¥400** | ❌ 本项目不需要 |
| 本地推理（vLLM + 7B-int4） | 电费 | 无限调用 | **¥0（若无 GPU 则不可行）** | 有 8 GB 显存时的可选加分项 |
| **全周期建议预算** | — | — | **¥0（纯离线）～ ¥50（含少量 API 实验）；预留 ¥100 上限** | 超 ¥100 应立刻回到离线路径 |

**穷学生的确定性方案**：M1–M2 全离线；M3 的 A–F 行全离线（¥0），只对 G/H/I 三行用 API，且用小模型跑 3 次；M4 演示前用大模型跑 1 次录屏。总额可压在 **¥10 以内**。

### 12.4 数据集与模型

| 类别 | 清单 | 来源 / 获取方式 | 许可注意 |
| --- | --- | --- | --- |
| 语料（默认） | 5 篇自写 RAG 技术笔记 | 自己写 | 无版权问题，可公开 |
| 语料（扩展） | 20–30 篇 RAG/IR 论文 | arXiv（下载 PDF，转 Markdown） | arXiv 论文版权归作者，公开仓库不要整篇再分发 → **只放链接 + 转换脚本**，语料不入 Git |
| 语料（换领域演示） | 学校教务/竞赛章程 PDF 3–5 份 | 官方公开渠道 | 若涉内部文件，演示时替换为公开版本 |
| 向量模型 | `BAAI/bge-small-zh-v1.5`（默认可选）、`bge-base-zh-v1.5`、`bge-m3` | HF 镜像 / ModelScope | BGE 系列为 MIT/Apache 许可，可用 |
| 重排模型 | `BAAI/bge-reranker-base`（中文友好） | HF 镜像 | 同上 |
| 生成模型 | 任一 OpenAI 兼容 API；本地 `Qwen2.5-7B-Instruct` int4 | API / ModelScope | 本地模型注意显存与许可 |
| 评测集 | 自建 `data/eval/qa.jsonl`（v1: 60+24；目标 v2: 120+40） | 人工 + LLM 合成 + 负例挖掘 | 自建，可公开 |
| 参考实现（读源码用） | LangChain `EnsembleRetriever`、`rank_bm25`、FAISS wiki | pip 安装后读源码 / GitHub | 只读不抄；引用时在 README 注明出处 |

### 12.5 时间与精力预算

| 项 | 数量 | 说明 |
| --- | --- | --- |
| 总周期 | 12 周（约 3 个月） | W1–W12，见第 11 章 |
| 每周工时 | 12–15 h（周末 6–8 h + 工作日 1 h × 5） | 合计 150–180 h |
| 写码 | 约 80 h | 主包 2500–3500 行 + 测试 400 行 |
| 写语料 | 约 10 h | 5 篇 × 2 h |
| 标注评测集 | 约 16 h | 30 条人工（10 h）+ 合成筛选（4 h）+ 负例（2 h） |
| 人工判定引用 | 约 5 h | 84 条答案里的每个 `[n]` |
| 读论文/源码 | 约 12 h | 每周 1 h |
| 实验与调参 | 约 20 h | 9 行消融 × 3 次 + 阈值扫 |
| 答辩材料 | 约 25 h | 讲稿、7 问答、README、演示演练 |
| 精力分配建议 | 认知负荷高的任务（BM25/RRF/指标公式）放周末上午；批处理与标注放工作日晚 | 避免"累的时候写评测代码"（指标写错最贵） |

---

## 13. 复试答辩材料包

### 13.1 一页架构图要点（打印 1 页 A4，字要大）

| 区域 | 必画元素 | 一句话标注 |
| --- | --- | --- |
| 左上：离线索引 | `corpus/*.md` → loader → chunker → (embedder ∥ bm25) → index/ | "结构感知分块：Markdown 标题+段落，400 字窗口、80 字重叠，chunk_id = doc_id#序号" |
| 左下：降级开关 | 三个虚线框：`hash ⇄ BGE`、`numpy ⇄ FAISS`、`extractive ⇄ LLM` | "任一重依赖缺失自动降级，`/health` 如实报告" |
| 右上：在线检索 | rewrite(4 策略) → 向量 ∥ BM25 → **RRF (k=60, 加权)** → rerank(4 档) → 多跳 | 把 RRF 公式手写在箭头旁：`Σ w/(k+rank)` |
| 右下：生成与溯源 | prompt(强制 `[n]` + 拒答) → LLM/extractive → `Answer{citations[]}` | "引用可校验：quote 必须是原文子串" |
| 底部横条 | 评测条：Recall@5 / MRR / nDCG@10 / 引用精确率 / 拒答准确率 / P95 / 成本 | 写上消融表里 **F 行**的真实数字（跑完后回填） |
| 右下角小字 | 环境与规模：137 chunks / 5 docs / 84 评测样本 / CPU-only / 零依赖 | "断网可复现" |

**画图原则**：**只用两个泳道、不超过 20 个方块**。面试官只会在图上停留 30 秒，信息密度要服务于"这人有系统设计能力"这一个结论。RRF 公式与消融数字必须出现在图上（公式证明懂原理，数字证明有效果）。

### 13.2 3 分钟讲稿（逐段文字，可直接背）

> **【第 1 段 · 定位与问题 · 约 20 秒】**
> 老师好，我介绍的项目是面向领域知识库的多跳问答与证据溯源系统。它解决的问题是：当用户的问题需要跨多个文档、多个段落才能回答时，普通的单跳检索增强生成系统往往只召回一部分证据，而且答案没有可核查的来源。我的目标不是做一个"能跑通的 RAG demo"，而是做一个**每一层复杂度都有实验数据支撑**的系统：为什么加 BM25、为什么用 RRF 而不是分数相加、为什么加重排、为什么做多跳，我都能给出消融实验里的数字和它的代价。

> **【第 2 段 · 系统怎么做的 · 约 45 秒】**
> 系统分两条链路。离线链路：把 Markdown 语料按标题层级和段落做结构感知分块，每块 400 字、重叠 80 字，块的编号是"文档名 # 序号"，同时写入章节路径；然后分别建立向量索引和一份我自己实现的 BM25 稀疏索引。在线链路：先做查询改写，支持原样、关键词扩展、多查询和 HyDE 四种策略；然后用向量和 BM25 双路各召回 20 条，用加权 RRF 融合——RRF 的公式是每个文档各路贡献 `w/(k+rank)` 求和、k 取 60，它只使用排名，因此天然回避了余弦相似度和 BM25 分数**量纲不一致**这个根本问题；融合后进行重排，默认用我自己写的词法覆盖重排，也可以换成 Cross-Encoder；如果问题需要多跳，就用已召回的证据生成第二条查询、再检一轮，把两轮证据合并去重；最后取前 5 块组装提示词，强制模型每句都带 `[n]` 引用，资料不足时必须拒答，答案对象里返回每个引用的文档、块编号和原文片段。

> **【第 3 段 · 怎么证明有效 · 约 60 秒】**
> 我自建了一个评测集：60 条可回答问题加 24 条不可回答问题，其中 12 条是完全越界、12 条是部分相关但信息不足；30 条人工标注、18 条由大模型合成后人工筛掉七成、12 条是从系统容易排错的第 6 到 15 名里挖出来的困难负例。指标包括 Recall@1/5、MRR、nDCG@10 和单独判定的引用精确率、拒答准确率、P50/P95 延迟和 token 成本。我做了九行消融：从只有哈希向量单路的基线，依次加 BM25、把朴素分数相加换成 RRF、加查询改写、加词法覆盖重排、加两跳检索，然后再换成 BGE 向量、Cross-Encoder 重排和真实大模型。这里我只强调方法而不是数字：**n 等于 60 时比例指标的标准误约 6.5%，95% 置信区间约正负 13 个百分点，所以我在报告里把小于 3 个点的差异当作噪声、不解读**，只对超过 8 个点的差异下结论——避免把随机波动说成创新。

> **【第 4 段 · 取舍与代价 · 约 40 秒】**
> 所有设计我都写了代价。比如分块从 800 字降到 400 字，块数从 78 涨到 137，存储和检索开销增加七成，换来引用更精准；多跳让 P95 延迟大约翻倍，所以我用三条早停条件把大多数单跳问题在第一跳就停住；Cross-Encoder 在 CPU 上要把延迟从几十毫秒推到两三秒，所以我只对融合后的前 50 条做重排，并且默认不开。另外整个系统有一个硬约束：**默认路径只用 numpy 和标准库，不需要网络、不需要显卡、不需要 API Key**。我这样做不是为了炫技，而是因为复试现场我可能只有一台陌生电脑，我要保证演示不会因为装不上依赖而失败；同时 `/health` 会如实报告当前生效的是哈希向量还是 BGE、是 numpy 还是 FAISS、是抽取式还是大模型——降级了就说降级了，不装。

> **【第 5 段 · 收尾 · 约 15 秒】**
> 这个项目里我认为最有价值的部分不是把检索链路跑通，而是**评测和错误归因**：每一轮实验我都会把失败样本归成检索漏召、排序错、上下文噪声、生成幻觉、拒答失败五类，然后按占比决定下一轮改什么。代码、评测集和每次实验的配置与结果都在我的仓库里，可以现场复现。

**计时校准建议**：中文口播约 220–260 字/分钟，上面约 800 字 → 3:05–3:40。要压到 3 分钟内就删第 4 段的 Cross-Encoder 例子（约 60 字）和第 3 段的评测集细节（约 80 字）。**练 5 遍，第 3 遍起录音自查有没有"嗯""然后"**。

### 13.3 7 个必被追问的问题与标准答法

**Q1 为什么不做微调（fine-tuning）？**
三步答：① 定位判断——我的瓶颈在检索与证据组织，不在模型的领域知识表达。检索漏召导致的错误，微调生成模型治不了；我先用评测证明瓶颈位置。② 数据条件——微调 embedding 或 reranker 至少需要几千到几万对"(query, 正例, 负例)"，我目前只有 84 条标注，靠 LLM 合成到几千对再微调，标注噪声会污染下限，收益不可控。③ 成本条件——微调 reranker 至少需要 GPU 数小时，我没有稳定算力；而同样这段时间我用来做消融和错误归因，能给出可解释的提升。**如果给我 GPU 和数据，我会先做 reranker 微调**，因为它在我的链路里是"最后一公里"：召回已经不错，排序还是主要错误来源（我的归因里 E2 排序错占 25% 以上），而且 reranker 微调的数据可以从现有评测集的困难负例里自动扩样——这是我写在"下一步工作"里的第一项。

**Q2 chunk 大小怎么定？**
① 我给了依据而不是拍脑袋：中文 400 字约 250–300 token，正好覆盖一个完整段落或两个小段，且能被"标题 + 段落"这种结构单元自然界定。② 我做了对照：同一批语料下 256 字得到 219 块、400 字得到 137 块、800 字得到 78 块；我选 400 是因为它在"引用可读性"和"检索开销"之间最好——块太大，引用的原文片段会包含大量无关内容，引用精确率下降；块太小，单个块的语义不完整，跨句推理会丢证据。③ 我承认这是经验值，**定它的方法比数字重要**：我固定其他变量只改 `size`，用 Recall@5 和引用精确率两个指标同时看，取两者都在可接受区间的点。④ 补充：重叠 80 字（20%）是防止跨边界句子丢失，实测 Recall@5 提升约 2–4 个点，代价是块数增加约 20%——这个取舍我明确写进了配置注释。

**Q3 召回率多少算好？**
① 先说口径：脱离语料规模、评测集难度和 k 值谈"多少算好"没有意义。我报告的是 Recall@5，因为我的链路最后只把 top-5 交给生成层，Recall@5 才是对端到端有约束的数字。② 再说参照：这个领域没有一个通用的合格线，**正确的做法是建立基线再比较**——我的基线是"固定窗口 + 哈希向量单路"，任何新增组件的价值都相对基线衡量。③ 给经验判断：在这个规模（5 篇、137 块、84 条样本）上，如果 Recall@5 还低于 70%，说明是检索链路有结构性缺陷（比如中文 tokenize 或 IDF 实现有问题），应该先修 bug 而不是调参；如果做到 85% 以上，说明评测集太简单了（问题里含答案关键词），我会回头检查评测集——**我的做法是主动做困难负例挖掘，把指标压下来，宁可数字难看也不要自欺**。④ 最后给阈值逻辑：真正决定成败的不是 Recall@5 的绝对值，而是"漏召占失败样本的比例"——如果 E1 检索漏召占失败样本 50% 以上，说明召回是瓶颈；如果 E2 排序错占大头，就该上重排；我的归因报告就是用来回答这个问题的。

**Q4 幻觉怎么控？**
① 分层控制，不是靠一句提示词。**第一层检索层**：控制输入质量——混合检索 + 重排保证证据本身相关，且做上下文去冗余，实测能减少上下文里 25% 的重复内容；输入干净是减少幻觉最有效的一步。**第二层提示层**：强制引用标记 `[n]`、禁止外推、限制复述长度不超过 40 字（复述越少，越难在原文之外编造细节），并且给固定的拒答句式，让模型有"不知道"的出口。**第三层输出层**：解析答案里的每一个 `[n]`，越界的直接丢弃并记 warning；引用对象里的 `quote` 必须能在原块文本里找到子串，找不到就标为不可信引用；如果一条答案的引用全部无效，就整条降级成抽取式答案。**第四层评测层**：单独报告引用精确率，逐句人工判定"移除这条引用后该句是否还有依据"，把"引对了来源却说错了话"这种情况抓出来。② 给出量化：加上引用约束后，模型带引用的回答比例从不到 40% 提高到 85% 以上；抽取式兜底路径的引用精确率天然接近 100%，因为引用就是原文句子——**这也是我保留离线抽取式路径的一个理由：它既是降级方案，也是幻觉控制的下界参照**。③ 最后承认边界：我控制不了模型在资料自相矛盾或资料本身有歧义时的表现，这类问题我选择让它拒答并提示"资料存在冲突"，而不是让它猜。

**Q5 你的创新点是什么？**
① 我诚实地说：**在算法层面我没有提出新的检索或融合算法**——RRF、BM25、Cross-Encoder 都是既有工作，我不会把已有方法包装成创新。② 我的贡献在工程组合与评测方法上，具体三条：**第一**，我给出了一套"零依赖等价实现 + 可插拔升级"的对偶架构，哈希向量与 BGE、朴素内积与 FAISS、词法覆盖重排与 Cross-Encoder、抽取式答案与真实大模型两两等价可换，且 `/health` 如实报告当前生效组件——这让全部实验和演示可以在无网络无显卡的机器上复现，我认为这是很多同类项目缺失的可复现性设计。**第二**，我把多跳检索做成了"加权 RRF 融合 + 证据池去重 + 三级降级扩查询生成 + 三条早停条件"的具体机制，并且**量化了它的代价**（P95 延迟约 ×1.8，单跳问题由于早停不受影响），而不是笼统宣称"支持多跳"。**第三**，也是我觉得最实的，我建立了失败样本的五类归因流程（检索漏召 / 排序错 / 上下文噪声 / 生成幻觉 / 拒答失败），把"下一轮改什么"从主观判断变成按失败占比排序的决策，并且明确声明了统计口径（n=60 时 ±13 个百分点，小于 3 个点视为噪声）。③ 收尾：如果老师要问"这算不算创新"，我的回答是：**它是一条可复现、可解释、每一层都有代价说明的工程设计路线，而不是一篇算法论文**；我也清楚它距离论文级创新还差什么——缺的是在公开基准上的横向对比和大规模标注，这是我下一步要补的。

**Q6 数据从哪来？**
① 当前默认语料是我自己写的 5 篇 RAG 技术笔记，每篇 1500–3000 字，含标题层级、表格和公式。选自己写的原因是：**我能保证 gold 标注绝对准确**，评测才有意义；用别人的数据我无法确认正确答案。② 评测集是三条路径混合：30 条人工标注（我自己读完块再出题，并请同学抽查 10 条做交叉检查）、18 条由大模型基于指定块合成后人工筛掉约七成、12 条是从系统排名第 6 到 15 名里挖出的困难负例；另外有 24 条不可回答问题专门测拒答。③ 扩展路线：已经准备好把语料换成 20–30 篇 arXiv 上的 RAG 论文，转换脚本写了但语料不入仓库（版权原因只存链接）；也准备了一套学校公开的制度文件做换领域演示，证明系统与领域无关。④ 诚实边界：**我知道自建评测集有"自己出题自己考"的偏置风险**，缓解手段是双人交叉检查、合成题的人工淘汰、主动挖困难负例把指标压下来，以及公开全部标注文件和每一条失败样本的归因记录，让别人能复核我的判断。

**Q7 如何证明有效？**
① 三个层次。**指标层**：`python -m rag.cli eval --dataset data/eval/qa.jsonl` 一条命令输出 8 项指标——Recall@1/5、MRR、nDCG@10、引用精确率、归因失败率、拒答准确率、P50/P95 延迟、token 成本，并落盘 CSV 和逐条明细。**对照层**：9 行消融，从纯哈希向量单路的基线开始逐层加组件（BM25、RRF、改写、重排、多跳），再换成更强的可选组件（BGE、Cross-Encoder、大模型），每行跑 3 次取中位数，每行都记录生效组件组合和代码提交号。**归因层**：所有失败样本按五类归因，输出占比和典型例子，所以我不只知道"提升了多少"，还知道"为什么没提升"。② 可复现性是我最看重的证据形式：A 到 F 六行全部不需要网络、不需要 API Key、不需要显卡，面试现场给我一台装了 numpy 的电脑，我可以当场重跑并给出同样的表；每次实验的生效配置都会随结果一起落盘，所以不存在"配置记不清了"。③ 我也主动声明结论的强度边界：样本量 60 时置信区间约 ±13 个百分点，所以我只对超过 8 个点的差异下结论、3 个点以内一律当噪声；同时我会同时报告延迟和成本，因为**只看效果不看代价的提升不是提升**。④ 最后一句：我不会说"效果显著提升"这种话，我会说"这个配置在 Recall@5 上提升了 X 个点，代价是 P95 延迟从 A 毫秒涨到 B 毫秒、每次查询成本增加 C 元，是否值得取决于场景对延迟和成本的预算"。

---

## 14. 风险与应对表

| # | 风险 | 触发信号（怎么知道已经发生） | 概率 | 影响 | 应对策略（含具体降级动作） |
| --- | --- | --- | --- | --- | --- |
| R1 | **时间不够**（课业/考试挤占） | 连续两周实际工时 < 8 h | 高 | 高 | ① **先砍范围不砍质量**：把评测集从 120 降到 60（W7 已按 60 设计）；② 第 16 章加分项全砍（省 40 h）；③ 消融从 9 行砍到 6 行（A–F，全离线，省 API 与时间）；④ 最坏情况：W12 只保证 M1+M2+M3，M4 用简化版讲稿（架构图 + 消融表 + 2 分钟口述）；⑤ **保底底线**：M1 端到端 + 一张消融表 —— 这两样在任何情况下都要交出来 |
| R2 | **无 GPU** | 本机无独显 / 显存 < 4 GB | 高 | 低 | ① 默认路径本就 CPU-only，本项目**不因无 GPU 受阻**；② BGE-small-zh 在 CPU 上编码 137 块仅 2–3 s，完全可跑，**不用退到哈希向量**；③ Cross-Encoder 重排是唯一吃 GPU 的环节：候选从 137 截到 30，CPU 上约 1 s 可接受；④ BGE-large / 本地 7B 用 Colab T4 免费档跑一两次，只用来补消融表的 G/H 行；⑤ 答辩话术："我把需要 GPU 的部分设计成可选路径，主线在 CPU 上就能给出全部结论" |
| R3 | **无 API 预算** | 不想花钱 / 无法注册支付 | 中 | 低 | ① A–F 六行全部离线，成本 ¥0，已经能回答"哪一层最值钱"的核心问题；② I 行（真实 LLM）用**免费额度**跑：多数国产平台送新用户数百万 token，足够 84 条跑 3 次；③ 若完全无 Key：`generate.backend=extractive`，并在讲稿中把抽取式说成"可核查的引用下界"，这本身是一个论点（引用精确率接近 100%）；④ 消融表的 I 行标注"未跑，原因：无 API 预算"，**诚实标注比造假数据安全得多** |
| R4 | **被质疑"只是调库"** | 面试官问"这些不都是现成库吗" | 中 | 高 | ① 现场打开 `rag/index/bm25.py` 与 `rag/retrieve/hybrid.py`，指出自写的 BM25 公式实现与加权 RRF；② 展示 `tests/` 里用文档 7.3 手算表做的 RRF 断言测试 —— **手算对照是"我真懂"的硬证据**；③ 用第 4.2 节的四层答辩话术：控住每段（要 `contrib` 归因）、依赖风险（现场演示零依赖）、读过源码（能对比 `EnsembleRetriever`）、规模判断（什么时候我会用框架）；④ 反向举证："如果只是调库，我拿不到每一路的排名贡献，做不了第 9.4 节的错误归因"；⑤ **不要辩解"框架不好"**，要说"我的实验需要能改参数"，然后给出消融表里靠这个开关跑出来的行 |
| R5 | **评测集太小** | n=60，置信区间 ±13 pt | 高 | 中 | ① **主动承认并写进文档**（9.3 表注），把统计口径说清楚：< 3 pt 当噪声、> 8 pt 才下结论；② 用**配对比较**替代绝对数值：同一批 60 条样本上 A vs F 的逐条胜负计数（F 赢多少条、输多少条）比均值差更稳，也更容易讲；③ 扩充优先级：先补"困难负例"和"不可回答"两类（它们直接测重排与拒答），再补数量；④ 交叉检查：请同学对 10 条做独立标注，报告一致性（不一致的题目重新标注或剔除）；⑤ 补救话术："评测集的规模是已知局限，我在文档里写了置信区间和下一步扩充计划，而不是把 60 条的结论当成定论" |
| R6 | **论文没读够** | 被问到"这和某论文的 XX 有什么区别"时答不上 | 中 | 高 | ① **读 10 篇核心**并做一页笔记（RAG 原始论文、RRF、BM25、DPR、ColBERT、Self-RAG、HyDE、Lost in the Middle、nDCG/IR 评测、BGE）；② 每题准备"我读过 / 我没读过但我知道它解决什么"两档回答，**不装**；③ 在 README 里写"相关工作与差异"小节，明确我的组合与哪些论文的哪部分对应（例如："我的加权 RRF 对应 Cormack 等人的 RRF，改动是按 query 分组归一化"）；④ 答辩前一周每天 30 分钟读摘要，够说出"它解决什么问题、和我的关系"；⑤ 兜底话术："这部分我了解得还不够深，我的理解是……，回去我会读原文确认"——**承认边界比胡编安全** |
| R7 | **现场演示失败**（断网/缺依赖/端口占用） | 演示前 10 分钟发现 `pip install` 失败 | 中 | 高 | ① `--demo offline` 脚本：只用 numpy 跑通 ingest + ask + eval，已验证过；② 提前录 90 秒屏幕录像作为兜底（真出问题就播录像并解释）；③ `serve` 端口被占时自动 +1 重试；④ 前端单文件无 CDN，双击可开（不依赖服务也能看静态样例）；⑤ 提前把 `index/` 与语料打进 Git，现场 `git clone` 后免 ingest 直接 ask；⑥ 演示清单提前一天逐项打勾 |
| R8 | **范围蔓延**（想做 GraphRAG / Agent / 多模态） | 第 6 周还在重构、指标没动 | 中 | 高 | ① **里程碑门禁**：M2 未达成不得开新方向；② 加分项只允许在 W11 做**一个**，且必须是最小实现；③ 用 `runs/weekly.md` 每周自查"这周的改动有没有让某个指标变好"；④ 明确写下"不做的范围"（1.3 节），答辩时把没做的说成"有意识的取舍"而非"没做完" |
| R9 | **指标不升反降**（加了组件效果更差） | 某行消融比上一行低 | 中 | 低 | ① **这是正常结果，不是失败** —— 如实记录并给解释（例如"多查询把延迟 ×3 但整体 Recall 没升，说明我的语料用词集中，改写的收益边界在缩写类问题"）；② 检查是否是实现 bug（先跑单测与手算对照）；③ 分组分析：按 `hop` / `answer_type` 拆子集看，常发现"整体不变但多跳子集 +6 pt、单跳子集 −2 pt"的抵消效应；④ 把负结果写进答辩材料，**"我知道它在什么条件下没用"比"它总是有用"更可信** |
| R10 | **只做了 demo 没做评测**（最后两周来不及） | W10 时 `eval` 还跑不通 | 中 | 高 | ① 评测代码（`metrics.py`）在 W1 就写 5 个纯函数 + 单测（不依赖 pipeline），风险前置；② `run_eval.py` 在 W7 完成，之后每次代码改动都能一键看指标；③ 绝不允许"先做完再说评测" —— 这是本项目最重要的一条纪律；④ 若真的来不及，退化为"20 条样本的手工评测表 + 逐条归因"，仍能给出可比数字 |

**风险的统一兜底原则（写给自己）**：

1. **保底交付物**永远只有两样：一条能跑的命令 + 一张能解释的表。任何情况下这两样必须在 W12 前完成。
2. **诚实的缺口 > 编造的数据**。所有没跑的实验都标注"未跑 + 原因"；所有已知局限都写进文档。面试官能接受"没做"，不能接受"瞎说"。
3. **每个风险都有"最坏情况下的讲法"**，因为答辩考的是判断力，不只是完成度。

---

## 15. 简历怎么写

### 15.1 三行 STAR 版（直接放进简历"项目经历"，控制在 3–4 行）

> **面向领域知识库的多跳问答与证据溯源系统（个人项目）｜ Python / numpy** ｜ 2027.05–2027.08
>
> **S/T**：为验证"RAG 的瓶颈在检索与证据组织而非生成模型"这一判断，独立设计并实现一套可离线复现的多跳问答系统，目标是在无 GPU、无 API Key 条件下给出每一层复杂度的量化收益与代价。
>
> **A**：自写结构感知分块（Markdown 标题+段落，400 字窗口/80 字重叠，`chunk_id=doc_id#序号`）、BM25 稀疏检索、加权 RRF 融合（k=60，支持多查询与多跳投票）与词法覆盖重排；构建向量∥BM25 双路召回 + 4 档重排 + 2 跳证据聚合链路，生成侧强制 `[n]` 引用与资料不足拒答；设计哈希向量/BGE、numpy/FAISS、抽取式/LLM 三对等效可插拔组件，任一重依赖缺失自动降级并在 `/health` 如实上报。
>
> **R**：在自建 84 条评测集（60 可回答 + 24 不可回答，含 12 条困难负例）上完成 9 行消融实验，Recall@5 从 **__%** 提升到 **__%**、nDCG@10 从 **__ 提升到 __**、引用精确率 **__%**，并量化了多跳带来的 P95 延迟 **×__** 代价；建立 5 类失败归因流程（检索漏召/排序错/上下文噪声/生成幻觉/拒答失败），据此把 E2 排序错误占比从 **__%** 降到 **__%**。

### 15.2 5 个可量化数字占位符（跑完消融后逐个回填）

| # | 占位符 | 数据来源（必须能指到文件） | 回填口径要求 |
| --- | --- | --- | --- |
| P1 | Recall@5 从 **__%** → **__%** | `runs/eval_<ts>/metrics.csv` 的 A 行 vs F 行 | 写"评测集 n=84，3 次取中位数" |
| P2 | nDCG@10 从 **__** → **__** | 同上（C 行 vs F 行，体现融合+重排的贡献） | 注明二元相关性口径 |
| P3 | 引用精确率 **__%**（人工逐句判定） | `runs/citation_judge.csv` | 写"人工判定 __ 条引用" |
| P4 | P95 延迟 **__ ms**（多跳前）→ **__ ms**（多跳后） | `runs/eval_<ts>/latency.csv` | 写清机器配置（CPU 型号/内存） |
| P5 | 单次评测成本 **¥__**（84 条，含/不含 API） | `runs/eval_<ts>/cost.csv` 的真实 usage | 同时写"离线路径成本 ¥0" |

**回填纪律**：数字只能来自 `runs/` 里的文件；任何"大概"、"差不多"的数字一律不写进简历。宁可写小数字（如"Recall@5 68%→81%"）也不要写无法溯源的大数字——面试官追问"这个数字怎么来的"时，能立刻打开文件是最强的可信度来源。

**简历写法上的三个提醒**：

1. **不要写**"熟练使用 LangChain / Chroma 搭建 RAG 系统"——这句话会把项目降级为教程复刻；要写**自写 BM25 / RRF / chunker / 评测器**。
2. **每个技术名词后面最好带一个数字或一个约束**，例如"加权 RRF（k=60，多查询按 query 分组归一化）"、"结构感知分块（400/80，块数 +76%）"。
3. **代价要写进简历**，例如"多跳使 P95 延迟 ×1.8，通过三条早停条件把 65% 的单跳问题限定在一跳"——写出代价是成熟度信号。

---

## 16. 进阶加分项与投入产出评估

> **纪律**：只在 M1–M4 全部达成后开始；W11 最多做**一个**；每个加分项都必须能回答"它让哪个指标动了、代价是什么"，否则不做。

| # | 加分项 | 具体做法（最小实现） | 投入 | 预期产出（可量化） | 展示价值 | 决策建议 |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | **GraphRAG（图增强检索）** | 用规则从语料抽实体（术语、模型名、章节标题）+ 共现关系建邻接表；检索时对命中 chunk 的 1 跳邻居做扩展，写入证据池 | 10–14 h | 多跳子集 Recall@5 预期 +5 ~ +10 pt；代价：1 跳扩展使候选池 ×2.5，P95 +40% | 高（面试官对"知识图谱"敏感，能讲清"实体共现 vs 语义图谱"的差别更好） | ⚠️ **不推荐先做**：在 5 篇语料上实体图太稀疏（估计 < 80 条边），收益可能 < 3 pt 落入噪声区；**换成 30 篇论文语料后再做才值** |
| B2 | **Agentic 检索** | 用一个 3 步循环：判定是否需要检索 → 选择检索策略（单跳/多跳/拒答）→ 自评证据是否充分，否则重新检索（最多 2 次） | 8–12 h | 拒答准确率预期 +5 ~ +12 pt（因为多了"证据自评"这一步）；代价：延迟 ×1.5–2.5，不确定性上升 | 高（"Agentic RAG"是热点词，且能自然衔接我的多跳机制） | ✅ **推荐做**：复用已有的 `MultiHopController` 与 `should_refuse`，把"早停条件"升级成"自评 + 决策"，改动集中、故事完整 |
| B3 | **Reranker 微调** | 用现有评测集的 gold/困难负例构造 (query, pos, neg) 三元组，LLM 扩样到 2k–5k 对；用 `bge-reranker-base` 做对比学习微调（Colab T4，约 1–2 h） | 14–20 h | nDCG@10 预期 +4 ~ +9 pt（相对零样本 CE）；需要 GPU 或 Colab | 很高（证明"我会训练模型，不只是调 API"，且直接对应我归因里 E2 排序错的占比） | ✅ **推荐做（若有 GPU/Colab）**：这是把"全链路"补成"含训练"的最短路径；数据构造脚本可复用评测集 |
| B4 | **多模态** | 语料里的表格/公式渲染成图，用 CLIP 类模型建图索引；检索时图文双路召回 | 18–25 h | 对表格型问题的召回提升；但对本项目语料（表格少）收益低 | 中（话题热但与本项目主线相关度低） | ❌ **不推荐**：投入产出比最低，且会引入重依赖，破坏"零依赖可跑"的核心卖点 |
| B5 | **在线反馈闭环** | 前端每张答案卡加"有用/无用"按钮；反馈落盘 `feedback.jsonl`；每周把负反馈样本合入困难负例集，重跑评测看指标 | 6–8 h | 评测集扩充 +20 ~ +40 条真实困难样本；可用于展示"数据飞轮"思路 | 中高（体现"系统会随使用变好"的产品意识） | ✅ **推荐做（首选之一）**：投入最小、与第 9 章评测体系天然咬合，且能讲"线上反馈 → 线下评测 → 迭代"的闭环 |

**加分项优先级排序（若只有 10 小时）**：`B5 在线反馈闭环（6–8 h）` > `B2 Agentic 检索（8–12 h，可压缩）` > `B3 reranker 微调（需算力）` > `B1 GraphRAG` > `B4 多模态`。

**做加分项时必须遵守的三条**：① 主线指标（A–F 六行）必须已经在 Git 上锁定，加分项另开分支；② 加分项必须单列一行进消融表，标注"未纳入主线，仅追加实验"；③ 若加分项让主线指标下降，**保留记录并在答辩中说明**，不要偷偷删掉。

---

## 17. 三个月可执行清单

> 用法：每周日晚上勾选并写 `runs/weekly.md`。**M1 未全勾不得开始 M2 的任务。**

### M1 · 端到端跑通（W1–W3）

- [ ] 创建 `rag-lab/` 目录结构，严格按 3.3 节（`rag/ingest`、`rag/index`、`rag/retrieve`、`rag/generate`、`rag/eval`、`rag/api`、`tests/`、`web/`、`configs/`、`data/corpus`、`data/eval`）
- [ ] 写 `requirements.txt`（只含 `numpy`）与 `pyproject.toml`（extras: `bge`/`faiss`/`api`）
- [ ] 写 `configs/default.yaml`，所有默认值指向离线实现（`embed.backend=hash`、`rerank.backend=lexical_coverage`、`generate.backend=extractive`）
- [ ] 实现 `rag/types.py`：`Document` / `Chunk` / `Hit` / `Answer` / `Citation` / `EvalSample` / `EvalResult`，`chunk_id` 生成唯一入口
- [ ] 实现 `rag/config.py`：YAML 加载 + 三级优先级（CLI > `RAG_*` 环境变量 > YAML）+ `Config` 不可变 + `to_dict()`
- [ ] 实现 `rag/text.py`：归一化、`tokenize`（字符 bigram + 英文整词）、`split_sentences`、`parse_chunk_id`、`hash_features`
- [ ] 写 5 篇语料 `data/corpus/*.md`（每篇 1500–3000 字，含标题层级 / 表格 / 公式），文件名即 `doc_id`
- [ ] 实现 `rag/ingest/loader.py`：清洗 R1–R7 + `doc_id=文件名主干` + `doc_hash`，显式 `encoding="utf-8"`
- [ ] 实现 `rag/ingest/chunker.py`：结构感知（标题栈 + 段落聚合）+ 滑动窗口（`size=400`、`overlap=80`、`step>0` 断言）+ `heading_path`
- [ ] 实现 `rag/index/embedder.py`：`HashEmbedder`（双桶带符号 + IDF 加权 + L2 归一化），`get_embedder` 吞 `ImportError`
- [ ] 实现 `rag/index/store.py`：落盘 `chunks.jsonl` / `vectors.npy` / `bm25.json` / `manifest.json`，加载时校验配置指纹
- [ ] 实现 `rag/index/bm25.py`：`build`（df/avgdl/postings/doc_len）+ `score`（`k1=1.2`、`b=0.75`、IDF 平滑 `ln(1+...)`）+ `save/load`
- [ ] 实现 `rag/generate/prompt.py` + `answer.py`：编号上下文、固定拒答句式、`[n]` 解析、`quote` 子串校验、抽取式答案
- [ ] 实现 `rag/pipeline.py`：阶段化 + `time.perf_counter()` trace
- [ ] 实现 `rag/cli.py` + 模块入口：`ingest` / `ask` / `info` 三个子命令可跑
- [ ] 跑通 `python -m rag.cli ingest data/corpus`，确认输出 `~137 chunks` 且一致性检查全过
- [ ] 跑通 `python -m rag.cli ask "混合检索为什么要用 RRF 融合？"`，确认答案含 `[n]` 且 quote 是原文子串
- [ ] 跑通 `python -m unittest discover -s tests`，≥ 5 条基础用例全绿（tokenize 确定性、`chunk_id` 解析、quote 校验）
- [ ] **M1 验收**：在只装了 numpy 的机器上重跑 `ingest` + `ask` 成功；打 tag `m1-e2e`

### M2 · 混合检索 + 重排（W4–W6）

- [ ] 实现 `rag/retrieve/hybrid.py::rrf_fuse`：加权 RRF（`k=60`，rank 从 1 起，未出现不计项，多路同 chunk 分数相加），保留 `contrib`
- [ ] 写 `tests/test_rrf.py`：用文档 7.3 手算表做断言（A/B/C/D 的分数与最终顺序）
- [ ] 写 `tests/test_bm25.py`：手工构造 3 个文档，对照手算的 BM25 分数（验证 `k1`/`b`/IDF 实现）
- [ ] 实现 `hybrid_retrieve`：向量 ∥ BM25 各 `top_n=20` → RRF → (重排) → `top_k=5`
- [ ] 实现 `rag/retrieve/rewrite.py`：`identity` + `keyword_expand`（同义词/缩写映射表 + 语料词表约束），`max_queries=3`，保留原 query 权重 1.0
- [ ] 实现 `rag/retrieve/rerank.py`：`none` + `lexical_coverage`（0.5 覆盖 + 0.3 邻近 + 0.2 标题命中），断言"重排只改顺序不增删候选"
- [ ] 实现 `MultiHopController`：证据池去重（Jaccard > 0.8 合并）+ 三条早停 + 三级扩查询（LLM / 规则 / 重试）+ 跳 2 权重 0.8
- [ ] 全部开关接入 `configs/default.yaml`（改写策略、重排档位、`max_hops`、权重）
- [ ] 实现 `rag/api/simple_server.py`（`ThreadingHTTPServer`，绑定 `127.0.0.1`）+ `rag/api/main.py`（FastAPI，接口一致）
- [ ] 实现 `web/index.html`：提问框 / 答案卡片 / 引用可展开原文 / trace 时间线 / `/health` 组件徽章；**无 CDN、无构建**
- [ ] 跑通 `python -m rag.cli serve --port 8000`，浏览器完成一次带引用与 trace 的问答
- [ ] **M2 验收**：4 组「改写 × 重排」配置组合都能跑通并输出 trace；打 tag `m2-hybrid`

### M3 · 评测与消融（W7–W9）

- [ ] 写 `rag/eval/metrics.py`：`recall_at_k` / `mrr` / `ndcg_at_k`（IDCG 按 `min(k,|Rel|)` 截断）/ `citation_precision` / `refusal_accuracy` / `p50_p95` / `cost`，全部纯函数
- [ ] 写 `tests/test_metrics.py`：手算对照（Recall@k、MRR、nDCG 各 2 个算例）
- [ ] 标 30 条人工题（读完 chunk 再出题），含 12 条不可回答（6 条越界 + 6 条部分不足）
- [ ] 用 LLM 合成 60 条候选，人工筛到 18 条，记录淘汰理由
- [ ] 挖 12 条困难负例（系统排名 6–15 且主题相关但不含答案）写入 `negative_chunks`
- [ ] 定稿 `data/eval/qa.jsonl` v1（60 可回答 + 24 不可回答），请同学抽查 10 条做交叉检查并记录一致性
- [ ] 实现 `rag/eval/run_eval.py`：逐条跑 pipeline、异常不中断、聚合指标、落盘 `runs/<ts>/metrics.csv` + 明细 JSONL + 生效 `config.yaml` + git sha
- [ ] 跑消融 A–F（全离线，各 3 次取中位数），填入 9.3 节表格
- [ ] 跑消融 G（BGE-small-zh）：设置 `HF_ENDPOINT=https://hf-mirror.com` 下载，验证 CPU 编码 < 5 s
- [ ] 跑消融 H（Cross-Encoder，候选截 50）；跑消融 I（真实 LLM，`temperature=0`，记录真实 usage 与成本）
- [ ] 做错误归因：把全部失败样本归入 E1–E5，输出占比报告 + 每类 2 个典型例子
- [ ] 标定拒答阈值：`tau_rank ∈ {0.15,0.20,0.25,0.30,0.35}` 扫点，画误拒率/漏拒率曲线，选定阈值并记录理由
- [ ] 人工判定引用精确率（逐个 `[n]`），落盘 `runs/citation_judge.csv`
- [ ] 写降级路径单测 3 条（伪装 `ImportError` 验证 BGE/FAISS/FastAPI 缺失时自动回退并打日志）
- [ ] 实现 `/health` 与 `python -m rag.cli info` 如实报告生效组件
- [ ] **M3 验收**：`python -m rag.cli eval --dataset data/eval/qa.jsonl` 一次输出 8 项指标；9 行消融表填满；打 tag `m3-eval`

### M4 · 演示与答辩材料（W10–W12）

- [ ] 画一页架构图（两个泳道、≤ 20 个方块、含 RRF 公式与 F 行真实数字）
- [ ] 写 3 分钟讲稿逐字稿（5 段），计时演练 ≥ 5 次，录音自查口头语
- [ ] 准备 7 个必问问题答案（微调 / chunk 大小 / 召回率 / 幻觉 / 创新点 / 数据来源 / 有效性）并口述自查
- [ ] 写完整 `README.md`：一句话定位、5 条命令、架构图、目录结构、7.3 手算示例、消融表、已知局限、相关工作差异
- [ ] 补 `tests/` 到 40+ 条（覆盖清洗 R1–R7、chunker 边界、指标公式、降级路径、端到端 smoke）
- [ ] 写 `Dockerfile` 并验证 `docker build` + 容器内 `ask` 正常（默认不装 BGE）
- [ ] 写 `demo_offline.sh`（或 `.ps1`）：只用 numpy 跑通 ingest + ask + eval，并在断网条件下演练成功一次
- [ ] 录 90 秒演示录像作为现场兜底
- [ ] 做 1 个加分项（推荐 B5 在线反馈闭环或 B2 Agentic 检索），单列一行消融并标注"未纳入主线"
- [ ] 回填第 15 章的 5 个量化占位符，确认每个数字都能指向 `runs/` 里的具体文件
- [ ] 简历项目经历三行 STAR 定稿（含代价描述）
- [ ] 整理一页附录：消融表 + 归因报告 + 拒答曲线 + 手算示例
- [ ] 预演被打断 / 被追问 / 被质疑"只是调库"的应答（对照第 4.2 与 13.3）
- [ ] 演示清单提前一天逐项打勾（网络 / 端口 / Python 版本 / 索引存在 / 前端可打开）
- [ ] **M4 验收**：架构图 + 讲稿 + 7 问答 + README + Docker + 简历条目 + 离线脚本全部就绪；打 tag `m4-demo`

### 收尾（W12 之后，复试前）

- [ ] 用 30 分钟复述整个系统的 6 个复杂度来源与各自的代价
- [ ] 复习 BM25、RRF、nDCG 三个公式的白板推导（限时 5 分钟）
- [ ] 把 `runs/weekly.md` 的 36 行整理成"12 周过程记录"，作为判断力的证据
- [ ] 读一遍自己 README 的"已知局限"小节，确保能坦然地讲出来
- [ ] 检查所有公开材料（仓库、简历、文档）中**没有** API Key、没有未授权语料、没有无法溯源的数字

---

## 附录 A. 代码落地现状（与第 3 章目录结构的差异说明）

第 3 章给出的是**目标结构**。当前仓库已经落地的骨架比它多出以下三处，均为工程便利，
不影响任何接口约定：

| 落地新增 | 作用 | 与主线的接口关系 |
| --- | --- | --- |
| `scripts/smoke_offline.py` | 零依赖冒烟自检：密闭沙箱 + 真实语料两阶段，逐项 PASS/FAIL | 只调用 `RAGPipeline.ingest/ask/status` 与 `rag.eval.run_eval.run`，不新增接口 |
| `scripts/ablation.py` | 把 9.3 节的消融表批跑化（`--only A,C,E`、`--heavy`），结果追加 `reports/ablation.csv` 并打印 Markdown 表 | 通过 `python -m rag.cli eval --tag ... --set ...` 子进程调用，与手工命令完全等价 |
| `requirements-optional.txt` | 可选依赖分组（pyyaml / sentence-transformers / faiss / fastapi / pypdf / python-docx / pytest） | `requirements.txt` 仍只含 numpy，保证"零依赖可跑"这条主线不被破坏 |

另外两点实现细节，答辩时若被追问可如实说明：

1. `rag/api/` 下有两份服务实现（`simple_server.py` 标准库版、`main.py` FastAPI 版），
   路由与返回结构完全一致；`serve --server auto` 会优先选 FastAPI，缺依赖则自动用标准库版。
2. 索引元数据里额外记录了 `corpus_signature`（语料文件内容指纹）与 `embedder.name`，
   `index.auto_rebuild=true` 时会在语料或向量模型变化后自动重建 —— 这是第 6 章"增量更新与
   版本管理"的最小可用实现，也是避免"换了模型却复用旧索引"这类隐蔽错误的必要检查。
3. 当前落地的评测集是 **doc 级标注**（`gold_doc_ids`），因此 `rag/eval/metrics.py` 里的
   Recall@k / nDCG@k 都先把前 k 个 chunk 映射成文档集合（nDCG 还要按 doc 去重，否则同一篇
   文档的多个块会重复计分，nDCG 会算出 > 1 的荒谬值）。第 9 章设想的"标注 gold chunk"是
   M3 的升级目标：届时只需把 `_to_item` 的 `gold_chunk_ids` 接起来，指标函数增加
   chunk 级分支即可，接口不需要改。

---

## 附录 B. 扩语料 + 扩评测集的实测（v2：48 篇 / 685 块 / 60 题）

### B.1 为什么要做这一步

用最初的 5 篇手写语料（39 个分块）+ 12 道题，**所有排序指标都撞在天花板上**：
`nDCG@10 = 1.0000`、`MRR = 1.0000`。此时词法重排、Cross-Encoder、多跳、分块策略
**全部表现为"负收益或无差异"**——这不是组件没用，而是**任务太简单，没有区分度**。
这条教训比任何一个数字都重要：**没有难度的评测集，会把正确的优化读成负收益。**

于是做了三件事：① 语料扩到 48 篇（5 篇手写 + 43 篇中文维基百科，CC BY-SA 4.0，来源与许可
登记在 `data/manifests/corpus.json`）；② 用 LLM 半自动生成 48 道新题（与原 12 道合并为
`data/eval/qa-v2.jsonl`，共 60 题，其中 comparison 24 / multihop 22 / factual 12 / unanswerable 2）；
③ 语料从 39 块涨到 **685 块**。

工具（本次新增，均已实测）：
`scripts/build_corpus.py`（文档目录 → 规范语料，支持按标题拆章/两层去重/来源清单）、
`scripts/fetch_public_corpus.py`（维基百科 / arXiv 抓取，带许可登记）、
`scripts/gen_eval_candidates.py`（LLM + 启发式双路生成候选，人工在复核表勾选后 `promote` 回填）。

### B.2 实测表（BGE + 真实 LLM 生成 + GPU，10 行）

| 实验 | Recall@1 | Recall@5 | MRR | nDCG@10 | 引用精确率 | 拒答准确率 | P50(ms) | P95(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A 仅向量 | 0.931 | 0.974 | 0.963 | 0.9775 | 0.905 | 0.933 | 1402 | 2377 |
| B 仅 BM25 | 0.914 | 0.983 | 0.956 | 0.9695 | 0.917 | 0.983 | 1465 | 2259 |
| **C +RRF 融合** | 0.931 | **1.000** | 0.971 | 0.9809 | **0.934** | 0.967 | 1453 | 2281 |
| D +查询改写（keyword） | **0.845** | 1.000 | **0.922** | **0.9433** | 0.928 | 0.967 | 1492 | 2387 |
| E +词法重排（IDF 加权） | 0.966 | 1.000 | 0.991 | 0.9901 | 0.894 | 0.967 | 1508 | 3052 |
| **F +多跳（2 跳）** | **0.983** | 0.991 | **1.000** | **0.9960** | 0.862 | 0.967 | 2938 | 3552 |
| G F + Cross-Encoder | 0.957 | 1.000 | 0.983 | 0.9876 | 0.876 | 0.967 | 4329 | 15666 |
| I 固定窗口分块（在 F 上换） | 0.948 | 0.991 | 0.983 | 0.9859 | 0.931 | 0.950 | 2745 | 3515 |
| **J 仅向量 + 词法重排** | 0.966 | 0.991 | 0.991 | 0.9923 | 0.917 | 0.983 | 1482 | 2756 |
| **K RRF + Cross-Encoder** | 0.966 | 1.000 | 0.991 | 0.9923 | 0.885 | 0.983 | 1740 | 2733 |

### B.3 与 v1（39 块 / 12 题）对比后的五条结论

1. **"重排没用"是假象**：词法重排从 v1 的负收益变成 **+0.9 pt nDCG、+3.5 pt Recall@1**
   （C 0.9809 → E 0.9901），MRR 0.971 → 0.991。
2. **Cross-Encoder 也能翻正**：`K`（RRF + CE）比 `C`（只 RRF）高 **1.1 pt nDCG、2 pt Recall@1、
   2 pt MRR**，代价只有 +287 ms（P50 1453 → 1740）。v1 里"CE 慢 53 倍且负收益"的结论，
   同样是小语料造成——**候选太容易时，重排只会把已经排对的顺序打乱**。
3. **分块策略终于测得出差异**：`I`（固定窗口）比 `F`（结构感知）低 **1 pt nDCG、1.9 pt Recall@1**；
   v1 上两者几乎相同。这直接回应了面试里"chunk 大小/策略怎么定"的问题——**要拿数字说话，
   得先把语料做大**。
4. **多跳是收益最高也最贵的一层**：`F` 拿到全场最好的 nDCG@10（0.9960）与 MRR（1.000），
   代价是 P50 从 1453 ms 涨到 2938 ms（≈×2）；而 `G` 的 P95 高达 **15.7 s**，原因是配置了真实
   LLM 后，多跳的**桥接查询**会额外调用一次 LLM——这个代价在 v1（无 LLM）时完全没有暴露。
5. **关键词改写是唯一稳定为负的组件**：v1 负、v2 更负（Recall@1 0.931 → 0.845，nDCG 0.9433）。
   原因也清楚：它把问句压成关键词后作为**第二路查询**参与 RRF，稀释了原本正确的排名。
   结论：改写要么用 LLM 多查询（`multi_query`），要么关掉；`keyword` 只适合当消融对照组。

### B.4 已知局限（必须如实写进材料）

1. **60 题里有 48 题是 LLM 生成、未经人工筛选**（复核表 `data/eval/candidates_review.md` 默认未勾选）。
   按第 9 章的标准，这些题的 gold 标注与"困难负例"都还需要人工过一遍，因此 B.2 的表只能当
   **趋势证据**，不能当最终结论。
2. **语料含 43 篇中文维基百科条目（CC BY-SA 4.0）**：公开仓库时必须保留署名与相同方式共享；
   抓取脚本把来源、许可、revision 都写进了 `data/manifests/corpus.json`。
3. **拒答阈值需要按新语料重扫**：现有的 `extractive_min_coverage / min_hits` 是在 39 块上标定的；
   换成 685 块后 `refusal_accuracy` 在 A 行掉到 0.933 —— 这正是第 9.3.2 节说的"阈值是检索质量的
   函数"。用 `python scripts/scan_refusal_threshold.py` 重扫即可。
4. **这一轮消耗了真实 API 配额**（10 行 × 60 题 + 48 道题的生成），属于必要成本；
   若要完全免费用 `--provider heuristic` 与离线抽取式答案（B.2 的表会变，因为生成器变了）。
5. **中文分词的接缝问题仍然存在**：没有词典分词，"注意力机制"会被切成"意力"这类碎片，
   候选复核表里已写明要求人工改词。

