# rag-lab · 领域知识库问答（复杂 RAG 工程骨架）

**一句话定位**：一个可插拔、可评测、**只装 numpy 就能跑通全链路**的 RAG 工程骨架 ——
离线索引 → 查询改写 → 向量 ∥ BM25 双路召回 → RRF 融合 → 重排 → 强制引用溯源 →
逐条指标评测与消融实验，并附带一个单文件网页 demo 和一套零依赖 HTTP 服务。

配套的方案文档见 [PLAN.md](PLAN.md)（选型理由、12 周计划、复试答辩话术、硬件与预算清单）。

---

## 1. 30 秒上手（不需要 GPU、不需要 API Key、不需要联网）

```bash
cd rag-lab
pip install numpy                      # 唯一必需依赖
python -m rag.cli ingest data/corpus   # 建索引（哈希向量 + 自写 BM25）
python -m rag.cli ask "混合检索为什么要用 RRF 融合？"
```

预期输出：答案正文（带 `[1][2]` 引用标记）、引用卡片（doc_id / chunk_id / 分数 / 原文）、
以及 `rewrite / retrieve / rerank / generate / total` 五个阶段的耗时。

打开重依赖后会自动升级（装了什么就用什么，没装就降级，日志与 `/health` 都会如实报告）：

| 组件 | 装了就用 | 没装就用 | 效果差异 |
| --- | --- | --- | --- |
| 向量模型 | `sentence-transformers` + `BAAI/bge-small-zh-v1.5` | 内置哈希向量（CJK 二元组 + signed hashing，1024 维） | 语义泛化能力差距最大的一环，也是消融表里最该量化的一行 |
| 计算设备 | `device: auto` → 有 CUDA/MPS 就用 GPU（`/health` 会报告实际选择） | 没装 torch → CPU | 只影响速度；设备切换不会触发索引重建（向量结果与设备无关） |
| 向量检索 | `faiss`（IndexFlatIP） | numpy 矩阵乘暴力检索 | 十万块以内基本无差异（几毫秒） |
| 重排 | `sentence-transformers` CrossEncoder（bge-reranker-base） | 词法重排（覆盖率 + 短语命中 + 长度惩罚） | 词法重排能吃掉一部分收益，是廉价的 baseline |
| 生成 | 任何 OpenAI 兼容 `/chat/completions`（设 `OPENAI_API_KEY`） | 抽取式生成（命中原句拼接 + 引用） | 抽取式可用来判断"错在检索还是错在生成" |
| HTTP | `fastapi` + `uvicorn` | 标准库 `http.server` | 接口完全一致，前端无需改动 |
| 配置 | `pyyaml` | 内置默认配置（与 `configs/default.yaml` 内容一致） | 缺 pyyaml 时只会打印一行提示，不会报错退出；也可删掉 yaml、放一份 `configs/default.json` |

---

## 2. 五条命令

```bash
python -m rag.cli ingest data/corpus                            # 1) 建索引
python -m rag.cli ask "RRF 的常数 k 取多少？" --debug             # 2) 单次问答（含各通道候选）
python -m rag.cli eval --dataset data/eval/qa.jsonl --tag baseline  # 3) 评测并落盘
python -m rag.cli serve --port 8000                             # 4) 起服务 + 网页 demo
python -m rag.cli info                                          # 5) 查看生效配置与组件状态
```

其他入口：

```bash
python -m rag.cli eval --tag +rerank --set retrieval.rerank.enabled=true   # 消融：打开重排
python scripts/ablation.py --list                                          # 查看全部消融行（A~J）
python scripts/ablation.py                                                 # 批跑 A~J 并打印对比表
python scripts/ablation.py --heavy                                         # 追加 G(Cross-Encoder) / H(真实 LLM)
python scripts/smoke_offline.py                                            # 零依赖冒烟自检（推荐先跑）
python scripts/scan_refusal_threshold.py                                   # 扫离线拒答阈值（网格搜索）
python scripts/check_heavy_deps.py                                         # 重依赖自检：BGE/CE/faiss/FastAPI 是否真能用
python -m unittest discover -s tests -v                                    # 单元测试（95 条）
```

Windows 下推荐用包装脚本（自动找 venv、设置 HF_HOME 与 UTF-8 输出）。**如果执行策略拦住 `.\run.ps1`
（报 `SecurityError`，且 `Set-ExecutionPolicy` 不生效，通常是有更上层的策略），就用 `-ExecutionPolicy Bypass -File` 形式**：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1                          # 一次性装依赖（venv 建在 D:\rag-lab-venv）
powershell -ExecutionPolicy Bypass -File .\run.ps1 -m unittest discover -s tests
powershell -ExecutionPolicy Bypass -File .\run.ps1 -m rag.cli ask "混合检索为什么要用 RRF 融合？"
powershell -ExecutionPolicy Bypass -File .\run.ps1 scripts/check_heavy_deps.py  # 确认 BGE/CE/faiss/FastAPI 可用
```

也可以完全绕开脚本，直接调 venv 里的 python（等价）：

```powershell
$env:HF_HOME = "$PWD\.hf-cache"; $env:PYTHONUTF8 = "1"
& D:\rag-lab-venv\Scripts\python.exe scripts/check_heavy_deps.py
```

---

## 3. 架构：两条链路

```
                      ┌───────────────────────── 离线索引链路 ─────────────────────────┐
  语料(md/txt/pdf/docx)│  loader  ──►  normalize  ──►  chunker(structural/fixed/recursive) │
                      │                                    │                          │
                      │                    ┌───────────────┴───────────────┐          │
                      │                    ▼                               ▼          │
                      │            embedder.encode                 tokenize(CJK bigram)│
                      │                    │                               │          │
                      │             vector store(.npy)                BM25 倒排索引     │
                      └────────────────────┴───────────────┬───────────────┴──────────┘
                                                           ▼
                                                  .index/  (vectors.npy + chunks.jsonl
                                                            + bm25.json + meta.json)
                      ┌───────────────────────── 在线问答链路 ─────────────────────────┐
  问题 ──► rewriter ──► 多查询（keyword / multi_query / hyde）                          │
              │                     │                    │                            │
              │              vector search         BM25 search                       │
              │                     └────────┬───────────┘                            │
              │                       RRF 融合（k=60，加权）                            │
              │                              │                                        │
              │                     rerank（none / lexical / cross-encoder / llm）       │
              │                              │                                        │
              └──────────────► generator（强制 [n] 引用 / 允许拒答）                     │
                                             │                                        │
                                      Answer + citations + latency_ms + debug          │
                      └───────────────────────────────────────────────────────────────┘
```

多跳（`multihop.enabled=true`）时，第一轮检索出的 Top 片段会被用来构造"桥接查询"
（离线用未在问题中出现的关键词，有 LLM 时让 LLM 生成后续问题），第二轮结果与第一轮
再做一次 RRF 聚合，最后统一重排。

---

## 4. 目录结构

```
rag-lab/
├─ PLAN.md                  # 方案文档：选型、架构、评测、12 周计划、答辩话术
├─ README.md                # 本文档
├─ pyproject.toml           # 可 pip install -e .（可选）
├─ requirements.txt         # numpy（唯一必需依赖）
├─ requirements-optional.txt# 可选依赖分组说明
├─ configs/default.yaml     # 全部配置项与注释
├─ data/
│  ├─ corpus/*.md           # 示例语料（5 篇 RAG 技术笔记，doc_id = 文件名主干）
│  └─ eval/qa.jsonl         # 评测集（12 题：factual/comparison/multihop/unanswerable）
├─ rag/
│  ├─ cli.py                # 命令行：ingest / ask / eval / serve / info
│  ├─ config.py             # 默认值 <- 文件 <- --set 覆盖；路径统一相对项目根
│  ├─ types.py              # Chunk / Scored / Citation / Answer / RetrieveResult
│  ├─ text.py               # 中英混排分词（CJK 单字+二元组）、句切分、关键词、哈希
│  ├─ obs.py                # 日志与阶段计时
│  ├─ pipeline.py           # 索引生命周期 + 组件装配 + 一次问答的编排
│  ├─ ingest/               # loader（md/txt/json/jsonl/csv/pdf/docx）+ chunker
│  ├─ index/                # embedder（hashing/bge/openai）+ bm25（自写）+ store（numpy/faiss）
│  ├─ retrieve/             # rewrite + hybrid（RRF 融合、多跳）+ rerank（4 档）
│  ├─ generate/             # llm（OpenAI 兼容）+ prompt + answer（LLM/抽取式 + 引用解析）
│  ├─ eval/                 # metrics（8 项）+ run_eval（报告落盘与 ablation.csv）
│  └─ api/                  # simple_server（标准库）+ main（FastAPI，接口一致）
├─ scripts/
│  ├─ ablation.py           # 消融实验批跑（A~J，逐行追加 ablation.csv）
│  ├─ scan_refusal_threshold.py # 拒答阈值扫描（网格搜索离线判据）
│  ├─ smoke_offline.py      # 零依赖冒烟自检（端到端两阶段）
│  └─ check_ps_syntax.ps1   # PowerShell 脚本语法自检（纯 ASCII，避免自身踩编码坑）
├─ tests/                   # unittest：分词/分块/BM25/RRF/重排/指标/配置/生成桩/格式/端到端（109 条）
│  └─ support.py            # 测试临时目录（不用 tempfile，见第 10 节）
├─ setup.ps1 / run.ps1      # 依赖安装 / 命令包装（venv 默认建在沙箱之外的 D:\rag-lab-venv）
└─ web/index.html           # 单文件前端（无 CDN、无构建、可 file:// 打开）
```

---

## 5. 几个关键设计（面试最常被追问的点）

### 5.1 为什么用 RRF 而不是分数加权？

向量余弦相似度（0~1）与 BM25 分数（0~几十，且随语料规模漂移）不在同一量纲。分数加权必须
先归一化，而归一化方式（min-max / z-score）本身又引入超参，且在不同查询上不稳定。RRF 只用排名：

```
score(d) = Σ_r  w_r / (k + rank_r(d))        # 默认 k = 60
```

`k` 越小越强调"两路都排前面"的共识文档；`k=60` 是最广泛沿用的默认值。

### 5.2 手算示例（可直接作为面试白板题）

两路召回，`k = 60`：

| 文档 | 向量通道名次 | BM25 通道名次 | RRF 分数 |
| --- | --- | --- | --- |
| A | 1 | 未召回 | 1/61 = 0.016393 |
| B | 2 | 1 | 1/62 + 1/61 = 0.032522 |

**B 反超 A**：单路第一不一定赢，两路都靠前才是强共识信号。这正是融合的价值，也是
`tests/test_retrieve.py` 里被断言的性质。

### 5.3 chunk_id 稳定可复现

`chunk_id = <doc_id>#<序号>`，`doc_id` 取文件名主干，序号在文档内从 0 递增；
分块是确定性的（无随机、无时间依赖）。因此评测结论可以逐条回到具体片段，
换机器、换进程重跑结果不变。索引元数据里同时记录：

- `embedder.name`：换了向量模型 → 索引自动重建（否则会用错误的向量空间检索，**这是最隐蔽的一类 bug**）；
- `chunking`：分块参数变化 → 重建；
- `corpus_inputs` + `corpus_signature`：记录索引实际依据的语料来源与每个文件的内容指纹
  （`绝对路径:blake2b`）。语料内容变化 → 重建；`POST /ingest` 用自定义目录建过索引后，
  后续启动会继续以那份语料为准（而不是悄悄换回 `paths.corpus`），要切换语料重新跑一次 `ingest` 即可。

> 因此 `ingest` 始终是**全量重建**（不做增量）。元数据里已经存了每个文件的 `fingerprint`，
> 增量更新留给进阶项：只需要比对指纹、只重算变化文档的向量并更新倒排表。

### 5.4 引用与拒答是硬约束

提示词要求"每句结论必须带 `[n]`、资料不足必须说资料不足"；生成层再把 `[n]` 解析回
`(doc_id, chunk_id, quote, score)`。评测里单列两个指标：

- `citation_precision`：引用落在标注文档里的比例 —— 专门抓"答案对、出处编"的幻觉；
- `refusal_accuracy`：该拒答的拒答、该回答的不拒答 —— 抓拒答策略过/欠保守。

---

## 6. 评测与消融

指标实现见 `rag/eval/metrics.py`（纯函数，可单测）：

| 指标 | 定义 | 关注什么 |
| --- | --- | --- |
| Recall@k | gold 文档中出现在前 k 个 chunk 所属文档（去重）里的比例 | 该找的资料有没有捞回来 |
| Precision@k | 前 k 个 chunk 中属于 gold 文档的比例 | 候选集噪声 |
| MRR | 首个命中 gold 的 chunk 名次倒数 | 首个正确结果是否靠前 |
| nDCG@k | 二值相关性 DCG@k / IDCG@k（**按 doc 去重后**计算，避免同篇多块重复计分把 nDCG 抬到 1 以上） | 排序质量（同一查询下比较重排器） |
| citation_precision / recall | 引用落在 gold 内 / gold 被引用到的比例 | 出处是否可核验 |
| refusal_accuracy | 意图（该答 / 该拒）判断正确的比例 | 拒答策略是否合理 |
| latency_p50 / p95 | 端到端耗时分位 | 工程可用性 |

```bash
python -m rag.cli eval --tag baseline            # 落盘 reports/eval-*.{json,md} + 追加 ablation.csv
python scripts/ablation.py                       # 批跑 A~J 并打印对比表
python scripts/scan_refusal_threshold.py         # 扫离线拒答阈值（覆盖率 × 命中词数 网格）
```

### 6.1 已实测的 8 行（可离线复现，`python scripts/ablation.py`）

条件：5 篇示例语料 → 39 个分块、12 题评测集、哈希向量、抽取式生成器、`structural(512/96)`。

| 实验 | Recall@1 | Recall@5 | MRR | nDCG@10 | 引用精确率 | 拒答准确率 | P50(ms) | P95(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A 仅向量 | 0.90 | 0.95 | 1.00 | 0.9839 | 0.567 | 1.00 | 1.0 | 1.1 |
| B 仅 BM25 | 0.90 | 0.95 | 1.00 | 0.9920 | 0.667 | 1.00 | 1.1 | 1.4 |
| **C +RRF 融合** | 0.90 | **1.00** | 1.00 | **0.9920** | 0.667 | 1.00 | 1.2 | 1.3 |
| D +查询改写(keyword) | 0.90 | 0.95 | 1.00 | 0.9613 | 0.583 | 1.00 | 1.2 | 1.6 |
| E +词法重排(IDF 加权) | 0.80 | 0.95 | 0.95 | 0.9508 | 0.650 | 1.00 | 2.9 | 3.5 |
| F +多跳(2 跳) | 0.80 | 0.95 | 0.95 | 0.9470 | 0.617 | 1.00 | 3.6 | 4.2 |
| I 固定窗口分块(在 F 上换) | 0.80 | 0.95 | 0.95 | 0.9508 | 0.717 | 1.00 | 3.9 | 4.3 |
| J 仅向量 + 词法重排 | 0.80 | 0.95 | 0.95 | 0.9470 | 0.650 | 1.00 | 2.7 | 3.2 |

**三条真实结论**（比"每一层都提升"更值得讲，也是复试里最容易被追问的地方）：

1. **只有 C（混合检索 + RRF）是稳定正向**：Recall@5 0.95→1.00、引用精确率 0.567→0.667。
2. **改写（D）与词法重排（E/J）在这个语料上是负收益**，重排还把 P95 拉高约 3 倍。
   原因：语料只有 5 篇且术语高度集中，BM25 已把词面信号用尽，改写的扩展词反而稀释了
   融合排名；而词法重排的打分本身就是「查询词覆盖」的函数，与 BM25 高度相关、属重复信息。
   **修复过程是加分项**：最初等权覆盖率让 Recall@1 从 0.90 掉到 0.70，改成 **IDF 加权**
   后回到 0.80、nDCG@10 从 0.914 升到 0.951，但仍未超过纯 RRF 的 0.992 —— 因此默认配置
   保持 `retrieval.rerank.enabled: false`（要试就 `--set retrieval.rerank.enabled=true`）。
3. **多跳（F）没有增益**（0.947 vs C 的 0.992），符合"延迟涨了指标没涨 → 早停太松或
   扩查询质量差"的预期分支：离线桥接在 39 个分块的小语料上很容易跳到无关方向。

**拒答判据**：`scripts/scan_refusal_threshold.py` 做网格搜索，`0.10/1` 与 `0.10/2` 并列最优
（12/12），取更严的 `min_hits=2` 作为发货配置。但**余量只有 0.013，且最弱样本恰好压在阈值
边界上**——可回答题里最弱的 q08 覆盖率 0.100（正好等于 `min_coverage`），不可回答题里最强的
q12 是 0.087。这是明确的过拟合警告：扩到 60 题以上必须重扫；而且"词面覆盖率"本身是偏弱的
可回答性判据（更强的做法是向量相似度阈值或 LLM/NLI 判定，见 PLAN.md 第 16 章）。

> 评测集只有 12 题，单题差异噪声很大，**不要据此宣称显著性**：结论表述建议写成
> "在自建 12 题小样本上观察到的趋势"，并说明需要扩到 60 题以上（见 PLAN.md 第 9 章）。
> 上表的用途是**证明流程可跑通 + 展示负结果的处理方式**，不是效果结论。

### 6.2 换成 BGE 向量（GPU）后的同一张表

`python scripts/ablation.py --embedder bge --tag-suffix=-bge`（装好 `sentence-transformers` 后，
默认 `embedder.provider=auto` 就会走 BGE，因此这也代表默认配置；本机 RTX 4070 Laptop，
`/health` 会显示 `device.selected=cuda`）。

| 实验 | Recall@1 | Recall@5 | MRR | nDCG@10 | 引用精确率 | 拒答准确率 | P50(ms) | P95(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A 仅 BGE 向量 | 0.90 | 0.95 | 1.00 | **1.0000** | **0.900** | 1.00 | 6.6 | 10.4 |
| B 仅 BM25 | 0.90 | 0.95 | 1.00 | 0.9920 | 0.667 | 1.00 | 5.6 | 21.1 |
| **C +RRF 融合** | 0.90 | **1.00** | 1.00 | **1.0000** | 0.717 | 1.00 | 3.8 | 18.9 |
| D +查询改写(keyword) | 0.75 | 1.00 | 0.875 | 0.9202 | 0.650 | 0.917 | 7.2 | 27.6 |
| E +词法重排(IDF 加权) | 0.90 | 1.00 | 1.00 | 0.9920 | 0.733 | 1.00 | 8.4 | 27.2 |
| F +多跳(2 跳) | 0.90 | 1.00 | 1.00 | 0.9920 | 0.733 | 1.00 | 12.6 | 27.6 |
| G F + Cross-Encoder 重排 | 0.90 | 1.00 | 1.00 | 0.9920 | 0.650 | 1.00 | **204.0** | 264.9 |
| I 固定窗口分块(在 F 上换) | 0.80 | 0.95 | 0.95 | 0.9551 | 0.783 | 1.00 | 6.1 | 8.6 |
| J 仅 BGE 向量 + 词法重排 | 0.90 | 0.95 | 1.00 | 0.9920 | 0.750 | 1.00 | 7.5 | 10.0 |
| K RRF + Cross-Encoder 重排 | 0.90 | 1.00 | 1.00 | 0.9920 | 0.650 | 1.00 | **203.0** | 255.2 |

命令：`python scripts/ablation.py --embedder bge --only A,B,C,D,E,F,G,I,J,K --heavy --tag-suffix=-bge`
（`reports/ablation.csv` 每行都记录 `setting.effective.*`，可确认没有静默降级。）

**五条结论**：

1. **换向量模型是单项收益最大的改动**：引用精确率 0.567 → **0.900**、nDCG@10 0.9839 → **1.0000**
   （哈希 → BGE）。"排名看着还行"不等于"证据站得住"——引用精确率差了 33 个点。
2. **RRF 仍有召回增益**（Recall@5 0.95 → 1.00），但 nDCG@10 已到天花板，各层差距收敛。
3. **两种重排都是负收益，代价差 40 倍**：词法重排把 nDCG@10 从 1.0000 拉到 0.9920 只多花 4.6ms；
   **Cross-Encoder 同样 0.9920 却要多花约 200ms（P50 3.8 → 203ms，×53）**，引用精确率还从
   0.717 降到 0.650。根因是**天花板效应**：39 个分块 + BGE + RRF 已 nDCG@10 = 1.0000，
   重排没有提升空间；要验证重排价值必须扩语料/加难度，或在更弱的召回基线上评（在哈希基线上
   词法重排确实把 Recall@1 从 0.70 救回 0.80）。
4. **默认配置曾经正好等于负收益配置**：`rewrite.strategy=auto` 无 LLM 时退化成 `keyword`，
   默认配置的评测结果就是 D 行。现已改成"有 LLM 才改写，否则 identity"，与 C 行对齐。
   **教训：降级路径也要单独评测。**
5. **配置与实际生效可能不一致，必须让产物说话**：`--set embedder.provider=bge` 在模型缓存缺失时
   会静默降级为哈希向量（G/K 两行曾因此跑出"看着正常"的错误数字）。现在评测报告与
   `ablation.csv` 都记录 `effective.embedder / reranker / generator / device`，并对降级发显式 WARNING。

另有两项实测结论：

- **拒答阈值随检索质量变化**：换成 BGE 后 q08 覆盖率 0.100→0.133，与 q12（0.087）的余量从
  0.013 扩到 0.046，`0.10/0.12 × 1/2` 四组都给出 12/12。阈值是"当前检索质量"的函数，
  换模型/换分块/扩评测集都要重扫（`scripts/scan_refusal_threshold.py`）。
- **向量后端等价**：`index.backend=numpy` 与 `faiss` 在同一 BGE 索引上指标完全一致
  （1.0 / 1.0 / 1.0），faiss P50 略快（4.1ms vs 4.8ms）；显式切换后端会触发索引重建。

---

## 7. 配置与 `--set` 覆盖

优先级：`rag/config.py` 内置默认值 → `configs/default.yaml` → 命令行 `--set a.b.c=v`。
所有相对路径都相对于项目根目录，可以从任何工作目录启动。

最常用的覆盖项：

```bash
--set chunking.strategy=fixed                    # 分块策略消融
--set chunking.max_chars=256                     # 更小的块
--set embedder.provider=bge                      # 强制使用 BGE（失败会降级并告警）
--set retrieval.weights.vector=0 --set retrieval.weights.bm25=1   # 只保留 BM25
--set retrieval.rewrite.strategy=hyde            # 换改写策略（需 LLM）
--set retrieval.rerank.enabled=true --set retrieval.rerank.provider=lexical
--set multihop.enabled=true --set multihop.max_rounds=3
--set generation.provider=openai --set generation.model=gpt-4o-mini
--set index.backend=faiss                        # 强制 faiss（不可用则退回 numpy）
```

---

## 8. HTTP 接口

```bash
python -m rag.cli serve --port 8000     # 默认 auto：有 fastapi 用 fastapi，否则标准库
# 浏览器打开 http://127.0.0.1:8000/  （页面来自 web/index.html）
```

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` `/status` | 索引与组件状态（embedder / reranker / llm / generator / fusion） |
| POST | `/query` | `{"question": "...", "top_k": 5}` → `{answer, citations, refused, latency_ms, debug, contexts}` |
| POST | `/ingest` | `{"paths": ["data/corpus"]}` → 重建索引报告（总是全量重建，并自动重新装配组件） |
| GET | `/` | `web/index.html` |

`web/index.html` 是零依赖单文件页面：引用角标可跳转高亮、延迟分解条形图、
三通道候选列表、组件状态徽标；也可以直接双击用 `file://` 打开，在页面顶部填写后端地址。

---

## 9. 测试与自检

```bash
python scripts/smoke_offline.py          # 端到端冒烟：建索引 -> 问答 -> 评测，逐项 PASS/FAIL
python -m unittest discover -s tests -v  # 单元测试（需在 rag-lab 目录下执行）
```

测试覆盖：分词与句切分、结构感知/固定窗口分块（含重叠与稳定性）、BM25 排序与
存取往返、RRF 融合的三条性质、词法重排能否把相关片段顶上来、指标公式（含 nDCG 手算值）、
语料加载与 **doc_id 唯一性**、哈希向量与向量库（含 1-D 入参与零维保护、faiss 回落）、
**配置层**（YAML 与内置默认值逐键对齐、显式 `--config` 解析失败必须报错、自动发现的坏配置
退回默认值、`--set` 覆盖）、**生成层**（用桩 LLM 验证引用编号映射、越界引用被丢弃、
LLM 异常降级、拒答判定）、**格式层**（md/txt/json/jsonl/csv/docx，含 docx 端到端）、
设备解析（`auto` → cpu/cuda/mps），以及**全部离线组件**的端到端闭环、索引复用、
语料变化自动重建、`ingest()` 后重新装配。

---

## 10. 已知局限（主动说出来的局限，比被问出来更加分）

1. **示例语料只有 5 篇自写笔记**，评测集 12 题：结论只能作为方法论演示，不能当作效果证据；
   正式使用请替换为真实领域语料并把评测集扩到 60 题以上。
2. **离线哈希向量不是神经向量**：它对词面重合敏感、对同义改写不敏感，因此
   "向量通道"在离线模式下的表现被低估；打开 BGE 后消融表里的 C/D 行差异会变化。
3. **抽取式兜底答案可读性有限**：它是命中原句的拼接，不做改写；它的价值在于把
   "检索问题"与"生成问题"分离开，而不是替代 LLM。
4. **多跳是启发式的**：桥接查询靠关键词或 LLM 生成，没有做证据充分性判断与自我纠错
   （Self-RAG / CRAG 那类"检索-评判-再检索"的闭环留给进阶项）。
5. **没有做权限、缓存、限流、增量索引**：`/ingest` 是全量重建；生产化需要按文档
   指纹做增量更新（元数据里已经记录了 `fingerprint`，接口留好了）。
6. **BM25 索引常驻内存**：倒排表用 JSON 存盘，百万级语料应换成更紧凑的结构或数据库。

### 交付前静态审查发现并已修复的缺陷（留档，说明"我是怎么验的"）

代码无法在本机执行（会话 shell 不可用），因此交付前做了两路独立静态审查 + 补齐回归用例。
审查确认的 7 个真实缺陷都已修复，并且每一条都新增了对应的测试：

| 缺陷 | 后果 | 修复 |
| --- | --- | --- |
| 缺 pyyaml 时 `load_config` 直接抛错 | "只装 numpy 就能跑"这条主线在 CLI 入口失效 | 读取配置失败时打印提示并退回内置默认值（与 YAML 内容一致） |
| `reranker` 未注入 `HybridRetriever` | 重排开关与 `+rerank` 消融整行失真，且状态自相矛盾 | 装配时传入 `reranker=`，并断言 `debug["reranker"]` 与组件状态一致 |
| `corpus_signature` 恒按默认语料目录计算 | 自定义语料变了不重建、默认语料变化反而冲掉自定义索引 | 元数据记录 `corpus_inputs`，签名与重建都基于"索引实际依据的语料" |
| `ingest()` 后不重置装配状态 | 同进程重建索引后仍用旧 `store`/`chunk_map` 回答 | `ingest()` 末尾置 `_built = False`，并加了"换语料后能检索到新文档"的用例 |
| `_doc_id_for` 的 id 撞车 | 同名 doc_id → 同名 chunk_id → 检索静默指向错误文档 | 生成的 `stem-n` 也登记占用，用例覆盖 `a.md` / `a-1.md` / `sub/a.md` |
| `HashingEmbedder(dim=0)` | 取模 ZeroDivisionError | `dim` 夹到最小值 1 |
| Cross-Encoder 的"预热"是空操作 | 模型加载失败会推迟到 ask 期未捕获抛出 | 先加载模型再判空候选，降级分支真正可达 |
| nDCG@k 的 DCG 按 chunk 累加、IDCG 只按文档数算 | 同一文档多块命中时 **nDCG > 1**（可达 1.63），消融表与报告不可信 | DCG 前按 doc 去重，并加 `min(1.0, …)` 兜底 |
| `recall@k` 先去重再截断 | 与 `precision@k` 候选集口径不一致，多块文档系统性抬高 Recall@k | 先截断前 k 个 chunk，再去重成文档集合 |
| 前端引用卡片用数组下标当 `[n]` | 答案不按 `[1][2][3]` 顺序引用时，点角标跳到错误卡片 | 卡片 id 与角标统一用 `Citation.index`，正文按 index 建映射 |
| 前端把 `components.<name>` 对象当字符串 | 徽标与调试面板显示 `[object Object]` | 新增 `compName()`，取 `name`/`kind` |
| `do_GET` 无异常兜底、静态资源用裸前缀比较 | 异常时前端只看到 "Failed to fetch"；同前缀兄弟目录可被越权读取 | 整段 try/except 返回 500；改用 `root + os.sep` 比较 |
| `Content-Length` 未判负值、超限时未关连接 | 负值请求把连接挂死；超限后剩余字节被当成下一个请求解析 | 负值直接拒绝；超限时 `close_connection = True` |
| 配置解析失败被一律静默吞掉 | 显式 `--config` 写错会"看起来跑成功"，实际配置没生效 | 显式路径解析失败必须抛错；只有自动发现的默认配置才退回内置默认值 |

### 真正跑起来之后又发现的 3 个问题（静态审查看不到的那一类）

| 问题 | 表现 | 修复 |
| --- | --- | --- |
| `store.add()` 先比数量再规整形状 | 传入 1-D 向量（`encode_query` 的返回值）时，`len(vectors)` 是维度数而非 1，被误判成"数量不一致" | 先 `reshape`/空判，再比数量（[store.py](rag-lab/rag/index/store.py)） |
| 测试与冒烟脚本依赖 `tempfile` | 在受限令牌环境（沙箱 / CI / 部分容器）下 `mkdtemp` 建的目录连创建者都无法写入（`WinError 5`），13 个用例直接报错 | 新增 `tests/support.py` 的 `TempWorkspace`，在项目内 `.testtmp/` 自建临时目录；顺手让测试不再依赖系统 TEMP |
| 抽取式生成器的拒答判据用错了信号 | 用"长度 × 排名先验 × 覆盖率"的合成分数做阈值，会把"排名第一但只是词面沾边"的句子推过阈值 → **不可回答的题被作答**；同时把标题行、表格行当答案句子 | ① 剥掉分块时拼入的标题路径、过滤 Markdown 结构行；② 拒答改用 **单句查询词覆盖率 + 命中词数**，阈值由 `scripts/scan_refusal_threshold.py` 扫描确定（回填 [configs/default.yaml](rag-lab/configs/default.yaml)，含余量不足的警告） |
| `index.backend` 不在索引身份检查里 | 装好 faiss 后 `--set index.backend=faiss` **静默复用旧 numpy 索引** → "faiss vs numpy 对比实验"其实是同一个后端跑了两次，指标"完全一致"是假的 | 把后端纳入 `_index_matches()` 检查（`auto` 不迁移、显式指定必须与磁盘一致），并重跑一致性验证 |
| 默认配置正好等于负收益配置 | `rewrite.strategy=auto` 在无 LLM 时退化成 `keyword` 压缩，于是"默认配置跑评测"拿到的是 D 行数字（0.75 / 0.9202 / 拒答 0.917），而 C 行是（0.90 / 1.0000 / 1.0） | `auto` 改为"有 LLM 才改写，否则 identity"；`keyword` 保留为显式消融选项。**降级路径也会改变结果，必须单独评测** |
| 配置与实际生效不一致（静默降级） | `--set embedder.provider=bge` 在模型缓存缺失/离线时会**降级成哈希向量**：评测照跑、报告"看着正常"，数字却来自另一套实现（G/K 两行就是这么跑出错误结论的） | 评测报告、Markdown 与 `ablation.csv` 现在都记录 `effective.embedder / reranker / generator / device`，并对降级发显式 WARNING。**凡是"配置写了什么"可能不等于"实际生效什么"的地方，都必须把后者写进产物** |
| `run.ps1` 的两个 PowerShell 陷阱 | ① `$candidates[0]` 取到字符串首字符 → venv 路径变成 `D:\Scripts\python.exe`（单元素管道退化成标量）；② `param([string[]] $CommandArgs)` 因参数名前缀匹配把 python 的 `-c` 吃掉 | ① 管道结果一律用 `@()` 包成数组；② 去掉 `param()`，改用 `$args` 原样透传。已在 [run.ps1](rag-lab/run.ps1) 头部写明原因 |
| `.ps1` 必须存成 UTF-8 **带 BOM** | PowerShell 5.1 读无 BOM 脚本会按系统 ANSI(GBK) 解码，中文字符错位后**把字符串结尾的 `"` 当成双字节的第二个字节吃掉**——表现为"某行缺引号"的假语法错误（第 101 行背了锅，其实元凶在第 97 行的中文） | 两个脚本都存为带 BOM；新增 [check_ps_syntax.ps1](rag-lab/scripts/check_ps_syntax.ps1)（纯 ASCII）供随时自检 |

另外修掉若干细节：`store.meta` 属性位置、`add()` 的 1-D 向量、`_phrase_hit` 的 off-by-one、
重叠窗口按剩余预算裁剪（既不超 `max_chars` 也不丢重叠）、`aggregate()` 不再把缺失耗时的记录
按 0 ms 计入分位、词法重排改为 **IDF 加权覆盖率**（等权覆盖率会让中文常见字词淹没相关词）、
两处死代码与两处未使用导入。

---

## 11. 与"教程式 RAG"的差异

| 常见教程做法 | 本项目做法 | 为什么 |
| --- | --- | --- |
| `LangChain + Chroma` 一把梭 | 自己写分块、BM25、RRF、重排、评测与降级 | 面试考的是"你懂不懂链路"，黑盒无法回答"为什么 k=60""召回为什么掉了" |
| 只用向量召回 | 向量 ∥ BM25 + RRF，双路可比对 | 中文语料里参数名、编号、缩写必须词面命中 |
| 直接 `top_k=3` 送 LLM | 召回 20~50 → 融合 → 重排 → 送 3~6 | 召回与排序分开优化，才能定位瓶颈 |
| 只演示"能回答" | 8 项指标 + 逐条明细 + 消融 CSV | 没有度量就没有工程结论 |
| 强依赖模型/Key 才能跑 | 缺失依赖自动降级并如实上报 | 演示不翻车，评测结论可复现 |
| 引用靠"看起来对" | `[n]` → `(doc_id, chunk_id, quote)` 映射 + 引用精确率 | 幻觉最典型的形态是"答案对、出处编" |

---

## 12. 与 Cursor / VSCode 集成

集成分两层：**在编辑器里开发**（调试/测试/任务一键跑）与**把 RAG 当工具接进编辑器 AI**（MCP）。

### 12.1 开发配置：`.vscode/`

| 文件 | 内容 |
| --- | --- |
| `settings.json` | 解释器指向 `D:\rag-lab-venv\Scripts\python.exe`；unittest 发现配置；**集成终端自动注入 `HF_HOME` / `PYTHONUTF8` / `HF_HUB_OFFLINE`**（与 `run.ps1` 一致，避免"脚本能跑、终端里跑不了"）；排除 `.hf-cache`/`.testtmp` 等噪音 |
| `launch.json` | 7 个调试配置：`rag: ask` / `ingest` / `eval` / `serve(FastAPI)` / **`MCP 服务器`** / 当前测试文件 / 全部测试 |
| `tasks.json` | 12 个一键任务：重建索引、提问、评测（v2 集）、**消融实验（BGE + 全部行）**、全部测试、冒烟自检、重依赖自检、导入文档目录、生成评测候选（启发式/LLM）、起服务、拒答阈值扫描 |
| `extensions.json` | 推荐扩展：Python/Pylance/debugpy、Ruff、PowerShell、YAML、Markdown、**GitHub Copilot Chat**（用 MCP 需要它） |
| `mcp.json` | 把 RAG 作为 MCP 工具接进 VSCode Copilot Chat |

用编辑器打开 `rag-lab` 文件夹即可：跑默认测试任务（`Ctrl+Shift+B`）或在 `F5` 里选 `rag: ask` 直接断点调试问答链路。

### 12.2 把 RAG 接进编辑器 AI（MCP 服务器）

[rag/mcp/server.py](rag-lab/rag/mcp/server.py) 是一个**零依赖的 MCP（Model Context Protocol）stdio 服务器**——
协议本身就是"一行一个 JSON-RPC 消息"，所以没有引入任何第三方包：

```bash
python -m rag.mcp.server                 # 手动起（调试用）
python -m rag.mcp.server --allow-ingest  # 额外允许通过 MCP 重建索引
```

暴露 4 个工具：

| 工具 | 作用 | 是否花钱 |
| --- | --- | --- |
| `kb_stats()` | 索引状态与**实际生效**的组件（向量模型/重排器/生成器/设备） | 否 |
| `kb_search(query, top_k)` | 混合检索（向量 + BM25 + RRF，可重排），返回 chunk_id、分数与原文 | 否 |
| `kb_answer(question, top_k)` | 完整问答，带 `[n]` 引用与耗时；资料不足会明确拒答 | 配置了 LLM 才花钱 |
| `kb_ingest(paths)` | 全量重建索引（默认禁用，需 `--allow-ingest`） | 否 |

接线模板已放进仓库（换机器只需改里面的绝对路径）：

- **Cursor**：[.cursor/mcp.json](rag-lab/.cursor/mcp.json) → 重启 Cursor 或 Settings → MCP 点 Refresh，看到 `rag-lab` 变绿即可。
- **VSCode**：[.vscode/mcp.json](rag-lab/.vscode/mcp.json) → 命令面板 `MCP: List Servers` → 启动 `rag-lab` → 在 Copilot Chat 的 Agent 模式里让它调用这些工具。

接好后可以直接在编辑器里说：

> 用 rag-lab 查一下"RRF 的常数 k 取多少"，并把引用出处贴给我。

排错三条（都实际踩过）：

1. **stdout 必须只有协议消息**：混进任何日志都会让客户端解析失败。本项目日志一律走 stderr，
   并且有单元测试锁死这条约束（`test_stdio_output_is_protocol_only`）。
2. **首次调用才加载模型**：`kb_stats` 之前不会加载 BGE（惰性构建），以免拖慢编辑器启动；
   所以第一次 `kb_search` 会有几秒到十几秒的加载时间。
3. **`PYTHONPATH` 要指向项目根**：编辑器启动 MCP 进程时的 cwd 不一定等于项目根，模板里用
   绝对路径显式指定了。

---

## 13. 语料与评测集扩建（打破评测天花板）

5 篇手写语料会把所有指标顶到 `nDCG@10 = 1.0000`，此时**任何组件差异都测不出来**——这是本轮
最值钱的发现。扩建分三步，三个脚本各管一段：

```bash
# 1) 导入你自己的文档（md/txt/pdf/docx/json/jsonl/csv 都支持，可选按标题层级拆章）
python scripts/build_corpus.py "D:\\我的资料" --out data/corpus --split-level 2 --dry-run
python scripts/build_corpus.py "D:\\我的资料" --out data/corpus --prefix mydocs

# 2) 快速补充公开语料（带来源与许可登记；本例 = 5 篇手写 + 43 篇中文维基）
python scripts/fetch_public_corpus.py wikipedia --prefix wiki --limit 40 --delay 0.7 --titles 检索增强生成 向量数据库
python scripts/fetch_public_corpus.py arxiv --query 'abs:"retrieval augmented generation"' --limit 25

# 3) 半自动生成评测候选 → 人工在复核表勾选 → 回填评测集
python scripts/gen_eval_candidates.py generate --provider heuristic --per-doc 2 --max-docs 48
python scripts/gen_eval_candidates.py generate --provider llm --model deepseek-chat \
       --api-base https://api.deepseek.com/v1 --api-key-env DEEPSEEK_API_KEY --max-docs 30
python scripts/gen_eval_candidates.py promote --only-checked --qa data/eval/qa-v2.jsonl
python scripts/gen_eval_candidates.py stats
```

三条值得讲的取舍：

- **来源与许可单独登记**（`data/manifests/corpus.json`），**不写进正文**——写进正文会污染检索与生成；
- **去重分两层**：内容 SHA1 判完全重复、token Jaccard > 0.9 判近重复（维基条目互相抄段落很常见）；
- **"领域外问题"的判据要落在稀有词上**：最初用"IDF 加权覆盖率 < 0.3"，语料一扩到 48 篇就整体失效
  （长问题总会撞上一堆常见二元组，覆盖率全被抬到 0.5 上下，一条都产不出来）；改成
  **"主题实词在语料里出现 ≥2 次的比例"** 之后才分得开（领域内 0.71~1.00，领域外 0.00~0.40）。

### 13.1 扩建前后的真实对比（同一套 BGE + 真实 LLM 生成 + GPU）

| 实验（v2：48 篇 / 685 块 / 60 题） | R@1 | R@5 | MRR | nDCG@10 | 引用精确率 | P50 | P95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A 仅向量 | 0.931 | 0.974 | 0.963 | 0.9775 | 0.905 | 1402 | 2377 |
| B 仅 BM25 | 0.914 | 0.983 | 0.956 | 0.9695 | 0.917 | 1465 | 2259 |
| **C +RRF** | 0.931 | **1.000** | 0.971 | 0.9809 | **0.934** | 1453 | 2281 |
| D +改写 | **0.845** | 1.000 | **0.922** | **0.9433** | 0.928 | 1492 | 2387 |
| E +词法重排 | 0.966 | 1.000 | 0.991 | 0.9901 | 0.894 | 1508 | 3052 |
| **F +多跳** | **0.983** | 0.991 | **1.000** | **0.9960** | 0.862 | 2938 | 3552 |
| G F + Cross-Encoder | 0.957 | 1.000 | 0.983 | 0.9876 | 0.876 | 4329 | 15666 |
| I 固定窗口分块 | 0.948 | 0.991 | 0.983 | 0.9859 | 0.931 | 2745 | 3515 |
| **J 仅向量 + 词法重排** | 0.966 | 0.991 | 0.991 | 0.9923 | 0.917 | 1482 | 2756 |
| **K RRF + Cross-Encoder** | 0.966 | 1.000 | 0.991 | 0.9923 | 0.885 | 1740 | 2733 |

| 关键对比 | 旧（39 块 / 12 题） | 新（685 块 / 60 题） |
| --- | --- | --- |
| nDCG@10 天花板 | **1.0000**（无区分度） | 0.9775~0.9960（有区分度） |
| 词法重排 | **负**（1.0000 → 0.9920） | **正**（0.9809 → 0.9901，MRR 0.971 → 0.991） |
| Cross-Encoder（配 RRF） | 负，且慢 ×53 | **正**（0.9809 → 0.9923，仅 +287ms） |
| 分块策略（固定 vs 结构） | 几乎无差异 | **结构感知 +1 pt nDCG、+1.9 pt R@1** |
| 多跳 | 无收益 | **全场最佳**（nDCG 0.9960 / MRR 1.000，P50 ×2） |
| 关键词改写 | 负 | 负且更明显（R@1 0.931 → 0.845） |

**结论**：所谓"重排没用、多跳没用"是**小语料天花板造成的假象**。把语料从 39 块扩到 685 块、
题目从 12 道扩到 60 道（高难度的 comparison/multihop 占 46/60）之后，**词法重排、Cross-Encoder、
结构感知分块三者的收益全部翻正**，而"关键词改写"依然稳定为负——这才是能拿出去讲的结论。

> **两条必须一起说的代价**：① 多跳在配了真实 LLM 后，桥接查询会**多花一次 LLM 调用**，
> `G` 行 P95 因此飙到 15.7 s（v1 无 LLM 时完全看不到这个代价）；② 60 题里 48 题是 LLM 生成、
> **未经人工筛选**，按 PLAN 第 9 章标准只能算趋势证据，人工过一遍复核表
> （[candidates_review.md](rag-lab/data/eval/candidates_review.md)）后才是最终结论。

