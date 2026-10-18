# rag-lab · 领域知识库问答与证据溯源工程骨架

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 174 Passed](https://img.shields.io/badge/tests-174%20passed-brightgreen.svg)](tests/)
[![Dependency: Zero Heavy](https://img.shields.io/badge/dependencies-numpy%20only%20(core)-orange.svg)](requirements.txt)
[![MCP: Supported](https://img.shields.io/badge/MCP-JSON--RPC%202.0-purple.svg)](rag/mcp/server.py)

> **定位**：一个架构解耦、可渐进升级、自带严格评测体系与消融实验的高可靠 RAG（Retrieval-Augmented Generation，检索增强生成）工程骨架。
> 
> **核心特色**：**仅装 `numpy` 即可离线跑通全链路**（分块 → 索引 → 改写 → 向量 ∥ BM25 双路召回 → RRF 融合 → 重排 → 强制引用溯源与拒答 → 端到端指标评测）；装上重依赖（BGE / Cross-Encoder / Faiss / OpenAI 兼容 LLM / FastAPI）后**零代码修改、配置即插即用、平滑升级**。

配套的技术方案与选型论证文档见 [PLAN.md](PLAN.md)。

---

## 目录

- [1. 核心设计与特性](#1-核心设计与特性)
- [2. 快速上手（30 秒体验）](#2-快速上手30-秒体验)
- [3. 架构与数据流](#3-架构与数据流)
- [4. 常用命令清单](#4-常用命令清单)
- [5. 深入设计解析](#5-深入设计解析)
  - [5.1 为什么采用 RRF 倒数排名融合？](#51-为什么采用-rrf-倒数排名融合)
  - [5.2 结构感知分块与稳定指纹机制](#52-结构感知分块与稳定指纹机制)
  - [5.3 严格证据溯源与硬约束拒答](#53-严格证据溯源与硬约束拒答)
- [6. 评测指标与消融实验（实测结论）](#6-评测指标与消融实验实测结论)
  - [6.1 评测指标体系](#61-评测指标体系)
  - [6.2 60 题基准消融实验对比表（A~K）](#62-60-题基准消融实验对比表ak)
  - [6.3 关键实验结论与实证洞察](#63-关键实验结论与实证洞察)
- [7. 配置与命令行覆盖](#7-配置与命令行覆盖)
- [8. HTTP 服务与单文件前端](#8-http-服务与单文件前端)
- [9. 编辑器 AI 集成（原生 MCP 服务）](#9-编辑器-ai-集成原生-mcp-服务)
- [10. 工程实战踩坑与设计反思](#10-工程实战踩坑与设计反思)
- [11. 目录结构说明](#11-目录结构说明)

---

## 1. 核心设计与特性

| 维度 | 基础模式（零重依赖 / 离线运行） | 升级模式（安装可选依赖） | 设计与工程考量 |
| :--- | :--- | :--- | :--- |
| **向量模型** | 内置哈希向量（CJK 二元组 + 符号哈希，1024 维） | `BAAI/bge-small-zh-v1.5`（`sentence-transformers`） | 语义泛化能力的关键分水岭，消融表中重点量化 |
| **计算设备** | 纯 CPU 计算 | CUDA / MPS 自动探测与硬件加速 | 硬件变化不改变向量表征，无需重建索引；`/health` 自述设备状态 |
| **向量检索** | NumPy 纯矩阵乘（Cosine / IP 暴力检索） | `Faiss`（IndexFlatIP / 高维近邻检索） | 几十万规模内 NumPy 纯内积仅需数毫秒；大规模自动切换 |
| **关键词检索**| 纯 Python 手写 BM25（Okapi，倒排索引） | 无需额外依赖，原生支持 IDF 缓存与分词 | 针对高频词负分做平滑处理，免除第三方库侵入 |
| **结果重排** | 词法覆盖重排（IDF 加权覆盖率 + 短语命中 + 长度惩罚）| `BAAI/bge-reranker-base`（Cross-Encoder） | 词法重排作为零成本 Baseline，Cross-Encoder 提供精排上限 |
| **答案生成** | 离线抽取式生成（词面覆盖分析 + 原句抽取拼接） | 任意兼容 OpenAI 协议的 LLM（DeepSeek / GPT 等） | 离线模式用于剥离“错在检索”还是“错在生成”，定位流水线瓶颈 |
| **HTTP 服务**| Python 原生标准库 `http.server` | `FastAPI` + `uvicorn` | 接口规范严格一致，前端 demo 零感知透明适配 |
| **编辑器生态**| 原生 stdio MCP 服务器（零依赖 JSON-RPC 2.0） | 支持 Cursor / VSCode Copilot 作为 Agent 工具调用 | 将本地知识库作为标准 Context 工具接入主流 AI IDE |

---

## 2. 快速上手（30 秒体验）

无需 GPU、无需联网、无需配置 API Key，拉下代码即可跑通全链路：

```bash
# 1. 进入目录并安装基础依赖
git clone https://github.com/knightzeng8-web/rag-lab.git
cd rag-lab
pip install numpy

# 2. 离线构建知识库索引（使用内置哈希向量 + 手写 BM25）
python -m rag.cli ingest data/corpus

# 3. 发起一次检索与问答
python -m rag.cli ask "混合检索为什么要用 RRF 融合？"
```

**预期输出**：
- 结构化回答正文（带 `[1][2]` 显式引用出处）；
- 证据引用卡片（包含文档 ID、分块编号、得分、溯源原文）；
- 耗时指标明细（`rewrite` / `retrieve` / `rerank` / `generate` / `total` 阶段时延）。

### 升级到工业级组件（可选）

当你的环境中安装了对应的高级依赖后，系统会自动升级生效：

```bash
# 安装扩展依赖（支持 GPU 加速、BGE 向量、Cross-Encoder 与 FastAPI）
pip install torch sentence-transformers faiss-cpu fastapi uvicorn pyyaml

# 验证所有高级组件状态
python scripts/check_heavy_deps.py

# 配置 LLM API（支持任何 OpenAI 兼容端点）
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.deepseek.com/v1"

# 再次发起问答，体验神经检索与大模型生成的结合
python -m rag.cli ask "双编码器和交叉编码器在检索链路上有什么区别？"
```

> **Windows PowerShell 用户提示**：仓库附带了封装脚本 [run.ps1](run.ps1) 和 [setup.ps1](setup.ps1)，会自动处理 UTF-8 编码与虚拟环境路径。如遇执行策略拦截，可通过 `powershell -ExecutionPolicy Bypass -File .\run.ps1 <参数>` 执行。

---

## 3. 架构与数据流

整个系统清晰划分为**离线索引链路**与**在线问答链路**两条互不干扰的数据通道：

```
                      ┌───────────────────────── 离线索引链路 (Offline Ingestion) ─────────────────────────┐
  语料源文件           │  DocumentLoader  ──►  TextNormalize  ──►  Chunker (Structural / Fixed / Recursive)  │
  (md/txt/pdf/docx)   │                                                  │                                  │
                      │                                  ┌───────────────┴───────────────┐                  │
                      │                                  ▼                               ▼                  │
                      │                          Embedder.encode                 Tokenizer (CJK+Latin)      │
                      │                                  │                               │                  │
                      │                           Vector Store (.npy)              BM25 Inverted Index      │
                      └──────────────────────────────────┴───────────────┬───────────────┴──────────────────┘
                                                                         ▼
                                                              .index/ (Persistent Storage)
                                                              (vectors.npy + chunks.jsonl + bm25.json + meta.json)

                      ┌───────────────────────── 在线问答链路 (Online Query Pipeline) ──────────────────────┐
  用户问题 ──────────►│  QueryRewriter ──► 多重查询扩展 (Keyword / Multi-Query / HyDE)                        │
                      │         │                       │                     │                             │
                      │         │                Vector Search           BM25 Search                        │
                      │         │                       └────────┬────────────┘                             │
                      │         │                        RRF 融合 (k=60, 加权倒数排名)                       │
                      │         │                                │                                          │
                      │         │                    Reranker (Lexical / Cross-Encoder / LLM Listwise)      │
                      │         │                                │                                          │
                      │         └───────────────► Context Builder & Generator (强制 [n] 引用 / 自动拒答)    │
                      │                                          │                                          │
                      │                              Answer + Citations + Latency Breakdown                 │
                      └──────────────────────────────────────────┴──────────────────────────────────────────┘
```

- **多跳检索（Multi-hop）**：开启 `multihop.enabled=true` 时，系统基于第一轮检索出的证据片段提取桥接实体（离线模式利用 IDF 词频筛选，在线模式调用 LLM 构筑后续子问题），执行第二轮检索，并将两轮候选集再次融合重排，实现跨文档关联推理。

---

## 4. 常用命令清单

项目提供统一 CLI 入口：`python -m rag.cli <command>`：

```bash
# 1. 语料建库与索引重建
python -m rag.cli ingest data/corpus

# 2. 单次问答与详细通道排查（输出各通道 Top-10 结果）
python -m rag.cli ask "固定窗口分块与结构感知分块各自适合什么场景？" --debug

# 3. 启动本地 HTTP 服务与可视化前端（支持自动降级与健康检查）
python -m rag.cli serve --port 8000

# 4. 执行全量评测并导出报告
python -m rag.cli eval --dataset data/eval/qa-v2.jsonl --tag benchmark

# 5. 查看系统生效配置、加载设备（GPU/CPU）及各模块状态
python -m rag.cli info
```

### 实验与研发辅助工具

```bash
python scripts/smoke_offline.py          # 零依赖离线全链路冒烟测试（PASS / FAIL）
python scripts/check_heavy_deps.py       # 检查深度依赖（PyTorch / BGE / Faiss / CUDA 可用性）
python scripts/ablation.py               # 批量运行消融实验（生成对比表并追加到 reports/ablation.csv）
python scripts/scan_refusal_threshold.py # 网格搜索抽取式生成的拒答判定阈值
python -m unittest discover -s tests -v  # 执行完整单元测试集（174 个测试用例）
```

---

## 5. 深入设计解析

### 5.1 为什么采用 RRF 倒数排名融合？

在混合检索（Hybrid Search）中，向量召回与 BM25 召回的得分**不在同一量纲**：
- 向量余弦相似度取值在 $[0, 1]$ 之间；
- BM25 得分在 $[0, +\infty)$ 之间，且随着语料规模、文档长度分布严重漂移。

直接线性加权融合必须对分数做归一化（如 Min-Max 或 Z-Score），但这在检索中极不稳定：**单个异常高分的文档会压缩其他所有文档的得分分布**。

RRF（Reciprocal Rank Fusion）完全基于文档名次而不依赖原始分数：

$$\text{RRF}(d) = \sum_{m \in M} \frac{w_m}{k + \text{rank}_m(d)}$$

- $M$：召回通道集合（向量通道、BM25 通道等）；
- $w_m$：该通道的权重系数；
- $k$：平滑常量，业界标准默认值为 $60$。

#### 手算推导示例

假设 $k=60$，两个通道的表现如下：

| 候选文档 | 向量检索名次 | BM25 检索名次 | RRF 得分计算 | 最终得分 | 结论 |
| :---: | :---: | :---: | :--- | :---: | :---: |
| **Doc A** | 第 1 名 | 未进入 Top-20 | $\frac{1}{60 + 1} = 0.01639$ | 0.01639 | 单路拔尖 |
| **Doc B** | 第 2 名 | 第 1 名 | $\frac{1}{60 + 2} + \frac{1}{60 + 1} = 0.03252$ | **0.03252** | **双路共识（反超 A）** |

**Doc B 成功反超 Doc A**：这证明了 RRF 的核心价值——**优先奖励在多个通道中均取得靠前名次的“高共识度”文档**，有效抑制单通道的偶然噪声。

---

### 5.2 结构感知分块与稳定指纹机制

在长文档技术语料中，朴素固定窗口（Fixed Window）切分经常将同一个小节的内容腰斩，或丢失上下文归属。本项目实现三种分块策略（[rag/ingest/chunker.py](rag/ingest/chunker.py)）：

1. **`structural`（默认推荐）**：根据 Markdown 标题层级切分子块，继承父级标题路径（如 `架构设计 > 索引链路 > 向量存储`），并在分块前缀补充标题上下文；
2. **`fixed`**：固定长度滑动窗口（带 Overlap），作为实验 Baseline；
3. **`recursive`**：按段落、句子层级递进切分的回退策略。

#### 稳定且可复现的分块与索引管理

- **确定性 ID 编码**：分块 ID 格式为 `<doc_id>#<序号>`，完全基于文件与内容拓扑生成，跨机器、跨进程保持绝对可复现；
- **指纹防混淆检查**：`.index/meta.json` 中完整记录语料的 BLAKE2b 哈希指纹、分块超参及 `embedder.name`。当切换不同模型（如从哈希向量切为 BGE）或修改语料内容时，系统会自动感知并触发全量重建，**杜绝因向量空间错乱引发的隐蔽检索 Bug**。

---

### 5.3 严格证据溯源与硬约束拒答

系统在生成层施加了强约束（[rag/generate/answer.py](rag/generate/answer.py)）：
1. **引用格式强制绑定**：Prompt 硬性要求每个论点后必须标注文中来源序号 `[n]`；
2. **出处解析与反向校验**：解析生成的 `[n]` 映射回 `(doc_id, chunk_id, quote, score)`；对于模型捏造的越界引用（如仅提供 3 条背景却引用 `[4]`）实施自动剥离，并在评测中记入幻觉惩罚；
3. **无依据主动拒答**：检索到的相关度低于阈值或无召回结果时，明确触发拒答机制（`refusal`），拒绝强行答复。

---

## 6. 评测指标与消融实验（实测结论）

### 6.1 评测指标体系

实现位于 [rag/eval/metrics.py](rag/eval/metrics.py)，全部为无副作用的纯函数：

- **`Recall@k`**：标注答案所在的真实文档在前 $k$ 个候选中的召回覆盖比例；
- **`Precision@k`**：前 $k$ 个候选片段中命中相关文档的纯度；
- **`MRR`（Mean Reciprocal Rank）**：首个命中相关文档的排名的倒数均值；
- **`nDCG@k`**：基于位置衰减折损的累积增益（针对文档去重后计算，防止多片段重复计分使指标虚高超过 1.0）；
- **`citation_precision`**：答案给出的引用出处准确落在真实相关文档中的比例（衡量“答案写对了但出处瞎编”的幻觉）；
- **`refusal_accuracy`**：该拒答的主动拒答、该回答的正常回答的决策准确率；
- **`P50 / P95 Latency`**：端到端延迟分布（毫秒）。

---

### 6.2 60 题基准消融实验对比表（A~K）

在 48 篇真实技术语料（685 个分块）、60 道人工与模型混合标注题集（涵盖事实题、对比题、多跳题与不可答负样本）下，实测消融对比表如下：

> **实验环境**：BGE 向量模型 + NVIDIA RTX 4070 Laptop GPU + 真实 LLM 生成服务。

| 实验标签 | 架构组合说明 | Recall@1 | Recall@5 | MRR | nDCG@10 | 引用精确率 | 拒答准确率 | P50 (ms) | P95 (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A-baseline** | 单路 BGE 向量召回 | 0.931 | 0.974 | 0.963 | 0.9775 | 0.905 | 0.933 | 1402 | 2377 |
| **B-bm25** | 单路 BM25 关键词召回 | 0.914 | 0.983 | 0.956 | 0.9695 | 0.917 | 0.983 | 1465 | 2259 |
| **C-rrf** | **向量 ∥ BM25 + RRF 融合** | **0.931** | **1.000** | **0.971** | **0.9809** | **0.934** | 0.967 | 1453 | 2281 |
| **D-rewrite** | RRF 融合 + 关键词查询改写 | 0.845 | 1.000 | 0.922 | 0.9433 | 0.928 | 0.967 | 1492 | 2387 |
| **E-rerank** | RRF + 词法覆盖重排 | 0.966 | 1.000 | 0.991 | 0.9901 | 0.894 | 0.967 | 1508 | 3052 |
| **F-multihop** | **RRF + 多跳迭代检索 (2跳)** | **0.983** | 0.991 | **1.000** | **0.9960** | 0.862 | 0.967 | 2938 | 3552 |
| **G-ce** | 多跳 + Cross-Encoder 重排 | 0.957 | 1.000 | 0.983 | 0.9876 | 0.876 | 0.967 | 4329 | 15666 |
| **I-chunking** | 固定窗口分块（消融结构分块） | 0.948 | 0.991 | 0.983 | 0.9859 | 0.931 | 0.950 | 2745 | 3515 |
| **J-vec-rerank**| 单路向量 + 词法重排 | 0.966 | 0.991 | 0.991 | 0.9923 | 0.917 | 0.983 | 1482 | 2756 |
| **K-rrf-ce** | **RRF 融合 + Cross-Encoder 重排** | **0.966** | **1.000** | **0.991** | **0.9923** | 0.885 | **0.983** | 1740 | 2733 |

---

### 6.3 关键实验结论与实证洞察

1. **混合检索 + RRF 融合效果稳固**：
   单路向量（A）与单路 BM25（B）各自均有召回漏洞，而双路经 RRF 融合后（C），**Recall@5 直接达到了 100%**，同时引用精确率提升到 93.4%。双路互补能显著拉高召回率天花板。
2. **多跳迭代（F）在复杂推理题上展现高天花板**：
   在包含多文档依赖的题型中，多跳检索将 MRR 提升至 1.000，nDCG@10 达到 0.9960。但代价是端到端延迟翻倍（P50 从 1.4s 增加至 2.9s），工程落地时需针对查询意图判定是否需要开启多跳。
3. **结构感知分块（C）显著优于固定切分（I）**：
   固定切分破坏了小节与标题的拓扑结构，不仅各项指标出现下滑，而且在长尾上下文理解上容易发生幻觉。
4. **警惕天花板效应与伪优化**：
   在仅有几篇小文档的极端极简语料上，强模型往往会跑出 `nDCG@10 = 1.0000` 的满分；此时重排器反而会因为多余的计算而显得只有负收益。当语料扩展至数百篇以上且难度提升后，Cross-Encoder 与重排技术的排序正向收益才会真实呈现。

---

## 7. 配置与命令行覆盖

系统配置遵循三级加载优先级：
$$\text{内置默认值 (rag/config.py)} \longrightarrow \text{YAML 配置文件 (configs/default.yaml)} \longrightarrow \text{CLI 动态覆盖 (--set)}$$

支持在执行任意命令时使用 `--set key=value` 实现零配置入侵的消融实验：

```bash
# 1. 尝试更细粒度的切块策略
python -m rag.cli ask "什么是分块重叠？" --set chunking.max_chars=256 --set chunking.overlap_chars=48

# 2. 启用 Cross-Encoder 精排并指定重排候选规模
python -m rag.cli ask "BM25 算法的核心公式" --set retrieval.rerank.enabled=true --set retrieval.rerank.provider=cross_encoder

# 3. 动态切换向量后端为 Faiss
python -m rag.cli info --set index.backend=faiss
```

---

## 8. HTTP 服务与单文件前端

系统内置了开箱即用的 Web 服务（[rag/api/](rag/api/)）与单文件可视化前端（[web/index.html](web/index.html)）：

```bash
python -m rag.cli serve --port 8000
```

服务启动后访问 `http://127.0.0.1:8000/`：

- **零前端构建**：无 Node.js、无 Webpack、无三方 CDN 依赖，纯原生 HTML/CSS/JS 编写；
- **全链路可视化**：实时展示向量、BM25 和融合通道的 Top 命中对比、分阶段耗时瀑布图，以及支持点击交互联动的引用卡片高亮；
- **RESTful API**：
  - `GET /health`：当前组件加载情况与运行设备报告；
  - `POST /query`：执行问答与检索；
  - `POST /ingest`：全量语料重新建库。

---

## 9. 编辑器 AI 集成（原生 MCP 服务）

项目内置了遵循标准 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) 规范的 stdio 服务（[rag/mcp/server.py](rag/mcp/server.py)），无需任何三方适配包，原生支持接入 **Cursor** 与 **VSCode Copilot**。

```bash
# 手动以 MCP 协议标准输入输出启动
python -m rag.mcp.server
```

对外暴露的核心 Tool 包括：
- `kb_stats`：获取知识库索引规模与实际生效的硬件/模型状态；
- `kb_search(query, top_k)`：进行纯检索，返回相关原文与分数；
- `kb_answer(question, top_k)`：执行带溯源引用的完整问答；
- `kb_ingest(paths)`：触发知识库重建。

### 客户端接入配置

- **Cursor**：配置位于 [.cursor/mcp.json](.cursor/mcp.json)；
- **VSCode**：配置模板位于 [.vscode/mcp.json](.vscode/mcp.json)。

配置完成后，即可在 AI 对话框中直接调用本地知识库：
> *"@rag-lab 请检索知识库并总结 BM25 中参数 b 的物理含义，附带引用片段。"*

---

## 10. 工程实战踩坑与设计反思

在构建与调优本系统的过程中，总结了一批极具参考价值的底层工程细节与调试经验：

1. **nDCG 计算中的多片段重复计分陷阱**：
   在初始实现中，DCG@k 是按照命中 chunk 直接累加，而标准 IDCG 只按标注文档数量计算。当一个 gold 文档被命中多个 chunk 时，分子被重复累加，**计算出的 nDCG 会超过 1.0（高达 1.63）**。修复方案是在计算增益前必须以文档为粒度去重。
2. **分词与 BM25 的 CJK 优化**：
   中文分词如果不引入重量级词典（如 jieba），单纯的单字切分会导致大量的低信息量高频碰撞。本项目设计了 `拉丁词 + CJK 单字 + CJK 二元组 (Bigram)` 的轻量混合切分策略，二元组天然充当了无词典 n-gram 词表，在零额外依赖下取得了极为稳固的倒排检索效果。
3. **输入形状规整与矩阵维度断言**：
   向量库 `store.add()` 处理来自 `encode_query` 的 1-D 向量时，若直接调用 `len(vectors)` 会将其误判为维度长度而非样本数量。必须前置 `np.asarray` 并规整为 `(N, dim)` 的 2-D 形式后再进行一致性断言。
4. **PowerShell 管道与 UTF-8 编码处理**：
   在 Windows 脚本编写中，PowerShell 5.1 默认按 ANSI 读取无 BOM 的脚本，极易因双字节字符错位导致语法解析失败；同时，单元素管道会退化为标量字符串，导致提取首项时被截断为首字符。脚本中统一采用带 BOM 编码并强制 `@()` 数组包装保证了跨版本稳定性。
5. **静默降级与产物透明度**：
   在复杂的解耦系统中，如果模型加载失败会静默回退，但报告中若不标明实际生效的组件，很容易产出“看着正常但实际跑在错误后端上”的伪数据。因此系统在所有评测产物与 `/health` 中显式记录 `effective.embedder / reranker / generator / device`，保证实验的可复现性与透明性。

---

## 11. 目录结构说明

```
rag-lab/
├── configs/
│   └── default.yaml         # 系统默认配置项（分块、检索、重排、模型与评测）
├── data/
│   ├── corpus/              # 语料库目录（支持 markdown、txt、pdf、docx、csv、json）
│   ├── eval/                # 标注评测数据集（qa.jsonl 基础集 / qa-v2.jsonl 60题基准）
│   └── manifests/           # 语料元数据与来源登记
├── rag/                     # 核心工程包
│   ├── api/                 # Web 服务层（simple_server 原生实现 / main.py FastAPI 实现）
│   ├── eval/                # 评测系统（metrics.py 8项指标计算 / run_eval.py 执行器）
│   ├── generate/            # 生成层（llm.py 适配器 / answer.py 抽取与生成 / prompt.py）
│   ├── index/               # 索引与存储（bm25.py 自写BM25 / embedder.py / store.py 向量库）
│   ├── ingest/              # 语料接入（chunker.py 三种分块策略 / loader.py 多格式解析）
│   ├── mcp/                 # 原生 Model Context Protocol stdio 服务器
│   ├── retrieve/            # 检索与重排（hybrid.py RRF与多跳 / rerank.py / rewrite.py）
│   ├── cli.py               # 命令行统一入口
│   ├── config.py            # 配置管理与解析
│   ├── pipeline.py          # 端到端装配与执行流水线
│   └── text.py              # 文本预处理、分词、停用词与哈希工具
├── reports/                 # 评测输出目录（实验 JSON / Markdown 报告及 ablation.csv）
├── scripts/                 # 实验与工具脚本（消融实验、语料构建、阈值扫描、依赖检测等）
├── tests/                   # 174 个全量测试用例（覆盖分词、BM25、RRF、API、MCP、端到端）
├── web/
│   └── index.html           # 零依赖单文件前端可视化 Demo
├── LICENSE                  # MIT 许可证
├── PLAN.md                  # 架构设计、评测规范与深度技术方案文档
├── pyproject.toml           # 项目安装配置
└── requirements.txt         # 核心运行依赖（仅 numpy）
```

---

## 开源协议

本项目采用 [MIT 许可证](LICENSE) 开源。欢迎交流、测试与贡献！
