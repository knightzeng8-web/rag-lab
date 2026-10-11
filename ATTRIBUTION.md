# 语料来源与署名（ATTRIBUTION）

本仓库的**代码**可自由使用（许可证见仓库根目录；未附 LICENSE 时默认保留所有权利）。
但 `data/corpus/` 下的**语料内容**有各自的来源与许可，**必须按下列方式署名**。

## 1. 手写示例语料（5 篇）

| doc_id | 文件 | 来源 | 许可 |
| --- | --- | --- | --- |
| `01-rag-overview` … `05-multihop-and-agentic` | `data/corpus/*.md` | 本项目作者自撰 | 随本仓库代码一同授权 |

## 2. 中文维基百科语料（43 篇，`data/corpus/wiki-*.md`）

- **来源**：中文维基百科（https://zh.wikipedia.org）
- **许可**：**CC BY-SA 4.0**（https://creativecommons.org/licenses/by-sa/4.0/deed.zh）
- **抓取方式**：`python scripts/fetch_public_corpus.py wikipedia --titles …`，走 MediaWiki API
- **逐篇的原始链接、条目名与 revision 号**记录在 [`data/manifests/corpus.json`](data/manifests/corpus.json)
  的 `source` / `title` 字段里，可用它逐条回溯到原始页面版本。

> **CC BY-SA 4.0 的两条硬性要求**（转载或再分发时必须遵守）：
> 1. **署名**：注明来源（中文维基百科）、许可（CC BY-SA 4.0），并保留指向原文的链接；
> 2. **相同方式共享**：如果你修改了这些文本并再分发，**衍生作品必须同样以 CC BY-SA 4.0 发布**
>    （即：不能把这些语料并进一个仅闭源的语料库里再分发）。
>
> 只在本地做检索/评测实验、不对外分发时，不受这两条约束；一旦公开仓库或分发，就必须遵守。
> 若不想承担这个义务，把 `data/corpus/wiki-*.md` 换成你自己的语料即可（见下）。

## 3. 换成你自己的语料（推荐）

```bash
python scripts/build_corpus.py "D:\\我的资料" --out data/corpus --split-level 2 --prefix mydocs
python scripts/gen_eval_candidates.py generate --provider llm --model deepseek-chat \
       --api-base https://api.deepseek.com/v1 --api-key-env DEEPSEEK_API_KEY --max-docs 30
```

`scripts/build_corpus.py` 会把每篇文档的来源与内容哈希写进 `data/manifests/corpus.json`，
便于回溯；换成自有语料后，本文件第 2 节整段都可以删掉。

## 4. 模型权重

本仓库**不包含**任何模型权重。`BAAI/bge-small-zh-v1.5` 与 `BAAI/bge-reranker-base` 由
`sentence-transformers` 在首次运行时自行下载（默认缓存在 `.hf-cache/`，已被 `.gitignore` 忽略），
其许可分别见各自的模型卡（均为 MIT/Apache-2.0 类宽松许可）。
