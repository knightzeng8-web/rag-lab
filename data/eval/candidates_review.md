# 评测集候选复核表

- 候选来源：`C:\Users\zeng\Documents\deepseek-harness\default-workspace\rag-lab\data\eval\candidates.jsonl`（本表由 generate 子命令生成）
- 候选数量：48

## 筛选规则（勾选前请逐条确认）

1. **不可回答的题**：必须确认语料里确实没有对应内容；只要某篇文档能答，
   就必须改成可答题并补上 `gold_doc_ids`，否则拒答指标会被算反。
   机械判据只能提供线索：备注里的「实词命中率」越高、越说明语料其实讲过这个主题。
2. **多跳题**：必须真的需要两篇以上文档；如果单看一篇就能答，降级成 factual。
3. **gold_doc_ids**：必须与语料里真实存在的文档 id 完全一致（`promote` 会拒绝不存在的 id）。
4. **答案可核对**：`gold_answer` 必须能在语料里找到依据；有疑问的题直接不勾。
5. **重复**：与 `data/eval/qa.jsonl` 已有问题重复的题不要勾（`promote` 会自动跳过）。
6. **术语类题目**（comparison / multihop 的「X」）：中文分词不支持词典切分，
   偶尔会把长词切成「意力」（来自「注意力」）这类接缝碎片；脚本已经用
   「跨词边界检测 + 标题内碎片判据」过滤掉大部分，但**勾选前请人工把术语改成
   语料里真实存在的说法**，改不动就直接不勾。

## 勾选方式

把要采纳的行首 `- [ ]` 改成 `- [x]`，然后执行：

```powershell
python scripts/gen_eval_candidates.py promote --only-checked --dry-run   # 先预览
python scripts/gen_eval_candidates.py promote --only-checked             # 真正写回
```

| 勾选 | id | type | difficulty | question | gold_doc_ids | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
- [ ] | c001 | factual | medium | 根据语料，「三、Advanced RAG 的三条改造主线」的关键要点是什么？ | 01-rag-overview | 答案在 01-rag-overview 的小节标题「三、Advanced RAG 的三条改造主线」所在小节里，可直接摘录。 |
- [ ] | c002 | comparison | medium | 资料中「wiki-人工神经网络」与「wiki-深度学习」对「网络」的说法有什么差异？ | wiki-人工神经网络, wiki-深度学习 | 「网络」在 wiki-人工神经网络 出现 86 次、wiki-深度学习 出现 85 次（idf=0.78），差异必须对照两篇才能说清。 |
- [ ] | c003 | multihop | hard | 「RAG 是什么：从 Naive RAG 到 Advanced RAG」提到了「稠密」，而「混合检索：向量召回、BM25 与 RRF 融合」解释了「稠密」的做法… | 01-rag-overview, 02-chunking, 03-hybrid-retrieval | 「稠密」的线索分散在 01-rag-overview（1 次）与 03-hybrid-retrieval（2 次），单看任意一篇都答不全。 |
- [ ] | c004 | unanswerable | medium | 《清明上河图》现藏于哪家博物馆的主要流程和注意事项是什么？ | （空） | 主题「《清明上河图》现藏于哪家博物馆」属领域外：关键实词 于哪(出现1次/df=1)、上河(出现0次/df=0)、博物(出现0次/df=0)… |
- [ ] | c005 | comparison | medium | 资料中「wiki-机器学习」与「wiki-深度学习」对「学习」的说法有什么差异？ | wiki-机器学习, wiki-深度学习 | 「学习」在 wiki-机器学习 出现 63 次、wiki-深度学习 出现 83 次（idf=0.51），差异必须对照两篇才能说清。 |
- [ ] | c006 | unanswerable | medium | 《马拉喀什条约》的签署年份与缔约国数量的主要流程和注意事项是什么？ | （空） | 主题「《马拉喀什条约》的签署年份与缔约国数量」属领域外：关键实词 国数(出现1次/df=1)、数量(出现42次/df=14)、与缔(出现0次… |
- [ ] | c007 | factual | medium | 「三、参数怎么选：从评测倒推」在资料中是如何定义的？ | 02-chunking | 答案在 02-chunking 的小节标题「三、参数怎么选：从评测倒推」所在小节里，可直接摘录。 |
- [ ] | c008 | comparison | medium | 资料中「wiki-大型语言模型」与「wiki-深度学习」对「训练」的说法有什么差异？ | wiki-大型语言模型, wiki-深度学习 | 「训练」在 wiki-大型语言模型 出现 56 次、wiki-深度学习 出现 49 次（idf=0.73），差异必须对照两篇才能说清。 |
- [ ] | c009 | unanswerable | medium | 关于「丝绸之路沿线楼兰古城的废弃原因」，资料给出的关键结论是什么？ | （空） | 主题「丝绸之路沿线楼兰古城的废弃原因」属领域外：关键实词 原因(出现11次/df=9)、丝绸(出现0次/df=0)、之路(出现0次/df=0… |
- [ ] | c010 | factual | medium | 「四、工程细节与常见坑」在资料中是如何定义的？ | 02-chunking | 答案在 02-chunking 的小节标题「四、工程细节与常见坑」所在小节里，可直接摘录。 |
- [ ] | c011 | comparison | medium | 资料中「wiki-支持向量机」与「wiki-朴素贝叶斯分类器」对「分类」的说法有什么差异？ | wiki-支持向量机, wiki-朴素贝叶斯分类器 | 「分类」在 wiki-支持向量机 出现 38 次、wiki-朴素贝叶斯分类器 出现 45 次（idf=1.03），差异必须对照两篇才能说清。 |
- [ ] | c012 | unanswerable | easy | 资料里关于「古罗马斗兽场建成时的皇帝是谁」有哪些说明？ | （空） | 主题「古罗马斗兽场建成时的皇帝是谁」属领域外：关键实词 时的(出现4次/df=4)、兽场(出现0次/df=0)、古罗(出现0次/df=0)、… |
- [ ] | c013 | factual | easy | 资料里关于「二、融合策略」是怎么说的？ | 03-hybrid-retrieval | 答案在 03-hybrid-retrieval 的小节标题「二、融合策略」所在小节里，可直接摘录。 |
- [ ] | c014 | comparison | hard | 资料中「wiki-提示工程」与「wiki-提示词注入」对「提示」的说法有什么差异？ | wiki-提示工程, wiki-提示词注入 | 「提示」在 wiki-提示工程 出现 37 次、wiki-提示词注入 出现 38 次（idf=2.02），差异必须对照两篇才能说清。 |
- [ ] | c015 | unanswerable | medium | 资料里关于「科幻电影《银翼杀手》的导演与上映年份」有哪些说明？ | （空） | 主题「科幻电影《银翼杀手》的导演与上映年份」属领域外：关键实词 与上(出现3次/df=3)、上映(出现0次/df=0)、导演(出现0次/df… |
- [ ] | c016 | factual | medium | 「3. 学习式融合（LTR）」在资料中是如何定义的？ | 03-hybrid-retrieval | 答案在 03-hybrid-retrieval 的小节标题「3. 学习式融合（LTR）」所在小节里，可直接摘录。 |
- [ ] | c017 | comparison | hard | 资料中「wiki-Transformer架构」与「wiki-注意力机制」对「意力」的说法有什么差异？ | wiki-Transformer架构, wiki-注意力机制 | 「意力」在 wiki-Transformer架构 出现 58 次、wiki-注意力机制 出现 33 次（idf=2.… |
- [ ] | c018 | unanswerable | medium | 请问「敦煌莫高窟第 220 窟壁画的创作年代」在语料中是怎么描述的？ | （空） | 主题「敦煌莫高窟第 220 窟壁画的创作年代」属领域外：关键实词 画的(出现1次/df=1)、的创(出现1次/df=1)、年代(出现28次/… |
- [ ] | c019 | factual | medium | 资料里关于「三、端到端指标：生成层怎么评」是怎么说的？ | 04-rerank-and-eval | 答案在 04-rerank-and-eval 的小节标题「三、端到端指标：生成层怎么评」所在小节里，可直接摘录。 |
- [ ] | c020 | unanswerable | medium | 请问「哥伦布首次抵达美洲的年份与登陆地点」在语料中是怎么描述的？ | （空） | 主题「哥伦布首次抵达美洲的年份与登陆地点」属领域外：关键实词 哥伦(出现3次/df=1)、首次(出现9次/df=8)、与登(出现0次/df=… |
- [ ] | c021 | factual | medium | 「二、检索指标：定义与用法」在资料中是如何定义的？ | 04-rerank-and-eval | 答案在 04-rerank-and-eval 的小节标题「二、检索指标：定义与用法」所在小节里，可直接摘录。 |
- [ ] | c022 | unanswerable | medium | 请问「贝多芬第九交响曲首演时的指挥与地点」在语料中是怎么描述的？ | （空） | 主题「贝多芬第九交响曲首演时的指挥与地点」属领域外：关键实词 时的(出现4次/df=4)、的指(出现19次/df=9)、与地(出现0次/df… |
- [ ] | c023 | factual | medium | 资料里关于「五、何时不要上多跳」是怎么说的？ | 05-multihop-and-agentic | 答案在 05-multihop-and-agentic 的小节标题「五、何时不要上多跳」所在小节里，可直接摘录。 |
- [ ] | c024 | factual | medium | 资料里关于「二、迭代检索：多跳的骨架」是怎么说的？ | 05-multihop-and-agentic | 答案在 05-multihop-and-agentic 的小节标题「二、迭代检索：多跳的骨架」所在小节里，可直接摘录。 |
- [ ] | c025 | factual | easy | 「性能及分析」在资料中是如何定义的？ | wiki-BERT | 答案在 wiki-BERT 的小节标题「性能及分析」所在小节里，可直接摘录。 |
- [ ] | c026 | factual | easy | 「获奖情况」在资料中是如何定义的？ | wiki-BERT | 答案在 wiki-BERT 的小节标题「获奖情况」所在小节里，可直接摘录。 |
- [ ] | c027 | factual | medium | 资料里关于「Artifacts」是怎么说的？ | wiki-Claude-(语言模型) | 答案在 wiki-Claude-(语言模型) 的小节标题「Artifacts」所在小节里，可直接摘录。 |
- [ ] | c028 | factual | medium | 资料里关于「Fable 5与Mythos 5出口管制」是怎么说的？ | wiki-Claude-(语言模型) | 答案在 wiki-Claude-(语言模型) 的小节标题「Fable 5与Mythos 5出口管制」所在小节里，可直接摘录。 |
- [ ] | c029 | factual | medium | 资料里关于「F1-score」是怎么说的？ | wiki-F-score | 答案在 wiki-F-score 的小节标题「F1-score」所在小节里，可直接摘录。 |
- [ ] | c030 | factual | medium | 「F-score」在资料中是如何定义的？ | wiki-F-score | 答案在 wiki-F-score 的小节标题「F-score」所在小节里，可直接摘录。 |
- [ ] | c031 | factual | easy | 资料里关于「决策边界」是怎么说的？ | wiki-K-近邻算法 | 答案在 wiki-K-近邻算法 的小节标题「决策边界」所在小节里，可直接摘录。 |
- [ ] | c032 | factual | easy | 「注释」在资料中是如何定义的？ | wiki-K-近邻算法 | 答案在 wiki-K-近邻算法 的小节标题「注释」所在小节里，可直接摘录。 |
- [ ] | c033 | factual | medium | 根据语料，「tf-idf的理论依据及不足」的关键要点是什么？ | wiki-Tf-idf | 答案在 wiki-Tf-idf 的小节标题「tf-idf的理论依据及不足」所在小节里，可直接摘录。 |
- [ ] | c034 | factual | easy | 资料里关于「Tf-idf」是怎么说的？ | wiki-Tf-idf | 答案在 wiki-Tf-idf 的小节标题「Tf-idf」所在小节里，可直接摘录。 |
- [ ] | c035 | factual | easy | 根据语料，「混合架构」的关键要点是什么？ | wiki-Transformer架构 | 答案在 wiki-Transformer架构 的小节标题「混合架构」所在小节里，可直接摘录。 |
- [ ] | c036 | factual | medium | 「Skip-grams和CBOW」在资料中是如何定义的？ | wiki-Word2vec | 答案在 wiki-Word2vec 的小节标题「Skip-grams和CBOW」所在小节里，可直接摘录。 |
- [ ] | c037 | factual | easy | 「實作」在资料中是如何定义的？ | wiki-Word2vec | 答案在 wiki-Word2vec 的小节标题「實作」所在小节里，可直接摘录。 |
- [ ] | c038 | factual | easy | 资料里关于「理论性质」是怎么说的？ | wiki-人工神经网络 | 答案在 wiki-人工神经网络 的小节标题「理论性质」所在小节里，可直接摘录。 |
- [ ] | c039 | factual | easy | 资料里关于「模板分類」是怎么说的？ | wiki-信息抽取 | 答案在 wiki-信息抽取 的小节标题「模板分類」所在小节里，可直接摘录。 |
- [ ] | c040 | factual | easy | 资料里关于「判斷文句」是怎么说的？ | wiki-信息抽取 | 答案在 wiki-信息抽取 的小节标题「判斷文句」所在小节里，可直接摘录。 |
- [ ] | c041 | factual | easy | 根据语料，「倒排索引」的关键要点是什么？ | wiki-倒排索引 | 答案在 wiki-倒排索引 的小节标题「倒排索引」所在小节里，可直接摘录。 |
- [ ] | c042 | factual | easy | 根据语料，「相关」的关键要点是什么？ | wiki-倒排索引 | 答案在 wiki-倒排索引 的小节标题「相关」所在小节里，可直接摘录。 |
- [ ] | c043 | factual | easy | 「主流实现」在资料中是如何定义的？ | wiki-光学字符识别 | 答案在 wiki-光学字符识别 的小节标题「主流实现」所在小节里，可直接摘录。 |
- [ ] | c044 | factual | easy | 根据语料，「前期处理」的关键要点是什么？ | wiki-光学字符识别 | 答案在 wiki-光学字符识别 的小节标题「前期处理」所在小节里，可直接摘录。 |
- [ ] | c045 | factual | easy | 资料里关于「向量数据库」是怎么说的？ | wiki-向量数据库 | 答案在 wiki-向量数据库 的小节标题「向量数据库」所在小节里，可直接摘录。 |
- [ ] | c046 | factual | easy | 根据语料，「万不」的关键要点是什么？ | wiki-向量数据库 | 答案在 wiki-向量数据库 的高 IDF 术语「万不」(idf=3.49)所在小节里，可直接摘录。 |
- [ ] | c047 | factual | easy | 根据语料，「命名实体识别」的关键要点是什么？ | wiki-命名实体识别 | 答案在 wiki-命名实体识别 的小节标题「命名实体识别」所在小节里，可直接摘录。 |
- [ ] | c048 | factual | easy | 资料里关于「一一」是怎么说的？ | wiki-命名实体识别 | 答案在 wiki-命名实体识别 的高 IDF 术语「一一」(idf=3.49)所在小节里，可直接摘录。 |
