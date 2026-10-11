# 评测集候选复核表

- 候选来源：`data/eval/candidates-llm.jsonl`（本表由 generate 子命令生成）
- 候选数量：48

## 筛选规则（勾选前请逐条确认）

1. **不可回答的题**：必须确认语料里确实没有对应内容；只要某篇文档能答，
   就必须改成可答题并补上 `gold_doc_ids`，否则拒答指标会被算反。
2. **多跳题**：必须真的需要两篇以上文档；如果单看一篇就能答，降级成 factual。
3. **gold_doc_ids**：必须与语料里真实存在的文档 id 完全一致（`promote` 会拒绝不存在的 id）。
4. **答案可核对**：`gold_answer` 必须能在语料里找到依据；有疑问的题直接不勾。
5. **重复**：与 `data/eval/qa.jsonl` 已有问题重复的题不要勾（`promote` 会自动跳过）。

## 勾选方式

把要采纳的行首 `- [ ]` 改成 `- [x]`，然后执行：

```powershell
python scripts/gen_eval_candidates.py promote --only-checked --dry-run   # 先预览
python scripts/gen_eval_candidates.py promote --only-checked             # 真正写回
```

| 勾选 | id | type | difficulty | question | gold_doc_ids | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
- [ ] | c001 | factual | easy | 资料中提到的四种主流分块策略分别是什么？ | 02-chunking | 由 02-chunking 的片段生成，原文摘录：分块策略：固定窗口、递归、结构感知与语义分块 分块（chunking）决定检索的最小语义单… |
- [ ] | c002 | comparison | medium | BM25 与稠密向量召回各自的强项和弱点有何不同？ | 03-hybrid-retrieval | 由 03-hybrid-retrieval 的片段生成，原文摘录：混合检索：向量召回、BM25 与 RRF 融合 单一检索通道总有盲区：稠密… |
- [ ] | c003 | multihop | medium | 混合检索把两路或多路候选合并后，希望在评测集上同时达到什么目标？ | 03-hybrid-retrieval | 由 03-hybrid-retrieval 的片段生成，原文摘录：混合检索：向量召回、BM25 与 RRF 融合 单一检索通道总有盲区：稠密… |
- [ ] | c004 | factual | easy | 固定窗口分块的典型配置是什么？它适合哪类文本？ | 02-chunking | 由 02-chunking 的片段生成，原文摘录：分块策略：固定窗口、递归、结构感知与语义分块 分块（chunking）决定检索的最小语义单… |
- [ ] | c005 | comparison | medium | 双编码器（Bi-Encoder）与交叉编码器（Cross-Encoder）在编码方式和复杂度上有何区别？ | 04-rerank-and-eval | 由 04-rerank-and-eval 的片段生成，原文摘录：重排与评测：Recall@k、MRR、nDCG 与引用准确率 "先宽召回、再… |
- [ ] | c006 | multihop | medium | 为什么召回阶段和重排阶段必须用不同指标分别度量？ | 04-rerank-and-eval | 由 04-rerank-and-eval 的片段生成，原文摘录：重排与评测：Recall@k、MRR、nDCG 与引用准确率 "先宽召回、再… |
- [ ] | c007 | comparison | medium | HyDE 与查询分解（Query Decomposition）在解决用户问题与文档表述语义鸿沟上的思路有何不同？ | 05-multihop-and-agentic | 由 05-multihop-and-agentic 的片段生成，原文摘录：多跳与 Agentic RAG：查询改写、证据聚合与自我纠错 单轮… |
- [ ] | c008 | multihop | hard | 多跳 RAG 的迭代检索循环包含哪几个步骤？其中判断信息是否充分后如何构造下一跳查询？ | 05-multihop-and-agentic | 由 05-multihop-and-agentic 的片段生成，原文摘录：多跳与 Agentic RAG：查询改写、证据聚合与自我纠错 单轮… |
- [ ] | c009 | comparison | medium | 截至2026年9月，Claude Haiku 5.5 与 Claude Opus 5.5 分别属于哪个模型层级？ | wiki-Claude-(语言模型) | 由 wiki-Claude-(语言模型) 的片段生成，原文摘录：Claude (语言模型) Claude是人工智能安全与研究公司Anthro… |
- [ ] | c010 | multihop | easy | Claude 这一名称的由来是什么？ | wiki-Claude-(语言模型) | 由 wiki-Claude-(语言模型) 的片段生成，原文摘录：Claude (语言模型) Claude是人工智能安全与研究公司Anthro… |
- [ ] | c011 | comparison | medium | F-score的一般式中，β参数由谁定义，它的取值对精确率和召回率的平衡有什么影响？ | wiki-F-score | 由 wiki-F-score 的片段生成，原文摘录：F-score F值，亦被稱做F-measure，是量測二元分類和資訊檢索演算法的預測精… |
- [ ] | c012 | multihop | medium | 根据资料，当参数β趋近于0和β趋近于无穷大时，F-score分别退化成什么？ | wiki-F-score | 由 wiki-F-score 的片段生成，原文摘录：F-score F值，亦被稱做F-measure，是量測二元分類和資訊檢索演算法的預測精… |
- [ ] | c013 | comparison | easy | K-近邻算法与K-平均算法之间是什么关系？ | wiki-K-近邻算法 | 由 wiki-K-近邻算法 的片段生成，原文摘录：K-近邻算法 在模式识别领域中，最近鄰居法（KNN算法，又譯K-近邻算法）是一种用于分类和… |
- [ ] | c014 | multihop | medium | 在K-NN分类中，当类别分布偏斜时“多数表决”会出现什么缺陷，资料中提到的解决方法之一是什么？ | wiki-K-近邻算法 | 由 wiki-K-近邻算法 的片段生成，原文摘录：K-近邻算法 在模式识别领域中，最近鄰居法（KNN算法，又譯K-近邻算法）是一种用于分类和… |
- [ ] | c015 | comparison | medium | 根据资料，tf-idf 与基于链接分析的评级方法在搜索引擎中分别起什么作用？ | wiki-Tf-idf | 由 wiki-Tf-idf 的片段生成，原文摘录：Tf-idf tf-idf（英語：term frequency–inverse docum… |
- [ ] | c016 | multihop | medium | 根据资料，tf-idf 中字词的重要性与它在文件中出现的次数、以及在语料库中出现的频率分别是什么关系？ | wiki-Tf-idf | 由 wiki-Tf-idf 的片段生成，原文摘录：Tf-idf tf-idf（英語：term frequency–inverse docum… |
- [ ] | c017 | comparison | medium | 与 RNN 相比，transformer 在处理输入数据时有什么不同，这种不同带来了什么好处？ | wiki-Transformer架构 | 由 wiki-Transformer架构 的片段生成，原文摘录：Transformer架构 在深度学习中，transformer（直译为“变… |
- [ ] | c018 | multihop | medium | 资料中提到的沃伦·麦卡洛克和沃尔特·皮茨创造的神经网络计算模型，与人工神经网络中节点和人工神经元的概念之间有什么关联？ | wiki-人工神经网络 | 由 wiki-人工神经网络 的片段生成，原文摘录：人工神经网络 人工神经网络（英語：artificial neural network，AN… |
- [ ] | c019 | comparison | medium | 封閉領域問答與開放領域問答在問題範圍與系統表現提升方式上有何差異？ | wiki-問答系統 | 由 wiki-問答系統 的片段生成，原文摘录：問答系統 問答系統（Question Answering System，QA System），… |
- [ ] | c020 | multihop | medium | Transformer架构的出现催生了哪些模型，这些模型展现出了什么大规模涌现行为？ | wiki-大型语言模型 | 由 wiki-大型语言模型 的片段生成，原文摘录：大型语言模型 大型语言模型（英語：large language model，LLM），也称… |
- [ ] | c021 | comparison | medium | 2017年推出的Transformer架构与循环神经网络相比，在结构上有什么不同，又带来了哪些好处？ | wiki-大型语言模型 | 由 wiki-大型语言模型 的片段生成，原文摘录：大型语言模型 大型语言模型（英語：large language model，LLM），也称… |
- [ ] | c022 | multihop | medium | 在运筹学和控制理论语境下，强化学习被称作什么？在最优控制理论中研究强化学习相关问题时，大部分研究关注的是什么？ | wiki-强化学习 | 由 wiki-强化学习 的片段生成，原文摘录：强化学习 强化学习（英語：Reinforcement learning，簡稱RL）是机器学习中… |
- [ ] | c023 | comparison | medium | 提示词注入与越狱（jailbreaking）有什么区别？ | wiki-提示词注入 | 由 wiki-提示词注入 的片段生成，原文摘录：提示词注入 提示词注入（prompt injection）是一种针对人工智能的网络安全漏洞与… |
- [ ] | c024 | multihop | medium | 提示词注入这一术语由谁创造，他同时指出了提示词注入与越狱的什么区别？ | wiki-提示词注入 | 由 wiki-提示词注入 的片段生成，原文摘录：提示词注入 提示词注入（prompt injection）是一种针对人工智能的网络安全漏洞与… |
- [ ] | c025 | factual | easy | Transformer 是在哪一年、由哪个团队在哪篇论文中提出的？ | wiki-Transformer架构 | 由 wiki-Transformer架构 的片段生成，原文摘录：Transformer架构 在深度学习中，transformer（直译为“变… |
- [ ] | c026 | comparison | medium | 最邻近搜索和KNN（k-nearest neighbor algorithm）之间是什么关系？ | wiki-最邻近搜索 | 由 wiki-最邻近搜索 的片段生成，原文摘录：最邻近搜索 最邻近搜索（Nearest Neighbor Search, NNS）又称为“最… |
- [ ] | c027 | multihop | medium | 当数据未被标记时，为什么不能使用支持向量机进行监督式学习？此时可以采用什么方法，它与支持向量机有什么关系？ | wiki-支持向量机 | 由 wiki-支持向量机 的片段生成，原文摘录：支持向量机 在机器学习中，支持向量机 （台湾称支援向量機，英語：support vector… |
- [ ] | c028 | factual | easy | 人工神经网络在机器学习和认知科学领域被定义为什么样的模型？ | wiki-人工神经网络 | 由 wiki-人工神经网络 的片段生成，原文摘录：人工神经网络 人工神经网络（英語：artificial neural network，AN… |
- [ ] | c029 | comparison | medium | 资料中提到单纯贝氏分类器与支持向量机相比如何？ | wiki-朴素贝叶斯分类器 | 由 wiki-朴素贝叶斯分类器 的片段生成，原文摘录：朴素贝叶斯分类器 朴素贝叶斯分类器（英語：Naive Bayes classifier… |
- [ ] | c030 | multihop | medium | 高德纳在《计算机程序设计艺术》第三章中把最邻近搜索称为什么问题？该问题的具体描述是什么？ | wiki-最邻近搜索 | 由 wiki-最邻近搜索 的片段生成，原文摘录：最邻近搜索 最邻近搜索（Nearest Neighbor Search, NNS）又称为“最… |
- [ ] | c031 | factual | easy | 問答系統與目前主流資訊檢索技術在外部行為上有哪兩點不同？ | wiki-問答系統 | 由 wiki-問答系統 的片段生成，原文摘录：問答系統 問答系統（Question Answering System，QA System），… |
- [ ] | c032 | comparison | medium | 资料中列举了机器学习的多种定义，其中汤姆·米切尔的定义与资料中其他定义在描述机器学习时所侧重的核心要素有何不同？ | wiki-机器学习 | 由 wiki-机器学习 的片段生成，原文摘录：机器学习 机器学习（英語：machine learning，简称ML）是人工智能的一个分支。… |
- [ ] | c033 | multihop | medium | 根据资料，单纯贝氏分类器在文本分类和自动医疗诊断中都有应用，那么它在文本分类中是以什么为特征判断文件所属类别的？ | wiki-朴素贝叶斯分类器 | 由 wiki-朴素贝叶斯分类器 的片段生成，原文摘录：朴素贝叶斯分类器 朴素贝叶斯分类器（英語：Naive Bayes classifier… |
- [ ] | c034 | comparison | medium | 梯度下降法与梯度上升法在迭代搜索方向上有何不同，它们分别接近函数的哪种极值点？ | wiki-梯度下降法 | 由 wiki-梯度下降法 的片段生成，原文摘录：梯度下降法 梯度下降法（英語：Gradient descent）是一种求解无约束最优化问题的… |
- [ ] | c035 | multihop | easy | 资料提到机器学习已广泛应用于多个领域，请列出其中至少五个领域。 | wiki-机器学习 | 由 wiki-机器学习 的片段生成，原文摘录：机器学习 机器学习（英語：machine learning，简称ML）是人工智能的一个分支。… |
- [ ] | c036 | factual | easy | 强化学习是除监督学习和非监督学习之外的第三种基本机器学习方法，它与监督学习的主要区别是什么？ | wiki-强化学习 | 由 wiki-强化学习 的片段生成，原文摘录：强化学习 强化学习（英語：Reinforcement learning，簡稱RL）是机器学习中… |
- [ ] | c037 | comparison | medium | 注意力机制的软权重与通常的权重在运行时有什么不同？ | wiki-注意力机制 | 由 wiki-注意力机制 的片段生成，原文摘录：注意力机制 注意力机制（英語：attention）是人工神经网络中一种模仿认知注意力的技术。… |
- [ ] | c038 | multihop | medium | 在注意力机制的概述中，注意力机制的输出结果是如何由软权重和词嵌入向量计算得到的？ | wiki-注意力机制 | 由 wiki-注意力机制 的片段生成，原文摘录：注意力机制 注意力机制（英語：attention）是人工神经网络中一种模仿认知注意力的技术。… |
- [ ] | c039 | comparison | medium | 根据资料，精确率与召回率分别可视为衡量什么的指标？它们各自越高代表什么？ | wiki-精确率与召回率 | 由 wiki-精确率与召回率 的片段生成，原文摘录：精确率与召回率 在分类任务中，精确率（Precision）和召回率（Recall）是用于… |
- [ ] | c040 | multihop | medium | 资料中提到的生长网（Cresceptron）与深度学习框架的起源有什么关系？ | wiki-深度学习 | 由 wiki-深度学习 的片段生成，原文摘录：深度学习 深度学习（英語：deep learning）是机器学习的分支，是一種以人工神經網路為… |
- [ ] | c041 | comparison | medium | 迁移学习在认知科学领域有哪些应用？ | wiki-迁移学习 | 由 wiki-迁移学习 的片段生成，原文摘录：迁移学习 迁移学习（英語：Transfer learning）是属于机器学习的一种研究领域。… |
- [ ] | c042 | multihop | medium | 知识图谱的概念最早由谁在何时提出，其被定义为什么？ | wiki-知識圖譜 | 由 wiki-知識圖譜 的片段生成，原文摘录：知識圖譜 知识图谱（英語：Knowledge Graph），是结构化的语义知识库，用于以符号形… |
- [ ] | c043 | factual | easy | 支持向量机（SVM）在分类与回归分析中属于哪一类学习模型？ | wiki-支持向量机 | 由 wiki-支持向量机 的片段生成，原文摘录：支持向量机 在机器学习中，支持向量机 （台湾称支援向量機，英語：support vector… |
- [ ] | c044 | comparison | medium | 根据资料，过拟合与欠拟合在模型偏差和方差上的表现有何不同？ | wiki-過適 | 由 wiki-過適 的片段生成，原文摘录：過適 在統計學中，過適（英語：overfitting，或稱擬合過度）是指過於緊密或精確地匹配特定資… |
- [ ] | c045 | multihop | hard | 在猫狗识别示例中，第一类错误率和第二类错误率分别是多少？它们与精确率、召回率的关系在资料中是如何描述的？ | wiki-精确率与召回率 | 由 wiki-精确率与召回率 的片段生成，原文摘录：精确率与召回率 在分类任务中，精确率（Precision）和召回率（Recall）是用于… |
- [ ] | c046 | comparison | medium | 根据资料，LSTM 与 GRU 是什么关系？ | wiki-長短期記憶 | 由 wiki-長短期記憶 的片段生成，原文摘录：長短期記憶 长短期记忆（英語：Long Short-Term Memory，LSTM）是一种… |
- [ ] | c047 | multihop | medium | 根据资料，1966年ALPAC报告发布后机器翻译研究经费遭到大幅削减，而机器翻译研究要到何时才因统计机器翻译系统的发展而得以更上一层楼？ | wiki-自然语言处理 | 由 wiki-自然语言处理 的片段生成，原文摘录：自然语言处理 自然語言處理（英語：Natural Language Processing，… |
- [ ] | c048 | comparison | medium | 随机森林方法结合了哪些已有的想法或方法来建造决策树的集合？ | wiki-随机森林 | 由 wiki-随机森林 的片段生成，原文摘录：随机森林 在機器學習中，隨機森林是一個包含多個決策樹的分類器，並且其輸出的類別是由個別樹輸出的… |
