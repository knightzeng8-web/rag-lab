# BERT

基于变换器的双向编码器表示技术（英語：Bidirectional Encoder Representations from Transformers，BERT）是用于自然语言处理（NLP）的预训练技术，由Google提出。2018年，雅各布·德夫林和同事创建并发布了BERT。Google正在利用BERT来更好地理解用户搜索语句的语义。2020年的一项文献调查得出结论：「在一年多一点的时间里，BERT已经成为NLP实验中无处不在的基线」，算上分析和改进模型的研究出版物超过150篇。
最初的英语BERT发布时提供两种类型的预训练模型：（1）BERTBASE模型，一个12层，768维，12个自注意头（self attention head），110M参数的神经网络结构；（2）BERTLARGE模型，一个24层，1024维，16个自注意头，340M参数的神经网络结构。两者的训练语料都是BookCorpus以及英語維基百科语料，单词量分别是8億以及25億。


## 结构
BERT的核心部分是一个Transformer模型，其中编码层数和自注意力头数量可变。结构与Vaswani等人(2017)的实现几乎“完全一致”。
BERT在两个任务上进行预训练： 语言模型（15%的token被掩盖，BERT需要从上下文中进行推断）和下一句预测（BERT需要预测给定的第二个句子是否是第一句的下一句）。训练完成后，BERT学习到单词的上下文嵌入。代价昂贵的预训练完成后，BERT可以使用较少的资源和较小的数据集在下游任务上进行微调，以改进在这些任务上的性能。


## 性能及分析
BERT在以下自然语言理解任务上的性能表现得最为卓越：

GLUE（General Language Understanding Evaluation，通用语言理解评估）任务集（包括9个任务）。
SQuAD（Stanford Question Answering Dataset，斯坦福问答数据集）v1.1和v2.0。
SWAG（Situations With Adversarial Generation，对抗生成的情境）。
有關BERT在上述自然语言理解任务中為何可以達到先进水平，目前還未找到明確的原因。目前BERT的可解释性研究主要集中在研究精心选择的输入序列对BERT的输出的影响关系，通过探测分类器分析内部向量表示，以及注意力权重表示的关系。


## 历史
BERT起源于预训练的上下文表示学习，包括半监督序列学习（Semi-supervised Sequence Learning），生成预训练（Generative Pre-Training），ELMo和ULMFit。与之前的模型不同，BERT是一种深度双向的、无监督的语言表示，且仅使用纯文本语料库进行预训练的模型。上下文无关模型（如word2vec或GloVe）为词汇表中的每个单词生成一个词向量表示，因此容易出现单词的歧义问题。BERT考虑到单词出现时的上下文。例如，词“水分”的word2vec词向量在“植物需要吸收水分”和“财务报表裡有水分”是相同的，但BERT根据上下文的不同提供不同的词向量，词向量与句子表达的句意有关。
2019年10月25日，Google搜索宣布他们已经开始在美国国内的英语搜索查询中应用BERT模型。2019年12月9日，据报道，Google搜索已经在70多种语言的搜索采用了BERT。2020年10月，几乎每一个基于英语的查询都由BERT处理。


## 获奖情况
在2019年计算语言学协会北美分会（NAACL）年会上，BERT获得了最佳长篇论文奖。


## 参见
## 参考文献
## 外部链接
官方GitHub仓库 （页面存档备份，存于互联网档案馆）
