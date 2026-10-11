# Word2vec

Word2vec是一群用來產生詞向量的相關模型。這些模型為淺层雙層的神經網路，用來訓練以重新建構語言學之詞文本。網路以詞表現，並且需猜測相鄰位置的輸入詞，在word2vec中词袋模型假設下，詞的順序是不重要的。 
訓練完成之後，word2vec模型可以把每個詞映射到一個向量，來表示詞与詞之間的關係。該向量為神經網路的隱藏層。
Word2vec依賴skip-grams或連續詞袋（CBOW）來建立神經詞嵌入。Word2vec為托馬斯·米科洛夫（Tomas Mikolov）在Google帶領的研究團隊創造。該演算法漸漸被其他人所分析和解釋。


## Skip-grams和CBOW
CBOW把一個詞從詞窗剔除。在CBOW下給定n詞圍繞著詞w，word2vec預測一個句子中其中一個缺漏的詞c，即以機率
  
    
      
        p
        (
        c
        
          |
        
        w
        )
      
    
    {\displaystyle p(c|w)}
  
來表示。相反地，Skip-gram給定詞窗中的文本，預測當前的詞
  
    
      
        p
        (
        w
        
          |
        
        c
        )
      
    
    {\displaystyle p(w|c)}
  
。


## 延伸
Word2vec用來建構整份文件（而分獨立的詞）的延伸應用已被提出，
該延伸稱為paragraph2vec或doc2vec，並且用C、Python和 Java/Scala實做成工具（參考下方）。Java和Python也支援推斷文件嵌入於未觀測的文件。


## 分析
對word2vec框架為何做词嵌入如此成功知之甚少，約阿夫·哥德堡（Yoav Goldberg）和歐莫·列維（Omer Levy）指出word2vec的功能導致相似文本擁有相似的嵌入（用余弦相似性計算）並且和約翰·魯伯特·弗斯的分佈假說有關。


## 實作
C （页面存档备份，存于互联网档案馆）
Java/Scala
Python （页面存档备份，存于互联网档案馆）
Python （页面存档备份，存于互联网档案馆）


## 參見
向量空間模型


## 参考文献
