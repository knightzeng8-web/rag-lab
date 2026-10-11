# 随机森林

在機器學習中，隨機森林是一個包含多個決策樹的分類器，並且其輸出的類別是由個別樹輸出的類別的眾數而定。
這個術語是1995年由貝爾實驗室的何天琴所提出的隨機決策森林（random decision forests）而來的。
然后Leo Breiman和Adele Cutler發展出推論出隨機森林的演算法。而"Random Forests"是他們的商標。
這個方法則是結合Breimans的「Bootstrap aggregating」想法和Ho的「random subspace method」以建造決策樹的集合。


## 历史
随机森林的引入最初是由华裔美国人何天琴于1995年先提出的。然后随机森林由Leo Breiman于2001年在一篇论文中提出的。这篇文章描述了一种结合随机节点优化和bagging，利用类CART过程构建不相关树的森林的方法。此外，本文还结合了一些已知的、新颖的、构成了现代随机森林实践的基础成分，特别是

使用out-of-bag误差来代替泛化误差
通过排列度量变量的重要性


## 算法
### 预备：决策树学习
决策树是机器学習的常用方法。 Hastie等说：“树学习是如今最能满足于数据挖掘的方法，因为它在特征值的缩放和其他各种转换下保持不变，对无关特征是穩健的，而且能生成可被檢查的模型。然而，它通常並不準確。”
特别的，生长很深的树容易学习到高度不规则的模式，即过学习，在训练集上具有低偏差和高變異數的特点。随机森林是平均多个深决策树以降低變異數的一种方法，其中，决策树是在一个数据集上的不同部分进行训练的。这是以偏差的小幅增加和一些可解释性的丧失为代價的，但是在最终的模型中通常会大大提高性能。


### Bagging
随机森林训练算法把bagging的一般技术应用到树学习中。给定训练集
  
    
      
        X
        =
        
          x
          
            1
          
        
        ,
        …
        ,
        
          x
          
            n
          
        
      
    
    {\displaystyle X=x_{1},\dots ,x_{n}}
  
和目标
  
    
      
        Y
        =
        
          y
          
            1
          
        
        ,
        …
        ,
        
          y
          
            n
          
        
      
    
    {\displaystyle Y=y_{1},\dots ,y_{n}}
  
，bagging方法重复（
  
    
      
        B
      
    
    {\displaystyle B}
  
次）从训练集中有放回地采样，然后在这些样本上训练树模型：

For b = 1, ..., B:
從X和Y抽取n個訓練樣本（有放回抽樣），分別記為Xb和Yb。
在Xb和Yb上訓練一個分類或回歸樹fb。
在训练结束之后，对未知样本x的预测可以通过对x上所有单个回归树的预测求平均来实现：

  
    
      
        
          
            
              f
              ^
            
          
        
        =
        
          
            1
            B
          
        
        
          ∑
          
            b
            =
            1
          
          
            B
          
        
        
          f
          
            b
          
        
        (
        
          x
          ′
        
        )
      
    
    {\displaystyle {\hat {f}}={\frac {1}{B}}\sum _{b=1}^{B}f_{b}(x')}
  

或者在分类任务中选择多数投票的类别。
这种bagging方法在不增加偏置的情况下降低了方差，从而带来了更好的性能。这意味着，即使单个树模型的预测对训练集的噪声非常敏感，但对于多个树模型，只要这些树并不相关，这种情况就不会出现。简单地在同一个数据集上训练多个树模型会产生强相关的树模型（甚至是完全相同的树模型）。Bootstrap抽样是一种通过产生不同训练集从而降低树模型之间关联性的方法。
此外，x'上所有单个回归树的预测的标准差可以作为预测的不确定性的估计：

  
    
      
        σ
        =
        
          
            
              
                
                  ∑
                  
                    b
                    =
                    1
                  
                  
                    B
                  
                
                (
                
                  f
                  
                    b
                  
                
                (
                
                  x
                  ′
                
                )
                −
                
                  
                    
                      f
                      ^
                    
                  
                
                
                  )
                  
                    2
                  
                
              
              
                B
                −
                1
              
            
          
        
        .
      
    
    {\displaystyle \sigma ={\sqrt {\frac {\sum _{b=1}^{B}(f_{b}(x')-{\hat {f}})^{2}}{B-1}}}.}
  

样本或者树的数量B是一个自由参数。通常使用几百到几千棵树，这取决于训练集的大小和性质。使用交叉验证，或者透過观察out-of-bag误差（那些不包含xᵢ的抽样集合在样本xᵢ的平均预测误差），可以找到最优的B值。当一些树训练到一定程度之后，训练集和测试集的误差开始趋于平稳。


### 从 bagging 到随机森林
上面的过程描述了树的原始的 bagging 算法。随机森林与这个通用的方案只有一点不同:它使用一种改进的学习算法，在学习过程中的每次候选分裂中选择特征的随机子集。这个过程有时又被称为“特征 bagging”。这样做的原因是 bootstrap 抽样导致的树的相关性：如果有一些特征预测目标值的能力很强，那么这些特征就会被许多树所选择，这样就会导致树的强相关性。何天琴分析了不同条件下 bagging 和随机子空间投影对精度提高的影响。
典型地，对于一个包含 p 个特征的分类问题，可以在每次划分时使用 
  
    
      
        
          
            p
          
        
      
    
    {\displaystyle {\sqrt {p}}}
  
 个特征。对于回归问题，作者推荐 p/3 但不少于 5 个特征。


### 极限树
再加上一个随机化步骤，就会得到极限随机树（extremely randomized trees），即极限树。与普通的随机森林相同，他们都是单个树的集成，但也有不同：首先，每棵树都使用整个学习样本进行了训练，其次，自上而下的划分是随机的。它并不计算每个特征的最优划分点（例如，基于信息熵或者基尼不纯度），而是随机选择划分点。该值是从特征经验范围内均匀随机选取的。在所有随机的划分点中，选择其中分数最高的作为结点的划分点。与普通的随机森林相似，可以指定每个节点要选择的特征的个数。该参数的默认值，对于分类问题，是
  
    
      
        
          
            n
          
        
      
    
    {\displaystyle {\sqrt {n}}}
  
，对于回归问题，是
  
    
      
        n
      
    
    {\displaystyle n}
  
，其中 
  
    
      
        n
      
    
    {\displaystyle n}
  
是模型的特征个数。


## 性质
### 特征的重要性
随机森林天然可用来对回归或分类问题中变量的重要性进行排序。下面的技术来自Breiman的论文，R语言包randomForest包含它的实现。
度量数据集 
  
    
      
        
          
            
              D
            
          
          
            n
          
        
        =
        {
        (
        
          X
          
            i
          
        
        ,
        
          Y
          
            i
          
        
        )
        
          }
          
            i
            =
            1
          
          
            n
          
        
      
    
    {\displaystyle {\mathcal {D}}_{n}=\{(X_{i},Y_{i})\}_{i=1}^{n}}
  
的特征重要性的第一步是，使用训练集训练一个随机森林模型。在训练过程中记录下每个数据点的out-of-bag误差，然后在整个森林上进行平均。
为了度量第
  
    
      
        j
      
    
    {\displaystyle j}
  
个特征的重要性，第
  
    
      
        j
      
    
    {\displaystyle j}
  
个特征的值在训练数据中被打乱，并重新计算打乱后的数据的out-of-bag误差。则第
  
    
      
        j
      
    
    {\displaystyle j}
  
个特征的重要性分数可以通过计算打乱前后的out-of-bag误差的差值的平均来得到，这个分数通过计算这些差值的标准差进行标准化。
产生更大分数的特征比小分数的特征更重要。这种特征重要性的度量方法的统计定义由Zhu et al.给出。
这种度量方法也有一些缺陷。对于包含不同取值个数的类别特征，随机森林更偏向于那些取值个数较多的特征，部分置換、growing unbiased trees可以用来解决这个问题。如果数据包含一些相互关联的特征组，那么更小的组更容易被选择。


### 与最近邻算法的关系
Lin和Jeon在2002年指出了随机森林算法和K-近邻算法(k-NN)的关系。 事实证明，这两种算法都可以被看作是所谓的“加权邻居的方案”。这些在数据集
  
    
      
        {
        (
        
          x
          
            i
          
        
        ,
        
          y
          
            i
          
        
        )
        
          }
          
            i
            =
            1
          
          
            n
          
        
      
    
    {\displaystyle \{(x_{i},y_{i})\}_{i=1}^{n}}
  
上训练的模型通过查看一个点的邻居来计算一个新点x'的预测值
  
    
      
        
          
            
              y
              ^
            
          
        
      
    
    {\displaystyle {\hat {y}}}
  
，并且使用权重函数W对这些邻居进行加权:

  
    
      
        
          
            
              y
              ^
            
          
        
        =
        
          ∑
          
            i
            =
            1
          
          
            n
          
        
        W
        (
        
          x
          
            i
          
        
        ,
        
          x
          ′
        
        )
        
        
          y
          
            i
          
        
        .
      
    
    {\displaystyle {\hat {y}}=\sum _{i=1}^{n}W(x_{i},x')\,y_{i}.}
  

其中， 
  
    
      
        W
        (
        
          x
          
            i
          
        
        ,
        
          x
          ′
        
        )
      
    
    {\displaystyle W(x_{i},x')}
  
是第i个点在同一棵树中相对于新的数据点x'的非负权重。对于任一特定的点x'，
  
    
      
        
          x
          
            i
          
        
      
    
    {\displaystyle x_{i}}
  
的权重的和必须为1。权重函数设定如下：

对于k-NN算法，如果xi是距离x'最近的k个点之一，则
  
    
      
        W
        (
        
          x
          
            i
          
        
        ,
        
          x
          ′
        
        )
        =
        
          
            1
            k
          
        
      
    
    {\displaystyle W(x_{i},x')={\frac {1}{k}}}
  
，否则为0。
对于树，如果xi与x'属于同一个包含k'个点的叶结点，则
  
    
      
        W
        (
        
          x
          
            i
          
        
        ,
        
          x
          ′
        
        )
        =
        
          
            1
            
              k
              ′
            
          
        
      
    
    {\displaystyle W(x_{i},x')={\frac {1}{k'}}}
  
，否则为0。
因为森林平均了m棵树的预测，且这些树具有独立的权重函数
  
    
      
        
          W
          
            j
          
        
      
    
    {\displaystyle W_{j}}
  
，故森林的预测值是：

  
    
      
        
          
            
              y
              ^
            
          
        
        =
        
          
            1
            m
          
        
        
          ∑
          
            j
            =
            1
          
          
            m
          
        
        
          ∑
          
            i
            =
            1
          
          
            n
          
        
        
          W
          
            j
          
        
        (
        
          x
          
            i
          
        
        ,
        
          x
          ′
        
        )
        
        
          y
          
            i
          
        
        =
        
          ∑
          
            i
            =
            1
          
          
            n
          
        
        
          (
          
            
              
                1
                m
              
            
            
              ∑
              
                j
                =
                1
              
              
                m
              
            
            
              W
              
                j
              
            
            (
            
              x
              
                i
              
            
            ,
            
              x
              ′
            
            )
          
          )
        
        
        
          y
          
            i
          
        
        .
      
    
    {\displaystyle {\hat {y}}={\frac {1}{m}}\sum _{j=1}^{m}\sum _{i=1}^{n}W_{j}(x_{i},x')\,y_{i}=\sum _{i=1}^{n}\left({\frac {1}{m}}\sum _{j=1}^{m}W_{j}(x_{i},x')\right)\,y_{i}.}
  

上式表明了整个森林也采用了加权的邻居方案，其中的权重是各个树的平均。在这里，x'的邻居是那些在任一树中属于同一个叶节点的点
  
    
      
        
          x
          
            i
          
        
      
    
    {\displaystyle x_{i}}
  
。只要
  
    
      
        
          x
          
            i
          
        
      
    
    {\displaystyle x_{i}}
  
与x'在某棵树中属于同一个叶节点，
  
    
      
        
          x
          
            i
          
        
      
    
    {\displaystyle x_{i}}
  
就是x'的邻居。


## 基于随机森林的非监督学习
作为构建的一部分，随机森林预测器自然会导致观测值之间的不相似性度量。还可以定义未标记数据之间的随机森林差异度量：其思想是构造一个随机森林预测器，将“观测”数据与适当生成的合成数据区分开来。 观察到的数据是原始的未标记数据，合成数据是从参考分布中提取的。随机森林的不相似性度量之所以吸引人，是因为它能很好地处理混合变量类型，对输入变量的单调变换是不敏感的，而且在存在异常值的情况下度量结果依然可靠。由于其固有变量的选择，随机森林不相似性很容易处理大量的半连续变量。


## 學習演算法
根據下列演算法而建造每棵樹：

用
  
    
      
        N
      
    
    {\displaystyle N}
  
來表示訓練用例（样本）的個數，
  
    
      
        M
      
    
    {\displaystyle M}
  
表示特征数目。
输入特征数目
  
    
      
        m
      
    
    {\displaystyle m}
  
，用于确定决策树上一个节点的决策结果；其中
  
    
      
        m
      
    
    {\displaystyle m}
  
應远小於
  
    
      
        M
      
    
    {\displaystyle M}
  
。
從
  
    
      
        N
      
    
    {\displaystyle N}
  
個訓練用例（样本）中以有放回抽样的方式，取樣
  
    
      
        N
      
    
    {\displaystyle N}
  
次，形成一个训练集（即bootstrap取樣），並用未抽到的用例（样本）作預測，評估其誤差。
對於每一個節點，隨機選擇
  
    
      
        m
      
    
    {\displaystyle m}
  
個特征，决策树上每个节点的决定都是基于这些特征确定的。根據這
  
    
      
        m
      
    
    {\displaystyle m}
  
個特征，計算其最佳的分裂方式。
每棵樹都會完整成長而不會剪枝（Pruning，這有可能在建完一棵正常樹狀分類器後會被採用）。


## 優點
隨機森林的優點有：

對於很多種資料，它可以產生高準確度的分類器。
它可以處理大量的輸入變數。
它可以在決定類別時，評估變數的重要性。
在建造森林時，它可以在內部對於一般化後的誤差產生不偏差的估計。
它包含一個好方法可以估計遺失的資料，並且，如果有很大一部分的資料遺失，仍可以維持準確度。
它提供一個實驗方法，可以去偵測variable interactions。
對於不平衡的分類資料集來說，它可以平衡誤差。
它計算各例中的親近度，對於数据挖掘、偵測離群點（outlier）和將資料視覺化非常有用。
使用上述。它可被延伸應用在未標記的資料上，這類資料通常是使用非監督式聚類。也可偵測偏離者和觀看資料。
學習過程是很快速的。


## 开源实现
The Original RF （页面存档备份，存于互联网档案馆） by Breiman and Cutler written in Fortran 77.
ALGLIB （页面存档备份，存于互联网档案馆） contains a modification of the random forest in C#, C++, Pascal, VBA.
party （页面存档备份，存于互联网档案馆） Implementation based on the conditional inference trees in R.
randomForest （页面存档备份，存于互联网档案馆） for classification and regression in R.
Python implementation （页面存档备份，存于互联网档案馆） with examples in scikit-learn.
Orange data mining suite includes random forest learner and can visualize the trained forest.
Matlab （页面存档备份，存于互联网档案馆） implementation.
SQP （页面存档备份，存于互联网档案馆） software uses random forest algorithm to predict the quality of survey questions, depending on formal and linguistic characteristics of the question.
Weka RandomForest （页面存档备份，存于互联网档案馆） in Java library and GUI.
ranger （页面存档备份，存于互联网档案馆） A C++ implementation of random forest for classification, regression, probability and survival. Includes interface for R.


## 参阅
机器学习
提升方法 - 机器学习方法
决策树学习  - 机器学习算法
集成学习——通过结合多个学习器来解决问题的一种机器学习范式
無母數統計
随机化算法 - 将一定程度的随机性作为其逻辑或程序的一部分的算法
梯度提升技術


## 参考文献
## 外部連結
（英文）Ho, Tin Kam (1995). "Random Decision Forest". Proc. of the 3rd Int'l Conf. on Document Analysis and Recognition,  Montreal, Canada, August 14-18, 1995, 278-282（Preceding Work）
（英文）Ho, Tin Kam (1998). "The Random Subspace Method for Constructing Decision Forests".  IEEE Trans. on Pattern Analysis and Machine Intelligence 20 (8), 832-844（Preceding Work）
（英文）Deng, H; Runger, G; Tuv, Eugene (2011). Bias of importance measures for multi-valued attributes and solutions, Proceedings of the 21st International Conference on Artificial Neural Networks (ICANN2011)
（英文）Amit, Yali and Geman, Donald (1997) "Shape quantization and recognition with randomized trees". Neural Computation 9, 1545-1588. （页面存档备份，存于互联网档案馆）（Preceding work）
（英文）Breiman, Leo "Looking Inside The Black Box". Wald Lecture II（Lecture）
（英文）Breiman, Leo (2001).  "Random Forests".  Machine Learning 45 (1), 5-32（Original Article）
（英文）Random Forest classifier description（Site of Leo Breiman）
（英文）Liaw, Andy & Wiener, Matthew "Classification and Regression by randomForest" R News (2002) Vol. 2/3 p. 18 （页面存档备份，存于互联网档案馆）（Discussion of the use of the random forest package for R）
（英文）Ho, Tin Kam (2002). "A Data Complexity Analysis of Comparative Advantages of Decision Forest Constructors". Pattern Analysis and Applications 5, p. 102-112 （页面存档备份，存于互联网档案馆）（Comparison of bagging and random subspace method）
