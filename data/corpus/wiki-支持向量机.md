# 支持向量机

在机器学习中，支持向量机
（台湾称支援向量機，英語：support vector machine，常简称為SVM，又名支援向量网络）是在分类与迴歸分析中分析数据的監督式學習模型与相关的学习算法。给定一组训练实例，每个训练实例被标记为属于两个类别中的一个或另一个，SVM训练算法建立一个将新的实例分配给两个类别之一的模型，使其成为非概率二元线性分类器。SVM模型是将实例表示为空间中的点，这样映射就使得单独类别的实例被尽可能宽的明显的间隔分开。然后，将新的实例映射到同一空间，并基于它们落在间隔的哪一侧来预测所属类别。
除了进行线性分类之外，SVM还可以使用所谓的核技巧有效地进行非线性分类，将其输入隐式映射到高维特征空间中。
当数据未被标记时，不能进行监督式学习，需要用非監督式學習，它会尝试找出数据到簇的自然聚类，并将新数据映射到这些已形成的簇。将支援向量机改进的聚类算法被称为支援向量聚类，当数据未被标记或者仅一些数据被标记时，支援向量聚类经常在工业应用中用作分类步骤的预处理。


## 动机
将数据进行分类是机器学习中的一项常见任务。
假设某些给定的数据点各自属于两个类之一，而目标是确定新数据点将在哪个类中。对于支持向量机来说，数据点被视为 
  
    
      
        p
      
    
    {\displaystyle p}
  
 维向量，希望知道是否可以用 
  
    
      
        (
        p
        −
        1
        )
      
    
    {\displaystyle (p-1)}
  
 维超平面来分开这些点。这就是所谓的线性分类器。可能有许多超平面可以把数据分类。最佳超平面的一个合理选择是以最大间隔把两个类分开的超平面。因此，要选择能够让到每边最近的数据点的距离最大化的超平面。如果存在这样的超平面，则称为最大间隔超平面，而其定义的线性分类器被称为最大间隔分类器，或者叫做最佳稳定性感知器。


## 定义
更正式地来说，支持向量机在高维或无限维空间中构造超平面或超平面集合，其可以用于分类、回归或其他任务。直觀來說，分類邊界距離最近的訓練資料點越遠越好，因為這樣可以缩小分類器的泛化誤差。

尽管原始问题可能是在有限维空间中陈述的，但用于区分的集合在该空间中往往线性不可分。为此，有人提出将原有限维空间映射到维数高得多的空间中，在该空间中进行分离可能会更容易。为了保持计算负荷合理，人们选择适合该问题的核函数 
  
    
      
        k
        (
        x
        ,
        y
        )
      
    
    {\displaystyle k(x,y)}
  
 来定义SVM方案使用的映射，以确保用原始空间中的变量可以很容易计算点积。高维空间中的超平面定义为与该空间中的某向量的点积是常数的点的集合。定义超平面的向量可以选择在数据集中出现的特征向量 
  
    
      
        
          x
          
            i
          
        
      
    
    {\displaystyle x_{i}}
  
 的图像的参数 
  
    
      
        
          α
          
            i
          
        
      
    
    {\displaystyle \alpha _{i}}
  
 的线性组合。通过选择超平面，被映射到超平面上的特征空间中的点集 
  
    
      
        x
      
    
    {\displaystyle x}
  
 由以下关系定义：
  
    
      
        
          
            ∑
            
              i
            
          
          
            α
            
              i
            
          
          k
          (
          
            x
            
              i
            
          
          ,
          x
          )
          =
          
            c
            o
            n
            s
            t
            a
            n
            t
          
          .
        
      
    
    {\displaystyle \textstyle \sum _{i}\alpha _{i}k(x_{i},x)=\mathrm {constant} .}
  
  注意，如果随着 
  
    
      
        y
      
    
    {\displaystyle y}
  
 逐渐远离 
  
    
      
        x
      
    
    {\displaystyle x}
  
，
  
    
      
        k
        (
        x
        ,
        y
        )
      
    
    {\displaystyle k(x,y)}
  
 变小，则求和中的每一项都是在衡量测试点 
  
    
      
        x
      
    
    {\displaystyle x}
  
 与对应的数据基点 
  
    
      
        
          x
          
            i
          
        
      
    
    {\displaystyle x_{i}}
  
 的接近程度。这样，上述内核的总和可以用于衡量每个测试点相对于待分离的集合中的数据点的相对接近度。


## 应用
用于文本和超文本的分类，在归纳和直推方法中都可以显著减少所需要的有类标的样本数。
用于图像分类。实验结果显示：在经过三到四轮相关反馈之后，比起传统的查询优化方案，支持向量机能够取得明显更高的搜索准确度。这同样也适用于图像分割系统，比如使用Vapnik所建议的使用特权方法的修改版本SVM的那些图像分割系统。
用于手写字体识别。
用于医学中分类蛋白质，超过90%的化合物能够被正确分类。基于支持向量机权重的置换测试已被建议作为一种机制，用于解释的支持向量机模型。支持向量机权重也被用来解释过去的SVM模型。为识别模型用于进行预测的特征而对支持向量机模型做出事后解释是在生物科学中具有特殊意义的相对较新的研究领域。


## 历史
原始SVM算法是由蘇聯數學家弗拉基米尔·瓦普尼克和亞歷克塞·澤范蘭傑斯于1963年发明的。1992年，伯恩哈德·E·博瑟（Bernhard E. Boser）、伊莎贝尔·M·盖昂（Isabelle M. Guyon）和瓦普尼克提出了一种通过将核技巧应用于最大间隔超平面来创建非线性分类器的方法。当前标准的前身（软间隔）由科琳娜·科特斯和瓦普尼克于1993年提出，并于1995年发表。


## 线性SVM
我们考虑以下形式的 
  
    
      
        n
      
    
    {\displaystyle n}
  
 点测试集：

  
    
      
        (
        
          
            
              
                x
                →
              
            
          
          
            1
          
        
        ,
        
          y
          
            1
          
        
        )
        ,
        
        …
        ,
        
        (
        
          
            
              
                x
                →
              
            
          
          
            n
          
        
        ,
        
          y
          
            n
          
        
        )
      
    
    {\displaystyle ({\vec {x}}_{1},y_{1}),\,\ldots ,\,({\vec {x}}_{n},y_{n})}
  

其中 
  
    
      
        
          y
          
            i
          
        
      
    
    {\displaystyle y_{i}}
  
 是 1 或者 −1，表明点 
  
    
      
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
      
    
    {\displaystyle {\vec {x}}_{i}}
  
 所属的类。
  
    
      
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
      
    
    {\displaystyle {\vec {x}}_{i}}
  
 中每个都是一个 
  
    
      
        p
      
    
    {\displaystyle p}
  
 维实向量。我们要求将 
  
    
      
        
          y
          
            i
          
        
        =
        1
      
    
    {\displaystyle y_{i}=1}
  
 的点集 
  
    
      
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
      
    
    {\displaystyle {\vec {x}}_{i}}
  
 与 
  
    
      
        
          y
          
            i
          
        
        =
        −
        1
      
    
    {\displaystyle y_{i}=-1}
  
 的点集分开的 “最大间隔超平面”，使得超平面与最近的点 
  
    
      
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
      
    
    {\displaystyle {\vec {x}}_{i}}
  
 之间的距离最大化。
任何超平面都可以写作满足下面方程的点集 
  
    
      
        
          
            
              x
              →
            
          
        
      
    
    {\displaystyle {\vec {x}}}
  

  
    
      
        
          
            
              w
              →
            
          
        
        ⋅
        
          
            
              x
              →
            
          
        
        −
        b
        =
        0
        ,
        
      
    
    {\displaystyle {\vec {w}}\cdot {\vec {x}}-b=0,\,}
  
 
其中 
  
    
      
        
          
            
              w
              →
            
          
        
      
    
    {\displaystyle {\vec {w}}}
  
（不必是归一化的）是该法向量。参数 
  
    
      
        
          
            
              b
              
                ‖
                
                  
                    
                      w
                      →
                    
                  
                
                ‖
              
            
          
        
      
    
    {\displaystyle {\tfrac {b}{\|{\vec {w}}\|}}}
  
 决定从原点沿法向量 
  
    
      
        
          
            
              w
              →
            
          
        
      
    
    {\displaystyle {\vec {w}}}
  
 到超平面的偏移量。


### 硬间隔
如果这些训练数据是线性可分的，可以选择分离两类数据的两个平行超平面，使得它们之间的距离尽可能大。在这两个超平面范围内的区域称为“间隔”，最大间隔超平面是位于它们正中间的超平面。这些超平面可以由方程：

  
    
      
        
          
            
              w
              →
            
          
        
        ⋅
        
          
            
              x
              →
            
          
        
        −
        b
        =
        1
        
      
    
    {\displaystyle {\vec {w}}\cdot {\vec {x}}-b=1\,}
  

或是

  
    
      
        
          
            
              w
              →
            
          
        
        ⋅
        
          
            
              x
              →
            
          
        
        −
        b
        =
        −
        1.
        
      
    
    {\displaystyle {\vec {w}}\cdot {\vec {x}}-b=-1.\,}
  

来表示。通过几何不难得到这两个超平面之间的距离是 
  
    
      
        
          
            
              2
              
                ‖
                
                  
                    
                      w
                      →
                    
                  
                
                ‖
              
            
          
        
      
    
    {\displaystyle {\tfrac {2}{\|{\vec {w}}\|}}}
  
，因此要使两平面间的距离最大，我们需要最小化 
  
    
      
        ‖
        
          
            
              w
              →
            
          
        
        ‖
      
    
    {\displaystyle \|{\vec {w}}\|}
  
。同时为了使得样本数据点都在超平面的间隔区以外，我们需要保证对于所有的 
  
    
      
        i
      
    
    {\displaystyle i}
  
 满足其中的一个条件：

  
    
      
        
          
            
              w
              →
            
          
        
        ⋅
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
        −
        b
        ≥
        1
        ,
      
    
    {\displaystyle {\vec {w}}\cdot {\vec {x}}_{i}-b\geq 1,}
  
          若 
  
    
      
        
          y
          
            i
          
        
        =
        1
      
    
    {\displaystyle y_{i}=1}
  

或是

  
    
      
        
          
            
              w
              →
            
          
        
        ⋅
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
        −
        b
        ≤
        −
        1
        ,
      
    
    {\displaystyle {\vec {w}}\cdot {\vec {x}}_{i}-b\leq -1,}
  
      若 
  
    
      
        
          y
          
            i
          
        
        =
        −
        1.
      
    
    {\displaystyle y_{i}=-1.}
  

这些约束表明每个数据点都必须位于间隔的正确一侧。
这两个式子可以写作：

  
    
      
        
          y
          
            i
          
        
        (
        
          
            
              w
              →
            
          
        
        ⋅
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
        −
        b
        )
        ≥
        1
        ,
        
        
           for all 
        
        1
        ≤
        i
        ≤
        n
        .
        
        
        (
        1
        )
      
    
    {\displaystyle y_{i}({\vec {w}}\cdot {\vec {x}}_{i}-b)\geq 1,\quad {\text{ for all }}1\leq i\leq n.\qquad \qquad (1)}
  

可以用这个式子一起来得到优化问题：“在 
  
    
      
        
          y
          
            i
          
        
        (
        
          
            
              w
              →
            
          
        
        ⋅
        
          
            
              
                x
                
                  i
                
              
              →
            
          
        
        −
        b
        )
        ≥
        1
      
    
    {\displaystyle y_{i}({\vec {w}}\cdot {\vec {x_{i}}}-b)\geq 1}
  
 条件下，最小化 
  
    
      
        ‖
        
          
            
              w
              →
            
          
        
        ‖
      
    
    {\displaystyle \|{\vec {w}}\|}
  
，对于 
  
    
      
        i
        =
        1
        ,
        
        …
        ,
        
        n
      
    
    {\displaystyle i=1,\,\ldots ,\,n}
  
"这个问题的解 
  
    
      
        
          
            
              w
              →
            
          
        
      
    
    {\displaystyle {\vec {w}}}
  
 与 
  
    
      
        b
      
    
    {\displaystyle b}
  
 决定了我们的分类器 
  
    
      
        
          
            
              x
              →
            
          
        
        ↦
        sgn
        ⁡
        (
        
          
            
              w
              →
            
          
        
        ⋅
        
          
            
              x
              →
            
          
        
        −
        b
        )
      
    
    {\displaystyle {\vec {x}}\mapsto \operatorname {sgn}({\vec {w}}\cdot {\vec {x}}-b)}
  
。
此几何描述的一个显而易见却重要的结果是，最大间隔超平面完全是由最靠近它的那些 
  
    
      
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
      
    
    {\displaystyle {\vec {x}}_{i}}
  
 确定的。这些 
  
    
      
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
      
    
    {\displaystyle {\vec {x}}_{i}}
  
 叫做支持向量。


### 软间隔
为了将SVM扩展到数据线性不可分的情况，我们引入铰链损失函数，
  
    
      
        max
        
          (
          
            0
            ,
            1
            −
            
              y
              
                i
              
            
            (
            
              
                
                  w
                  →
                
              
            
            ⋅
            
              
                
                  
                    x
                    
                      i
                    
                  
                  →
                
              
            
            −
            b
            )
          
          )
        
        .
      
    
    {\displaystyle \max \left(0,1-y_{i}({\vec {w}}\cdot {\vec {x_{i}}}-b)\right).}
  
当约束条件 (1) 满足时（也就是如果 
  
    
      
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
      
    
    {\displaystyle {\vec {x}}_{i}}
  
 位于边距的正确一侧）此函数为零。对于间隔的错误一侧的数据，该函数的值与距间隔的距离成正比。
然后我们希望最小化
  
    
      
        
          [
          
            
              
                1
                n
              
            
            
              ∑
              
                i
                =
                1
              
              
                n
              
            
            max
            
              (
              
                0
                ,
                1
                −
                
                  y
                  
                    i
                  
                
                (
                
                  
                    
                      w
                      →
                    
                  
                
                ⋅
                
                  
                    
                      
                        x
                        
                          i
                        
                      
                      →
                    
                  
                
                −
                b
                )
              
              )
            
          
          ]
        
        +
        λ
        ‖
        
          
            
              w
              →
            
          
        
        
          ‖
          
            2
          
        
        ,
      
    
    {\displaystyle \left[{\frac {1}{n}}\sum _{i=1}^{n}\max \left(0,1-y_{i}({\vec {w}}\cdot {\vec {x_{i}}}-b)\right)\right]+\lambda \lVert {\vec {w}}\rVert ^{2},}
  
其中参数 
  
    
      
        λ
      
    
    {\displaystyle \lambda }
  
 用来权衡增加间隔大小与确保 
  
    
      
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
      
    
    {\displaystyle {\vec {x}}_{i}}
  
 位于间隔的正确一侧之间的关系。因此，对于足够小的 
  
    
      
        λ
      
    
    {\displaystyle \lambda }
  
 值，如果输入数据是可以线性分类的，则软间隔SVM与硬间隔SVM将表现相同，但即使不可线性分类，仍能学习出可行的分类规则。


## 非线性分类
弗拉基米尔·瓦普尼克在1963年提出的原始最大间隔超平面算法构造了一个线性分类器。而1992年，伯恩哈德·E·博瑟（Bernhard E. Boser）、伊莎贝尔·M·盖昂（Isabelle M. Guyon）和瓦普尼克提出了一种通过将核技巧（最初由Aizerman et al.提出）应用于最大边界超平面来创建非线性分类器的方法。所得到的算法形式上类似，除了把点积换成了非线性核函数。这就允许算法在变换后的特征空间中拟合最大间隔超平面。该变换可以是非线性的，而变换空间是高维的；虽然分类器是变换后的特征空间中的超平面，但它在原始输入空间中可以是非线性的。
值得注意的是，更高维的特征空间增加了支持向量机的泛化误差，但给定足够多的样本，算法仍能表现良好。
常见的核函数包括：

齊次多項式：
  
    
      
        k
        (
        
          
            
              
                x
                
                  i
                
              
              →
            
          
        
        ,
        
          
            
              
                x
                
                  j
                
              
              →
            
          
        
        )
        =
        (
        
          
            
              
                x
                
                  i
                
              
              →
            
          
        
        ⋅
        
          
            
              
                x
                
                  j
                
              
              →
            
          
        
        
          )
          
            d
          
        
      
    
    {\displaystyle k({\vec {x_{i}}},{\vec {x_{j}}})=({\vec {x_{i}}}\cdot {\vec {x_{j}}})^{d}}
  

非齐次多项式：
  
    
      
        k
        (
        
          
            
              
                x
                
                  i
                
              
              →
            
          
        
        ,
        
          
            
              
                x
                
                  j
                
              
              →
            
          
        
        )
        =
        (
        
          
            
              
                x
                
                  i
                
              
              →
            
          
        
        ⋅
        
          
            
              
                x
                
                  j
                
              
              →
            
          
        
        +
        1
        
          )
          
            d
          
        
      
    
    {\displaystyle k({\vec {x_{i}}},{\vec {x_{j}}})=({\vec {x_{i}}}\cdot {\vec {x_{j}}}+1)^{d}}
  

高斯径向基函数：
  
    
      
        k
        (
        
          
            
              
                x
                
                  i
                
              
              →
            
          
        
        ,
        
          
            
              
                x
                
                  j
                
              
              →
            
          
        
        )
        =
        exp
        ⁡
        (
        −
        γ
        ‖
        
          
            
              
                x
                
                  i
                
              
              →
            
          
        
        −
        
          
            
              
                x
                
                  j
                
              
              →
            
          
        
        
          ‖
          
            2
          
        
        )
      
    
    {\displaystyle k({\vec {x_{i}}},{\vec {x_{j}}})=\exp(-\gamma \|{\vec {x_{i}}}-{\vec {x_{j}}}\|^{2})}
  
，其中 
  
    
      
        γ
        >
        0
      
    
    {\displaystyle \gamma >0}
  
。有时参数化表示 
  
    
      
        γ
        =
        1
        
          /
        
        
          2
          
            σ
            
              2
            
          
        
      
    
    {\displaystyle \gamma =1/{2\sigma ^{2}}}
  

双曲正切：
  
    
      
        k
        (
        
          
            
              
                x
                
                  i
                
              
              →
            
          
        
        ,
        
          
            
              
                x
                
                  j
                
              
              →
            
          
        
        )
        =
        tanh
        ⁡
        (
        κ
        
          
            
              
                x
                
                  i
                
              
              →
            
          
        
        ⋅
        
          
            
              
                x
                
                  j
                
              
              →
            
          
        
        +
        c
        )
      
    
    {\displaystyle k({\vec {x_{i}}},{\vec {x_{j}}})=\tanh(\kappa {\vec {x_{i}}}\cdot {\vec {x_{j}}}+c)}
  
，其中一些（而非所有）
  
    
      
        κ
        >
        0
      
    
    {\displaystyle \kappa >0}
  
 且 
  
    
      
        c
        <
        0
      
    
    {\displaystyle c<0}
  

由等式 
  
    
      
        k
        (
        
          
            
              
                x
                
                  i
                
              
              →
            
          
        
        ,
        
          
            
              
                x
                
                  j
                
              
              →
            
          
        
        )
        =
        φ
        (
        
          
            
              
                x
                
                  i
                
              
              →
            
          
        
        )
        ⋅
        φ
        (
        
          
            
              
                x
                
                  j
                
              
              →
            
          
        
        )
      
    
    {\displaystyle k({\vec {x_{i}}},{\vec {x_{j}}})=\varphi ({\vec {x_{i}}})\cdot \varphi ({\vec {x_{j}}})}
  
，核函数与变换 
  
    
      
        φ
        (
        
          
            
              
                x
                
                  i
                
              
              →
            
          
        
        )
      
    
    {\displaystyle \varphi ({\vec {x_{i}}})}
  
 有关。变换空间中也有 w 值，
  
    
      
        
          
            
              
                w
                →
              
            
          
          =
          
            ∑
            
              i
            
          
          
            α
            
              i
            
          
          
            y
            
              i
            
          
          φ
          (
          
            
              
                
                  x
                  →
                
              
            
            
              i
            
          
          )
        
      
    
    {\displaystyle \textstyle {\vec {w}}=\sum _{i}\alpha _{i}y_{i}\varphi ({\vec {x}}_{i})}
  
。与 w 的点积也要用核技巧来计算，即 
  
    
      
        
          
            
              
                w
                →
              
            
          
          ⋅
          φ
          (
          
            
              
                x
                →
              
            
          
          )
          =
          
            ∑
            
              i
            
          
          
            α
            
              i
            
          
          
            y
            
              i
            
          
          k
          (
          
            
              
                
                  x
                  →
                
              
            
            
              i
            
          
          ,
          
            
              
                x
                →
              
            
          
          )
        
      
    
    {\displaystyle \textstyle {\vec {w}}\cdot \varphi ({\vec {x}})=\sum _{i}\alpha _{i}y_{i}k({\vec {x}}_{i},{\vec {x}})}
  
。


## 计算SVM分类器
计算（软间隔）SVM分类器等同于使下面表达式最小化
  
    
      
        
          [
          
            
              
                1
                n
              
            
            
              ∑
              
                i
                =
                1
              
              
                n
              
            
            max
            
              (
              
                0
                ,
                1
                −
                
                  y
                  
                    i
                  
                
                (
                w
                ⋅
                
                  x
                  
                    i
                  
                
                +
                b
                )
              
              )
            
          
          ]
        
        +
        λ
        ‖
        w
        
          ‖
          
            2
          
        
        .
        
        (
        2
        )
      
    
    {\displaystyle \left[{\frac {1}{n}}\sum _{i=1}^{n}\max \left(0,1-y_{i}(w\cdot x_{i}+b)\right)\right]+\lambda \lVert w\rVert ^{2}.\qquad (2)}
  
如上所述，由于我们关注的是软间隔分类器，
  
    
      
        λ
      
    
    {\displaystyle \lambda }
  
 选择足够小的值就能得到线性可分类输入数据的硬间隔分类器。下面会详细介绍将(2)简化为二次规划问题的经典方法。之后会讨论一些最近才出现的方法，如次梯度下降法和坐标下降法。


### 原型
最小化(2)可以用下面的方式改写为目标函数可微的约束优化问题。
对所有 
  
    
      
        i
        ∈
        {
        1
        ,
        
        …
        ,
        
        n
        }
      
    
    {\displaystyle i\in \{1,\,\ldots ,\,n\}}
  
 我们引入变量 
  
    
      
        
          ζ
          
            i
          
        
        =
        max
        
          (
          
            0
            ,
            1
            −
            
              y
              
                i
              
            
            (
            w
            ⋅
            
              x
              
                i
              
            
            +
            b
            )
          
          )
        
      
    
    {\displaystyle \zeta _{i}=\max \left(0,1-y_{i}(w\cdot x_{i}+b)\right)}
  
。注意到 
  
    
      
        
          ζ
          
            i
          
        
      
    
    {\displaystyle \zeta _{i}}
  
 是满足 
  
    
      
        
          y
          
            i
          
        
        (
        w
        ⋅
        
          x
          
            i
          
        
        +
        b
        )
        ≥
        1
        −
        
          ζ
          
            i
          
        
      
    
    {\displaystyle y_{i}(w\cdot x_{i}+b)\geq 1-\zeta _{i}}
  
 的最小非负数。

因此，我们可以将优化问题叙述如下
  
    
      
        
          minimize 
        
        
          
            1
            n
          
        
        
          ∑
          
            i
            =
            1
          
          
            n
          
        
        
          ζ
          
            i
          
        
        +
        λ
        ‖
        w
        
          ‖
          
            2
          
        
      
    
    {\displaystyle {\text{minimize }}{\frac {1}{n}}\sum _{i=1}^{n}\zeta _{i}+\lambda \|w\|^{2}}
  

  
    
      
        
          subject to 
        
        
          y
          
            i
          
        
        (
        
          x
          
            i
          
        
        ⋅
        w
        +
        b
        )
        ≥
        1
        −
        
          ζ
          
            i
          
        
        
        
           and 
        
        
        
          ζ
          
            i
          
        
        ≥
        0
        ,
        
        
          for all 
        
        i
        .
      
    
    {\displaystyle {\text{subject to }}y_{i}(x_{i}\cdot w+b)\geq 1-\zeta _{i}\,{\text{ and }}\,\zeta _{i}\geq 0,\,{\text{for all }}i.}
  
 这就叫做原型问题。


### 对偶型
通过求解上述问题的拉格朗日对偶，得到简化的问题
  
    
      
        
          maximize
        
        
        
        f
        (
        
          c
          
            1
          
        
        …
        
          c
          
            n
          
        
        )
        =
        
          ∑
          
            i
            =
            1
          
          
            n
          
        
        
          c
          
            i
          
        
        −
        
          
            1
            2
          
        
        
          ∑
          
            i
            =
            1
          
          
            n
          
        
        
          ∑
          
            j
            =
            1
          
          
            n
          
        
        
          y
          
            i
          
        
        
          c
          
            i
          
        
        (
        
          x
          
            i
          
        
        ⋅
        
          x
          
            j
          
        
        )
        
          y
          
            j
          
        
        
          c
          
            j
          
        
        ,
      
    
    {\displaystyle {\text{maximize}}\,\,f(c_{1}\ldots c_{n})=\sum _{i=1}^{n}c_{i}-{\frac {1}{2}}\sum _{i=1}^{n}\sum _{j=1}^{n}y_{i}c_{i}(x_{i}\cdot x_{j})y_{j}c_{j},}
  

  
    
      
        
          subject to 
        
        
          ∑
          
            i
            =
            1
          
          
            n
          
        
        
          c
          
            i
          
        
        
          y
          
            i
          
        
        =
        0
        ,
        
        
          and 
        
        0
        ≤
        
          c
          
            i
          
        
        ≤
        
          
            1
            
              2
              n
              λ
            
          
        
        
        
          for all 
        
        i
        .
      
    
    {\displaystyle {\text{subject to }}\sum _{i=1}^{n}c_{i}y_{i}=0,\,{\text{and }}0\leq c_{i}\leq {\frac {1}{2n\lambda }}\;{\text{for all }}i.}
  
这就叫做对偶问题。由于对偶最小化问题是受线性约束的 
  
    
      
        
          c
          
            i
          
        
      
    
    {\displaystyle c_{i}}
  
 的二次函数，所以它可以通过二次规划算法高效地解出。
这里，变量 
  
    
      
        
          c
          
            i
          
        
      
    
    {\displaystyle c_{i}}
  
 定义为满足
  
    
      
        
          
            
              w
              →
            
          
        
        =
        
          ∑
          
            i
            =
            1
          
          
            n
          
        
        
          c
          
            i
          
        
        
          y
          
            i
          
        
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
      
    
    {\displaystyle {\vec {w}}=\sum _{i=1}^{n}c_{i}y_{i}{\vec {x}}_{i}}
  
. 此外，当 
  
    
      
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
      
    
    {\displaystyle {\vec {x}}_{i}}
  
 恰好在间隔的正确一侧时 
  
    
      
        
          c
          
            i
          
        
        =
        0
      
    
    {\displaystyle c_{i}=0}
  
，且当 
  
    
      
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
      
    
    {\displaystyle {\vec {x}}_{i}}
  
 位于间隔的边界时 
  
    
      
        0
        <
        
          c
          
            i
          
        
        <
        (
        2
        n
        λ
        
          )
          
            −
            1
          
        
      
    
    {\displaystyle 0<c_{i}<(2n\lambda )^{-1}}
  
。因此，
  
    
      
        
          
            
              w
              →
            
          
        
      
    
    {\displaystyle {\vec {w}}}
  
 可以写为支持向量的线性组合。
可以通过在间隔的边界上找到一个 
  
    
      
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
      
    
    {\displaystyle {\vec {x}}_{i}}
  
 并求解
  
    
      
        
          y
          
            i
          
        
        (
        
          
            
              w
              →
            
          
        
        ⋅
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
        +
        b
        )
        =
        1
        
        ⟺
        
        b
        =
        
          y
          
            i
          
        
        −
        
          
            
              w
              →
            
          
        
        ⋅
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
        .
      
    
    {\displaystyle y_{i}({\vec {w}}\cdot {\vec {x}}_{i}+b)=1\iff b=y_{i}-{\vec {w}}\cdot {\vec {x}}_{i}.}
  

得到偏移量 
  
    
      
        b
      
    
    {\displaystyle b}
  
。（注意由于 
  
    
      
        
          y
          
            i
          
        
        =
        ±
        1
      
    
    {\displaystyle y_{i}=\pm 1}
  
 因而 
  
    
      
        
          y
          
            i
          
          
            −
            1
          
        
        =
        
          y
          
            i
          
        
      
    
    {\displaystyle y_{i}^{-1}=y_{i}}
  
。）


### 核技巧
假设我们要学习与变换后数据点 
  
    
      
        φ
        (
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
        )
      
    
    {\displaystyle \varphi ({\vec {x}}_{i})}
  
 的线性分类规则对应的非线性分类规则。此外，我们有一个满足 
  
    
      
        k
        (
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
        ,
        
          
            
              
                x
                →
              
            
          
          
            j
          
        
        )
        =
        φ
        (
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
        )
        ⋅
        φ
        (
        
          
            
              
                x
                →
              
            
          
          
            j
          
        
        )
      
    
    {\displaystyle k({\vec {x}}_{i},{\vec {x}}_{j})=\varphi ({\vec {x}}_{i})\cdot \varphi ({\vec {x}}_{j})}
  
 的核函数 
  
    
      
        k
      
    
    {\displaystyle k}
  
。

我们知道变换空间中的分类向量 
  
    
      
        
          
            
              w
              →
            
          
        
      
    
    {\displaystyle {\vec {w}}}
  
 满足
  
    
      
        
          
            
              w
              →
            
          
        
        =
        
          ∑
          
            i
            =
            1
          
          
            n
          
        
        
          c
          
            i
          
        
        
          y
          
            i
          
        
        φ
        (
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
        )
        ,
      
    
    {\displaystyle {\vec {w}}=\sum _{i=1}^{n}c_{i}y_{i}\varphi ({\vec {x}}_{i}),}
  
其中 
  
    
      
        
          c
          
            i
          
        
      
    
    {\displaystyle c_{i}}
  
 可以通过求解优化问题
  
    
      
        
          
            
              
                
                  maximize
                
                
                
                f
                (
                
                  c
                  
                    1
                  
                
                …
                
                  c
                  
                    n
                  
                
                )
              
              
                
                =
                
                  ∑
                  
                    i
                    =
                    1
                  
                  
                    n
                  
                
                
                  c
                  
                    i
                  
                
                −
                
                  
                    1
                    2
                  
                
                
                  ∑
                  
                    i
                    =
                    1
                  
                  
                    n
                  
                
                
                  ∑
                  
                    j
                    =
                    1
                  
                  
                    n
                  
                
                
                  y
                  
                    i
                  
                
                
                  c
                  
                    i
                  
                
                (
                φ
                (
                
                  
                    
                      
                        x
                        →
                      
                    
                  
                  
                    i
                  
                
                )
                ⋅
                φ
                (
                
                  
                    
                      
                        x
                        →
                      
                    
                  
                  
                    j
                  
                
                )
                )
                
                  y
                  
                    j
                  
                
                
                  c
                  
                    j
                  
                
              
            
            
              
              
                
                =
                
                  ∑
                  
                    i
                    =
                    1
                  
                  
                    n
                  
                
                
                  c
                  
                    i
                  
                
                −
                
                  
                    1
                    2
                  
                
                
                  ∑
                  
                    i
                    =
                    1
                  
                  
                    n
                  
                
                
                  ∑
                  
                    j
                    =
                    1
                  
                  
                    n
                  
                
                
                  y
                  
                    i
                  
                
                
                  c
                  
                    i
                  
                
                k
                (
                
                  
                    
                      
                        x
                        →
                      
                    
                  
                  
                    i
                  
                
                ,
                
                  
                    
                      
                        x
                        →
                      
                    
                  
                  
                    j
                  
                
                )
                
                  y
                  
                    j
                  
                
                
                  c
                  
                    j
                  
                
              
            
          
        
      
    
    {\displaystyle {\begin{aligned}{\text{maximize}}\,\,f(c_{1}\ldots c_{n})&=\sum _{i=1}^{n}c_{i}-{\frac {1}{2}}\sum _{i=1}^{n}\sum _{j=1}^{n}y_{i}c_{i}(\varphi ({\vec {x}}_{i})\cdot \varphi ({\vec {x}}_{j}))y_{j}c_{j}\\&=\sum _{i=1}^{n}c_{i}-{\frac {1}{2}}\sum _{i=1}^{n}\sum _{j=1}^{n}y_{i}c_{i}k({\vec {x}}_{i},{\vec {x}}_{j})y_{j}c_{j}\\\end{aligned}}}
  

  
    
      
        
          subject to 
        
        
          ∑
          
            i
            =
            1
          
          
            n
          
        
        
          c
          
            i
          
        
        
          y
          
            i
          
        
        =
        0
        ,
        
        
          and 
        
        0
        ≤
        
          c
          
            i
          
        
        ≤
        
          
            1
            
              2
              n
              λ
            
          
        
        
        
          for all 
        
        i
        .
      
    
    {\displaystyle {\text{subject to }}\sum _{i=1}^{n}c_{i}y_{i}=0,\,{\text{and }}0\leq c_{i}\leq {\frac {1}{2n\lambda }}\;{\text{for all }}i.}
  
得到。与前面一样，可以使用二次规划来求解系数 
  
    
      
        
          c
          
            i
          
        
      
    
    {\displaystyle c_{i}}
  
。同样，我们可以找到让 
  
    
      
        0
        <
        
          c
          
            i
          
        
        <
        (
        2
        n
        λ
        
          )
          
            −
            1
          
        
      
    
    {\displaystyle 0<c_{i}<(2n\lambda )^{-1}}
  
 的索引 
  
    
      
        i
      
    
    {\displaystyle i}
  
，使得 
  
    
      
        φ
        (
        
          
            
              
                x
                →
              
            
          
          
            i
          
        
        )
      
    
    {\displaystyle \varphi ({\vec {x}}_{i})}
  
 位于变换空间中间隔的边界上，然后求解
  
    
      
        
          
            
              
                b
                =
                
                  
                    
                      w
                      →
                    
                  
                
                ⋅
                φ
                (
                
                  
                    
                      
                        x
                        →
                      
                    
                  
                  
                    i
                  
                
                )
                −
                
                  y
                  
                    i
                  
                
              
              
                
                =
                
                  [
                  
                    
                      ∑
                      
                        k
                        =
                        1
                      
                      
                        n
                      
                    
                    
                      c
                      
                        k
                      
                    
                    
                      y
                      
                        k
                      
                    
                    φ
                    (
                    
                      
                        
                          
                            x
                            →
                          
                        
                      
                      
                        k
                      
                    
                    )
                    ⋅
                    φ
                    (
                    
                      
                        
                          
                            x
                            →
                          
                        
                      
                      
                        i
                      
                    
                    )
                  
                  ]
                
                −
                
                  y
                  
                    i
                  
                
              
            
            
              
              
                
                =
                
                  [
                  
                    
                      ∑
                      
                        k
                        =
                        1
                      
                      
                        n
                      
                    
                    
                      c
                      
                        k
                      
                    
                    
                      y
                      
                        k
                      
                    
                    k
                    (
                    
                      
                        
                          
                            x
                            →
                          
                        
                      
                      
                        k
                      
                    
                    ,
                    
                      
                        
                          
                            x
                            →
                          
                        
                      
                      
                        i
                      
                    
                    )
                  
                  ]
                
                −
                
                  y
                  
                    i
                  
                
                .
              
            
          
        
      
    
    {\displaystyle {\begin{aligned}b={\vec {w}}\cdot \varphi ({\vec {x}}_{i})-y_{i}&=\left[\sum _{k=1}^{n}c_{k}y_{k}\varphi ({\vec {x}}_{k})\cdot \varphi ({\vec {x}}_{i})\right]-y_{i}\\&=\left[\sum _{k=1}^{n}c_{k}y_{k}k({\vec {x}}_{k},{\vec {x}}_{i})\right]-y_{i}.\end{aligned}}}
  
最后，可以通过计算下式来分类新点 
  
    
      
        
          
            
              z
              →
            
          
        
        ↦
        sgn
        ⁡
        (
        
          
            
              w
              →
            
          
        
        ⋅
        φ
        (
        
          
            
              z
              →
            
          
        
        )
        +
        b
        )
        =
        sgn
        ⁡
        
          (
          
            
              [
              
                
                  ∑
                  
                    i
                    =
                    1
                  
                  
                    n
                  
                
                
                  c
                  
                    i
                  
                
                
                  y
                  
                    i
                  
                
                k
                (
                
                  
                    
                      
                        x
                        →
                      
                    
                  
                  
                    i
                  
                
                ,
                
                  
                    
                      z
                      →
                    
                  
                
                )
              
              ]
            
            +
            b
          
          )
        
        .
      
    
    {\displaystyle {\vec {z}}\mapsto \operatorname {sgn}({\vec {w}}\cdot \varphi ({\vec {z}})+b)=\operatorname {sgn} \left(\left[\sum _{i=1}^{n}c_{i}y_{i}k({\vec {x}}_{i},{\vec {z}})\right]+b\right).}
  


### 现代方法
用于找到SVM分类器的最近的算法包括次梯度下降和坐标下降。当处理大的稀疏数据集时，这两种技术已经被证明有着显著的优点——当存在许多训练实例时次梯度法是特别有效的，并且当特征空间的维度高时，坐标下降特别有效。


#### 次梯度下降
SVM的次梯度下降算法直接用表达式
  
    
      
        f
        (
        
          
            
              w
              →
            
          
        
        ,
        b
        )
        =
        
          [
          
            
              
                1
                n
              
            
            
              ∑
              
                i
                =
                1
              
              
                n
              
            
            max
            
              (
              
                0
                ,
                1
                −
                
                  y
                  
                    i
                  
                
                (
                w
                ⋅
                
                  x
                  
                    i
                  
                
                +
                b
                )
              
              )
            
          
          ]
        
        +
        λ
        ‖
        w
        
          ‖
          
            2
          
        
        .
      
    
    {\displaystyle f({\vec {w}},b)=\left[{\frac {1}{n}}\sum _{i=1}^{n}\max \left(0,1-y_{i}(w\cdot x_{i}+b)\right)\right]+\lambda \lVert w\rVert ^{2}.}
  
注意 
  
    
      
        f
      
    
    {\displaystyle f}
  
 是 
  
    
      
        
          
            
              w
              →
            
          
        
      
    
    {\displaystyle {\vec {w}}}
  
 与 
  
    
      
        b
      
    
    {\displaystyle b}
  
 的凸函数。因此，可以采用传统的梯度下降（或SGD）方法，其中不是在函数梯度的方向上前进，而是在从函数的次梯度中选出的向量的方向上前进。该方法的优点在于，对于某些实现，迭代次数不随着数据点的数量 
  
    
      
        n
      
    
    {\displaystyle n}
  
 而增加或减少。


#### 坐标下降
SVM的坐标下降算法基于对偶问题

  
    
      
        
          maximize
        
        
        
        f
        (
        
          c
          
            1
          
        
        …
        
          c
          
            n
          
        
        )
        =
        
          ∑
          
            i
            =
            1
          
          
            n
          
        
        
          c
          
            i
          
        
        −
        
          
            1
            2
          
        
        
          ∑
          
            i
            =
            1
          
          
            n
          
        
        
          ∑
          
            j
            =
            1
          
          
            n
          
        
        
          y
          
            i
          
        
        
          c
          
            i
          
        
        (
        
          x
          
            i
          
        
        ⋅
        
          x
          
            j
          
        
        )
        
          y
          
            j
          
        
        
          c
          
            j
          
        
        ,
      
    
    {\displaystyle {\text{maximize}}\,\,f(c_{1}\ldots c_{n})=\sum _{i=1}^{n}c_{i}-{\frac {1}{2}}\sum _{i=1}^{n}\sum _{j=1}^{n}y_{i}c_{i}(x_{i}\cdot x_{j})y_{j}c_{j},}
  

  
    
      
        
          subject to 
        
        
          ∑
          
            i
            =
            1
          
          
            n
          
        
        
          c
          
            i
          
        
        
          y
          
            i
          
        
        =
        0
        ,
        
        
          and 
        
        0
        ≤
        
          c
          
            i
          
        
        ≤
        
          
            1
            
              2
              n
              λ
            
          
        
        
        
          for all 
        
        i
        .
      
    
    {\displaystyle {\text{subject to }}\sum _{i=1}^{n}c_{i}y_{i}=0,\,{\text{and }}0\leq c_{i}\leq {\frac {1}{2n\lambda }}\;{\text{for all }}i.}
  
对所有 
  
    
      
        i
        ∈
        {
        1
        ,
        
        …
        ,
        
        n
        }
      
    
    {\displaystyle i\in \{1,\,\ldots ,\,n\}}
  
 进行迭代，使系数 
  
    
      
        
          c
          
            i
          
        
      
    
    {\displaystyle c_{i}}
  
 的方向与 
  
    
      
        ∂
        f
        
          /
        
        ∂
        
          c
          
            i
          
        
      
    
    {\displaystyle \partial f/\partial c_{i}}
  
 一致。然后，将所得的系数向量 
  
    
      
        (
        
          c
          
            1
          
          ′
        
        ,
        
        …
        ,
        
        
          c
          
            n
          
          ′
        
        )
      
    
    {\displaystyle (c_{1}',\,\ldots ,\,c_{n}')}
  
 投影到满足给定约束的最接近的系数向量。（通常使用欧氏距离。）然后重复该过程，直到获得接近最佳的系数向量。所得的算法在实践中运行非常快，尽管已经证明的性能保证很少。


## 性质
SVM属于广义线性分类器的一族，并且可以解释为感知器的延伸。它们也可以被认为是吉洪诺夫正则化的特例。它们有一个特别的性质，就是可以同时最小化经验误差和最大化几何边缘区; 因此它们也被称为最大间隔分类器。
Meyer、Leisch和Hornik对SVM与其他分类器进行了比较。


### 参数选择
SVM的有效性取决于核函数、核参数和软间隔参数 C 的选择。
通常会选只有一个参数 
  
    
      
        γ
      
    
    {\displaystyle \gamma }
  
 的高斯核。C 和 
  
    
      
        γ
      
    
    {\displaystyle \gamma }
  
 的最佳组合通常通过在 C 和 
  
    
      
        γ
      
    
    {\displaystyle \gamma }
  
 为指数增长序列下网格搜索来选取，例如 
  
    
      
        C
        ∈
        {
        
          2
          
            −
            5
          
        
        ,
        
          2
          
            −
            3
          
        
        ,
        …
        ,
        
          2
          
            13
          
        
        ,
        
          2
          
            15
          
        
        }
      
    
    {\displaystyle C\in \{2^{-5},2^{-3},\dots ,2^{13},2^{15}\}}
  
; 
  
    
      
        γ
        ∈
        {
        
          2
          
            −
            15
          
        
        ,
        
          2
          
            −
            13
          
        
        ,
        …
        ,
        
          2
          
            1
          
        
        ,
        
          2
          
            3
          
        
        }
      
    
    {\displaystyle \gamma \in \{2^{-15},2^{-13},\dots ,2^{1},2^{3}\}}
  
。通常情况下，使用交叉驗證来检查参数选择的每一个组合，并选择具有最佳交叉验证精度的参数。或者，最近在贝叶斯优化中的工作可以用于选择C和γ，通常需要评估比网格搜索少得多的参数组合。或者，贝叶斯优化的最近进展可以用于选择 C 和 
  
    
      
        γ
      
    
    {\displaystyle \gamma }
  
，通常需要计算的参数组合比网格搜索少得多。然后，使用所选择的参数在整个训练集上训练用于测试和分类新数据的最终模型。


### 问题
SVM的潜在缺点包括以下方面：

需要对输入数据进行完全标记
未校准类成员概率
SVM仅直接适用于两类任务。因此，必须应用将多类任务减少到几个二元问题的算法；请参阅多类SVM一节。
解出的模型的参数很难理解。


## 延伸
支持向量聚類：支持向量聚類是一種建立在核函數上的類似方法，同適用於非監督學習和數據挖掘。它被認為是數據科學中的一種基本方法。
轉導支持向量機
多元分類支持向量機：SVM算法最初是為二值分類問題設計的，實現多分類的主要方法是將一個多分類問題轉化為多個二分類問題。常見方法包括“一對多法”和“一對一法”，一對多法是將某個類別的樣本歸為一類,其他剩餘的樣本歸為另一類，這樣k個類別的樣本就構造出了k個二分類SVM；一對一法則是在任意兩類樣本之間設計一個SVM。
支持向量回歸
結構化支持向量機：支持向量機可以被推廣為結構化的支持向量機，推廣後標籤空間是結構化的並且可能具有無限的大小。


## 实现
最大间隔超平面的参数是通过求解优化得到的。有几种专门的算法可用于快速解决由SVM产生的QP问题，它们主要依靠启发式算法将问题分解成更小、更易于处理的子问题。
另一种方法是使用内点法，其使用类似牛顿法的迭代找到卡羅需－庫恩－塔克條件下原型和对偶型的解。
这种方法不是去解决一系列分解问题，而是直接完全解决该问题。为了避免求解核矩阵很大的线性系统，在核技巧中经常使用矩阵的低秩近似。
另一个常见的方法是普莱特的序列最小优化算法（SMO），它把问题分成了若干个可以解析求解的二维子问题，这样就可以避免使用数值优化算法和矩阵存储。
线性支持向量机的特殊情况可以通过用于优化其类似问题邏輯斯諦迴歸的同类算法更高效求解；这类算法包括次梯度下降法（如PEGASOS）和坐标下降法（如LIBLINEAR）。LIBLINEAR有一些引人注目的训练时间上的特性。每次收敛迭代花费在读取训练数据上的时间是线性的，而且这些迭代还具有Q-线性收敛特性，使得算法非常快。
一般的核SVM也可以用次梯度下降法（P-packSVM）更快求解，在允许并行化时求解速度尤其快。
许多机器学习工具包都可以使用核SVM，有LIBSVM、MATLAB、SAS、SVMlight、kernlab、scikit-learn、Shogun、Weka、Shark、JKernelMachines、OpenCV等。


## 参见
核方法
费希尔核
多项式核函数
预测分析
相关向量机，函数形式与SVM相同的概率稀疏核模型
序列最小优化算法
空间映射


## 参考文献
### 引用
### 来源
## 外部链接
www.support-vector.net  The key book about the method, "An Introduction to Support Vector Machines" with online software
Burges, Christopher J. C.; A Tutorial on Support Vector Machines for Pattern Recognition （页面存档备份，存于互联网档案馆）, Data Mining and Knowledge Discovery 2:121–167, 1998
www.kernel-machines.org （页面存档备份，存于互联网档案馆） (general information and collection of research papers)
www.support-vector-machines.org （页面存档备份，存于互联网档案馆） (Literature, Review, Software, Links related to Support Vector Machines — Academic Site)
svmtutorial.online （页面存档备份，存于互联网档案馆） A simple introduction to SVM, easily accessible to anyone with basic background in mathematics
videolectures.net （页面存档备份，存于互联网档案馆） (SVM-related video lectures)
Karatzoglou, Alexandros et al.; Support Vector Machines in R （页面存档备份，存于互联网档案馆）, Journal of Statistical Software April 2006, Volume 15, Issue 9.
libsvm （页面存档备份，存于互联网档案馆）, LIBSVM is a popular library of SVM learners
liblinear （页面存档备份，存于互联网档案馆） is a library for large linear classification including some SVMs
Shark is a C++ machine learning library implementing various types of SVMs
dlib （页面存档备份，存于互联网档案馆） is a C++ library for working with kernel methods and SVMs
SVM light is a collection of software tools for learning and classification using SVM
SVMJS live demo （页面存档备份，存于互联网档案馆） is a GUI demo for JavaScript implementation of SVMs
Gesture Recognition Toolkit （页面存档备份，存于互联网档案馆） contains an easy to use wrapper for libsvm （页面存档备份，存于互联网档案馆）
