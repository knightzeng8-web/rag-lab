# 迁移学习

迁移学习（英語：Transfer learning）是属于机器学习的一种研究领域。它专注于存储已有问题的解决模型，并将其利用在其他不同但相关问题上。 比如说，用来辨识汽车的知识（或者是模型）也可以被用来提升识别卡车的能力。


## 历史
最早被引用的关于迁移学习的工作被认为属于洛麗安·普拉特。他在1993年制定了基于可辨识性的转移（DBT）算法。 
1997年，机器学习期刊发表了一期专门讨论迁移学习的期刊，而到了1998年，该领域已经发展到包括多任务学习，以及对其理论基础的更深入完善的分析。1998年，由普拉特和塞巴斯蒂安·特龍编辑的《Learning to Learn》便是对该主题的回顾。
迁移学习也被应用于认知科学，比如《Connection Science》杂志就于1996年出版了一版特殊期刊，描述了如何通过使用迁移学习重新利用已有神经网络。


## 定义
迁移学习是由域和任务定义的。域
  
    
      
        
          
            D
          
        
      
    
    {\displaystyle {\mathcal {D}}}
  
由特征空间
  
    
      
        
          
            X
          
        
      
    
    {\displaystyle {\mathcal {X}}}
  
和边缘概率分布
  
    
      
        P
        (
        X
        )
      
    
    {\displaystyle P(X)}
  
构成，其中
  
    
      
        X
        =
        {
        
          x
          
            1
          
        
        ,
        .
        .
        .
        ,
        
          x
          
            n
          
        
        }
        ∈
        
          
            X
          
        
      
    
    {\displaystyle X=\{x_{1},...,x_{n}\}\in {\mathcal {X}}}
  
。给定域
  
    
      
        
          
            D
          
        
        =
        {
        
          
            X
          
        
        ,
        P
        (
        X
        )
        }
      
    
    {\displaystyle {\mathcal {D}}=\{{\mathcal {X}},P(X)\}}
  
，任务由标签空间
  
    
      
        
          
            Y
          
        
      
    
    {\displaystyle {\mathcal {Y}}}
  
和目标预测函数
  
    
      
        f
        :
        
          
            X
          
        
        →
        
          
            Y
          
        
      
    
    {\displaystyle f:{\mathcal {X}}\rightarrow {\mathcal {Y}}}
  
两部分组成，函数
  
    
      
        f
      
    
    {\displaystyle f}
  
预测
  
    
      
        x
      
    
    {\displaystyle x}
  
对应的标签
  
    
      
        f
        (
        x
        )
      
    
    {\displaystyle f(x)}
  
。任务
  
    
      
        
          
            T
          
        
        =
        {
        
          
            Y
          
        
        ,
        f
        (
        x
        )
        }
      
    
    {\displaystyle {\mathcal {T}}=\{{\mathcal {Y}},f(x)\}}
  
是从含有样本对
  
    
      
        {
        
          x
          
            i
          
        
        ,
        
          y
          
            i
          
        
        }
      
    
    {\displaystyle \{x_{i},y_{i}\}}
  
的训练数据中学习得到的，其中
  
    
      
        
          x
          
            i
          
        
        ∈
        X
        ,
        
          y
          
            i
          
        
        ∈
        
          
            Y
          
        
      
    
    {\displaystyle x_{i}\in X,y_{i}\in {\mathcal {Y}}}
  
。
给定原域
  
    
      
        
          
            
              D
            
          
          
            S
          
        
      
    
    {\displaystyle {\mathcal {D}}_{S}}
  
及其任务
  
    
      
        
          
            
              T
            
          
          
            S
          
        
      
    
    {\displaystyle {\mathcal {T}}_{S}}
  
，目标域
  
    
      
        
          
            
              D
            
          
          
            T
          
        
      
    
    {\displaystyle {\mathcal {D}}_{T}}
  
及其任务
  
    
      
        
          
            
              T
            
          
          
            T
          
        
      
    
    {\displaystyle {\mathcal {T}}_{T}}
  
（满足
  
    
      
        
          
            
              D
            
          
          
            S
          
        
        ≠
        
          
            
              D
            
          
          
            T
          
        
      
    
    {\displaystyle {\mathcal {D}}_{S}\neq {\mathcal {D}}_{T}}
  
或
  
    
      
        
          
            
              T
            
          
          
            S
          
        
        ≠
        
          
            
              T
            
          
          
            T
          
        
      
    
    {\displaystyle {\mathcal {T}}_{S}\neq {\mathcal {T}}_{T}}
  
），迁移学习旨在通过利用
  
    
      
        
          
            
              D
            
          
          
            S
          
        
      
    
    {\displaystyle {\mathcal {D}}_{S}}
  
和
  
    
      
        
          
            
              T
            
          
          
            S
          
        
      
    
    {\displaystyle {\mathcal {T}}_{S}}
  
的知识，帮助学习
  
    
      
        
          
            
              D
            
          
          
            T
          
        
      
    
    {\displaystyle {\mathcal {D}}_{T}}
  
域的目标预测函数
  
    
      
        
          f
          
            T
          
        
        (
        ⋅
        )
      
    
    {\displaystyle f_{T}(\cdot )}
  
。


## 应用
迁移学习的算法基础可以源自马尔可夫逻辑网络和贝叶斯网络。迁移网络还被利用与发现癌症种类、建筑物人员限额、普适智能游戏玩家、语句分类以及筛选垃圾邮件（短信）。


## 機器學習以外
计算机领域的迁移学习，和心理学常常提到的学习迁移在概念上相似或有一定关系，但是两个领域在学术上的关系非常有限。


## 来源
Thrun, Sebastian; Pratt, Lorien. Learning to Learn. Springer Science & Business Media. 6 December 2012  [2022-12-23]. ISBN 978-1-4615-5529-2. （原始内容存档于2022-07-02）. 


## 参见
## 引用
