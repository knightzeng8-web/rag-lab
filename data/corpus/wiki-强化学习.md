# 强化学习

强化学习（英語：Reinforcement learning，簡稱RL）是机器学习中的一个领域，强调如何基于环境而行动，以取得最大化的预期利益。强化学习是除了监督学习和非监督学习之外的第三种基本的机器学习方法。与监督学习不同的是，强化学习不需要带标签的输入输出对，同时也无需对非最优解的精确地纠正。其关注点在于寻找（对未知领域的）探索和（对已有知识的）利用的平衡，强化学习中的“探索-利用”的交换，在多臂赌博机问题和有限MDP中研究得最多。
其灵感来源于心理学中的行为主义理论，即有机体如何在环境给予的奖励或惩罚的刺激下，逐步形成对刺激的预期，产生能获得最大利益的习惯性行为。这个方法具有普适性，因此在其他许多领域都有研究，例如博弈论、控制论、运筹学、信息论、仿真优化、多智能体系统、群体智能、统计学以及遗传算法。在运筹学和控制理论研究的语境下，强化学习被称作“近似动态规划”（approximate dynamic programming，ADP）。在最优控制理论中也有研究这个问题，虽然大部分的研究是关于最优解的存在和特性，并非是学习或者近似方面。在经济学和博弈论中，强化学习被用来解释在有限理性的条件下如何出现平衡。
在強化学习问题中，代理人（agent）與环境的交互通常被抽象为马尔可夫决策过程（Markov decision processes，MDP），因为很多强化学习算法在这种假设下才能使用动态规划的方法。传统的动态规划方法和强化学习算法的主要区别是，后者不需要关于MDP的知识，而且针对无法找到确切方法的大规模MDP。
对于时刻
  
    
      
        t
      
    
    {\displaystyle t}
  
下觀測（observation）
  
    
      
        
          o
          
            t
          
        
        ∈
        O
      
    
    {\displaystyle o_{t}\in O}
  
與環境實際狀態
  
    
      
        
          s
          
            t
          
        
        ∈
        S
      
    
    {\displaystyle s_{t}\in S}
  
的關係，MDP可以被分為：

完全可观测MDP，若
  
    
      
        
          o
          
            t
          
        
        =
        
          s
          
            t
          
        
      
    
    {\displaystyle o_{t}=s_{t}}
  

部分可观测MDP (POMDP)，若
  
    
      
        
          o
          
            t
          
        
        ⊂
        
          s
          
            t
          
        
      
    
    {\displaystyle o_{t}\subset s_{t}}
  

對於Q-learning，其對應的策略
  
    
      
        π
      
    
    {\displaystyle \pi }
  
可被定義為：

  
    
      
        π
        (
        a
        |
        s
        )
        =
        
          
            
              e
              
                Q
                (
                s
                ,
                a
                )
                
                  /
                
                τ
              
            
            
              
                ∑
                
                  
                    a
                    
                      ′
                    
                  
                  ∈
                  A
                
              
              
                e
                
                  Q
                  (
                  s
                  ,
                  
                    a
                    
                      ′
                    
                  
                  )
                  
                    /
                  
                  τ
                
              
            
          
        
      
    
    {\displaystyle \pi (a\vert s)={\frac {e^{Q(s,a)/\tau }}{\sum _{a^{\prime }\in A}e^{Q(s,a^{\prime })/\tau }}}}
  

其中
  
    
      
        τ
        ∈
        
          
            R
          
          
            +
          
        
      
    
    {\displaystyle \tau \in \mathbb {R} ^{+}}
  
为温度参数，用於控制策略的隨機程度。
  
    
      
        τ
      
    
    {\displaystyle \tau }
  
趨於0時，策略趨於貪心策略（
  
    
      
        
          
            a
            r
            g
            m
            a
            x
          
          
            a
            ∈
            A
          
        
        Q
        (
        s
        ,
        a
        )
      
    
    {\displaystyle \mathrm {argmax} _{a\in A}Q(s,a)}
  
）；
  
    
      
        τ
      
    
    {\displaystyle \tau }
  
趨於正無窮時，策略趨於均勻隨機。


## 介绍
由于其通用性很强，强化学习已经在诸如博弈论、控制论、运筹学、信息论、仿真优化、多智能体系统、群体智能和统计学等领域有了深入研究。在运筹学和控制文献中，强化学习被称为近似动态规划或神经动态规划。强化学习所感兴趣的问题在最优控制（一种关注最优解的存在性、表示和求解的理论，但较少涉及学习和近似）中也有所研究，尤其是环境的数学模型难以求得的时候。在经济学和博弈论中，强化学习可能被用来解释在有限的理性（rationality）下如何达到平衡状态。
基本的强化学习被建模为马尔可夫决策过程：

环境状态的集合
  
    
      
        S
      
    
    {\displaystyle S}
  
;
动作的集合
  
    
      
        A
      
    
    {\displaystyle A}
  
;
在状态之间转换的规则（转移概率矩阵）
  
    
      
        P
      
    
    {\displaystyle P}
  
；
规定转换后“即时奖励”的规则（奖励函数）
  
    
      
        R
      
    
    {\displaystyle R}
  
；
描述主体能够观察到什么的规则。
规则通常是帶有随机性的。主体通常可以观察即时奖励和最后一次转换。在许多模型中，主体被假设为可以观察现有的环境状态，这种情况称为“完全可观测”（full observability），反之则称为“部分可观测”（partial observability）。通常，主体被允许的动作是有限的，例如，在棋盤中棋子只能上、下、左、右移動，或是使用的钱不能多于所拥有的。
强化学习的主体与环境基于离散的时间步作用。在每一个时间
  
    
      
        t
      
    
    {\displaystyle t}
  
，主体接收到一个观测
  
    
      
        
          o
          
            t
          
        
      
    
    {\displaystyle o_{t}}
  
，通常其中包含奖励
  
    
      
        
          r
          
            t
          
        
      
    
    {\displaystyle r_{t}}
  
。然后，它从允许的集合中选择一个动作
  
    
      
        
          a
          
            t
          
        
      
    
    {\displaystyle a_{t}}
  
，然后送出到环境中去。环境则变化到一个新的状态
  
    
      
        
          s
          
            t
            +
            1
          
        
      
    
    {\displaystyle s_{t+1}}
  
，然后决定了和这个变化
  
    
      
        (
        
          s
          
            t
          
        
        ,
        
          a
          
            t
          
        
        ,
        
          s
          
            t
            +
            1
          
        
        )
      
    
    {\displaystyle (s_{t},a_{t},s_{t+1})}
  
相关联的奖励
  
    
      
        
          r
          
            t
            +
            1
          
        
      
    
    {\displaystyle r_{t+1}}
  
。强化学习主体的目标，是得到尽可能多的預期回報（return），其可以為獎勵
  
    
      
        
          ∑
          
            t
          
        
        
          r
          
            t
          
        
      
    
    {\displaystyle \sum _{t}r_{t}}
  
、帶折扣的回報（discounted return）
  
    
      
        ∑
        
          r
          
            t
          
        
        ⋅
        
          γ
          
            t
          
        
      
    
    {\displaystyle \sum r_{t}\cdot \gamma ^{t}}
  
，其中
  
    
      
        γ
        ∈
        (
        0
        ,
        1
        ]
      
    
    {\displaystyle \gamma \in (0,1]}
  
為回報因子，表示對短期獎勵的鼓勵。主体选择的动作是其历史的函数，它也可以选择随机的动作。
将这个主体的表现和自始自终以最优方式行动的主体相比较，它们之间的行动差异产生了“悔过”的概念。如果要接近最优的方案来行动，主体必须根据它的长时间行动序列进行推理：例如，要最大化我的未来收入，我最好现在去上学，虽然这样行动的即时货币奖励为负值。
因此，强化学习对于包含长期反馈的问题比短期反馈的表现更好。它在许多问题上得到应用，包括机器人控制、电梯调度、电信通讯，AlphaGo（蒙特卡洛树搜索+RL）和星际争霸II AI（AlphaStar）。
强化学习的强大能力来源于两个方面：使用样本来优化行为，使用函数近似来描述复杂的环境。它们使得强化学习可以使用在以下的复杂环境中：

模型的环境已知，且解析解不存在；
仅仅给出环境的模拟模型（模拟优化方法的问题）
从环境中获取信息的唯一办法是和它互动。前两个问题可以被考虑为规划问题，而最后一个问题可以被认为是genuine learning问题。使用强化学习的方法，这两种规划问题都可以被转化为机器学习问题。
策略迭代（Policy Iteration）是RL中策略梯度法的理论基础
值迭代（Value Iteration）与Q-learning存在收敛性等价证明


## 常用算法
蒙特卡洛学习 Monte-Carlo Learning
Temporal-Difference Learning
SARSA算法
Q学习


#### 现代基准补充
## 探索机制
强化学习需要比较聪明的探索机制，直接随机的对动作进行采样的方法性能比较差。虽然小规模的马氏过程已经被认识的比较清楚，这些性质很难在状态空间规模比较大的时候适用，这个时候相对简单的探索机制是更加现实的。
其中的一种方法就是 
  
    
      
        ϵ
      
    
    {\displaystyle \epsilon }
  
-貪婪演算法，这种方法会以比较大的概率(1-
  
    
      
        ϵ
      
    
    {\displaystyle \epsilon }
  
)去选择现在最好的动作。如果没有选择最优动作，就在剩下的动作中随机选择一个。
  
    
      
        ϵ
      
    
    {\displaystyle \epsilon }
  
 在这里是一个可调节的参数，更小的 
  
    
      
        ϵ
      
    
    {\displaystyle \epsilon }
  
 意味着算法会更加贪心。


#### 前沿方向补遗
多智能体RL  引入Nash Q-learning框架：
  
    
      
        
          Q
          
            i
          
        
        
          π
          
            ⋆
          
        
        (
        s
        ,
        a
        )
        =
        
          r
          
            i
          
        
        +
        γ
        
          s
          
            ′
          
        
        
          ∑
          
            
              s
              
                ′
              
            
          
        
        P
        (
        
          s
          
            ′
          
        
        |
        s
        ,
        a
        )
        
          V
          
            i
          
        
        
          π
          
            ⋆
          
        
        (
        
          s
          
            ′
          
        
        )
      
    
    {\displaystyle Q_{i}\pi ^{\star }(s,a)=r_{i}+\gamma s^{\prime }\sum _{s^{\prime }}P(s^{\prime }\vert s,a)V_{i}\pi ^{\star }(s^{\prime })}
  

其中
  
    
      
        
          π
          
            ⋆
          
        
      
    
    {\displaystyle \pi ^{\star }}
  
为纳什均衡策略

离线RL(Offline RL)  强调重要性权重约束：  πmin​E(s,a)∼D​[β(a∣s)π(a∣s)​Qπ(s,a)]  防止分布偏移问题


## 参考文献
