# 長短期記憶

长短期记忆（英語：Long Short-Term Memory，LSTM）是一种时间循环神经网络（RNN），论文首次发表于1997年。由于独特的设计结构，LSTM适合于处理和预测时间序列中间隔和延迟非常长的重要事件。
LSTM的表现通常比时间循环神经网络及隐马尔科夫模型（HMM）更好，比如用在不分段连续手写识别上。2009年，用LSTM构建的人工神经网络模型赢得过ICDAR手写识别比赛冠军。LSTM还普遍用于自主语音识别，2013年運用TIMIT自然演講資料庫達成17.7%錯誤率的紀錄。作为非线性模型，LSTM可作为复杂的非线性单元用于构造更大型深度神经网络。
通常情况，一个LSTM单元由细胞单元（cell）、输入门（input gate）、输出门（output gate）、遗忘门（forget gate）组成。


## 历史
1997年，賽普·霍克賴特和于尔根·施密德胡伯提出LSTM。版本包含了cells, input以及output gates。
2014年，Kyunghyun Cho et al.发明了门控循环单元（GRU）。
2016年，谷歌用LSTM进行谷歌翻译。 苹果公司、微软和亞馬遜公司也用LSTM生产产品，例如：iPhone、Amazon Alexa等。中国公司也正在用LSTM。


## 结构
LSTM是一種含有LSTM區塊（blocks）或其他的一種類神經網路，文獻或其他資料中LSTM區塊可能被描述成智慧型網路單元，因為它可以記憶不定時間長度的數值，區塊中有一個gate能夠決定input是否重要到能被記住及能不能被輸出output。
右圖底下是四個S函數單元，最左邊函數依情況可能成為區塊的input，右邊三個會經過gate決定input是否能傳入區塊，左邊第二個為input gate，如果這裡產出近似於零，將把這裡的值擋住，不會進到下一層。左邊第三個是forget gate，當這產生值近似於零，將把區塊裡記住的值忘掉。第四個也就是最右邊的input為output gate，他可以決定在區塊記憶中的input是否能輸出 。
LSTM有很多个版本，其中一个重要的版本是GRU（Gated Recurrent Unit），根据谷歌的测试表明，LSTM中最重要的是Forget gate，其次是Input gate，最次是Output gate。


## 方程
                  f
                  
                    t
                  
                
              
              
                
                =
                
                  σ
                  
                    g
                  
                
                (
                
                  W
                  
                    f
                  
                
                
                  x
                  
                    t
                  
                
                +
                
                  U
                  
                    f
                  
                
                
                  h
                  
                    t
                    −
                    1
                  
                
                +
                
                  b
                  
                    f
                  
                
                )
              
            
            
              
                
                  i
                  
                    t
                  
                
              
              
                
                =
                
                  σ
                  
                    g
                  
                
                (
                
                  W
                  
                    i
                  
                
                
                  x
                  
                    t
                  
                
                +
                
                  U
                  
                    i
                  
                
                
                  h
                  
                    t
                    −
                    1
                  
                
                +
                
                  b
                  
                    i
                  
                
                )
              
            
            
              
                
                  o
                  
                    t
                  
                
              
              
                
                =
                
                  σ
                  
                    g
                  
                
                (
                
                  W
                  
                    o
                  
                
                
                  x
                  
                    t
                  
                
                +
                
                  U
                  
                    o
                  
                
                
                  h
                  
                    t
                    −
                    1
                  
                
                +
                
                  b
                  
                    o
                  
                
                )
              
            
            
              
                
                  c
                  
                    t
                  
                
              
              
                
                =
                
                  f
                  
                    t
                  
                
                ∘
                
                  c
                  
                    t
                    −
                    1
                  
                
                +
                
                  i
                  
                    t
                  
                
                ∘
                
                  σ
                  
                    c
                  
                
                (
                
                  W
                  
                    c
                  
                
                
                  x
                  
                    t
                  
                
                +
                
                  U
                  
                    c
                  
                
                
                  h
                  
                    t
                    −
                    1
                  
                
                +
                
                  b
                  
                    c
                  
                
                )
              
            
            
              
                
                  h
                  
                    t
                  
                
              
              
                
                =
                
                  o
                  
                    t
                  
                
                ∘
                
                  σ
                  
                    h
                  
                
                (
                
                  c
                  
                    t
                  
                
                )
              
            
          
        
      
    
    {\displaystyle {\begin{aligned}f_{t}&=\sigma _{g}(W_{f}x_{t}+U_{f}h_{t-1}+b_{f})\\i_{t}&=\sigma _{g}(W_{i}x_{t}+U_{i}h_{t-1}+b_{i})\\o_{t}&=\sigma _{g}(W_{o}x_{t}+U_{o}h_{t-1}+b_{o})\\c_{t}&=f_{t}\circ c_{t-1}+i_{t}\circ \sigma _{c}(W_{c}x_{t}+U_{c}h_{t-1}+b_{c})\\h_{t}&=o_{t}\circ \sigma _{h}(c_{t})\end{aligned}}}
  


### 变量
          x
          
            t
          
        
        ∈
        
          
            R
          
          
            d
          
        
      
    
    {\displaystyle x_{t}\in \mathbb {R} ^{d}}
  
: LSTM的input（輸入）

  
    
      
        
          f
          
            t
          
        
        ∈
        
          
            R
          
          
            h
          
        
      
    
    {\displaystyle f_{t}\in \mathbb {R} ^{h}}
  
: forget gate（遺忘閥）

  
    
      
        
          i
          
            t
          
        
        ∈
        
          
            R
          
          
            h
          
        
      
    
    {\displaystyle i_{t}\in \mathbb {R} ^{h}}
  
: input gate（輸入閥）

  
    
      
        
          o
          
            t
          
        
        ∈
        
          
            R
          
          
            h
          
        
      
    
    {\displaystyle o_{t}\in \mathbb {R} ^{h}}
  
: output gate（輸出閥）

  
    
      
        
          h
          
            t
          
        
        ∈
        
          
            R
          
          
            h
          
        
      
    
    {\displaystyle h_{t}\in \mathbb {R} ^{h}}
  
: hidden state（隱藏狀態）

  
    
      
        
          c
          
            t
          
        
        ∈
        
          
            R
          
          
            h
          
        
      
    
    {\displaystyle c_{t}\in \mathbb {R} ^{h}}
  
: cell state（單元狀態）

  
    
      
        W
        ∈
        
          
            R
          
          
            h
            ×
            d
          
        
      
    
    {\displaystyle W\in \mathbb {R} ^{h\times d}}
  
、
  
    
      
        U
        ∈
        
          
            R
          
          
            h
            ×
            h
          
        
      
    
    {\displaystyle U\in \mathbb {R} ^{h\times h}}
  
、
  
    
      
        b
        ∈
        
          
            R
          
          
            h
          
        
      
    
    {\displaystyle b\in \mathbb {R} ^{h}}
  
: 訓練中的矩阵，网络学习计算元值


## 训练方法
為了最小化訓練誤差，梯度下降法（Gradient descent）如：應用時序性倒傳遞演算法，可用來依據錯誤修改每次的權重。梯度下降法在循環神經網路（RNN）中主要的問題初次在1991年發現，就是誤差梯度隨著事件間的時間長度成指數般的消失。當設置了LSTM 區塊時，誤差也隨著倒回計算，從output影響回input階段的每一個gate，直到這個數值被過濾掉。因此正常的倒循環類神經是一個有效訓練LSTM區塊記住長時間數值的方法。

Backpropagation through time、BPTT 


## 应用
机器控制
時間序列
语音识别
音乐
自然语言处理
手写识别
生物
飞机处理
自動駕駛汽車
自平衡滑行车
电脑游戏
动画
即時天氣預報（ConvLSTM）


## 参见
人工神经网络
深度学习
前额叶皮质基底节工作记忆（PBWM）
循环神经网络（RNN）
时间序列
Seq2Seq模型


## 参考
## 外部链接
理解LSTM网络 （页面存档备份，存于互联网档案馆），作者Christopher Olah，更新于2015年八月。
http://people.idsia.ch/~juergen/rnn.html （页面存档备份，存于互联网档案馆）
https://www.youtube.com/watch?v=cTqVhcrilrE&list=WL&index=51 （页面存档备份，存于互联网档案馆）
