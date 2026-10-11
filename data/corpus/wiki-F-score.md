# F-score

F值，亦被稱做F-measure，是量測二元分類和資訊檢索演算法的預測精確度時常用的一種指標。目前在辨識、偵測相關的演算法中經常會分別提到精確率（precision）和召回率（recall），F-score能同時考慮這兩個數值，平衡地反映這個演算法的精確度。


## 定義
### 一般式
          F
          
            β
          
        
        =
        (
        1
        +
        
          β
          
            2
          
        
        )
        ⋅
        
          
            
              
                p
                r
                e
                c
                i
                s
                i
                o
                n
              
              ⋅
              
                r
                e
                c
                a
                l
                l
              
            
            
              (
              
                β
                
                  2
                
              
              ⋅
              
                p
                r
                e
                c
                i
                s
                i
                o
                n
              
              )
              +
              
                r
                e
                c
                a
                l
                l
              
            
          
        
      
    
    {\displaystyle F_{\beta }=(1+\beta ^{2})\cdot {\frac {\mathrm {precision} \cdot \mathrm {recall} }{(\beta ^{2}\cdot \mathrm {precision} )+\mathrm {recall} }}}
  

  
    
      
        β
      
    
    {\displaystyle \beta }
  
是使用者自行定義的參數，由一般式可見F-score能同時考慮精確率和召回率。由於分子為精確率和召回率相乘，因此只要精確率或召回率趨近於0，F-score就會趨近於0，代表著這個演算法的精確度非常低。一個好的演算法，最好能夠平衡精確率和召回率，且儘量讓兩種指標都很高。当
  
    
      
        β
        →
        0
      
    
    {\displaystyle \beta \rightarrow 0}
  
时，F-score退化为精確率；当
  
    
      
        β
        →
        ∞
      
    
    {\displaystyle \beta \rightarrow \infty }
  
时，F-score退化为召回率。


### F1-score
一般而言，提到F-score且沒有特別的定義時，是指
  
    
      
        β
        =
        1
      
    
    {\displaystyle \beta =1}
  
時的F-score，也稱為F1-score，代表使用者同樣的注重精確率和召回率這兩個指標。其定義為精確率和召回率的調和平均：

  
    
      
        
          F
          
            1
          
        
        =
        
          
            2
            
              
                
                  r
                  e
                  c
                  a
                  l
                  l
                
                
                  −
                  1
                
              
              +
              
                
                  p
                  r
                  e
                  c
                  i
                  s
                  i
                  o
                  n
                
                
                  −
                  1
                
              
            
          
        
        =
        2
        
          
            
              
                p
                r
                e
                c
                i
                s
                i
                o
                n
              
              ⋅
              
                r
                e
                c
                a
                l
                l
              
            
            
              
                p
                r
                e
                c
                i
                s
                i
                o
                n
              
              +
              
                r
                e
                c
                a
                l
                l
              
            
          
        
      
    
    {\displaystyle F_{1}={\frac {2}{\mathrm {recall} ^{-1}+\mathrm {precision} ^{-1}}}=2{\frac {\mathrm {precision} \cdot \mathrm {recall} }{\mathrm {precision} +\mathrm {recall} }}}
  

F-score最理想的數值是趨近於1，做法是讓精確率和召回率都有很高的值。若兩者皆為1，使得
  
    
      
        2
        ⋅
        
          
            1
            2
          
        
        =
        1
      
    
    {\displaystyle 2\cdot {\frac {1}{2}}=1}
  
，則F-score = 1（100%），代表該算法有著最佳的精確度。


## F-score的組成元素
### TP, FN, FP, TN
前面的true/false修饰后面的positive/negative，后面的positive/negative是我们的方法的判断。
TP（true positive，真阳性）：我们的方法判断为真，且这个判断是对的。即事實上為真，而且被我們的方法判斷為真的情形。
FN（false negative，假阴性）：我们的方法判断为不真，且这个判断是错的。即事實上為真，卻被我們的方法判斷為不真的情形。
FP（false positive，假阳性）：我们的方法判断为真，且这个判断是错的。即事實上不為真，卻被我們的方法誤判為真的情形。
TN（true negative，真阴性）：我们的方法判断为不真，且这个判断是对的。即事實上不為真，而且被我們的方法判斷成不為真的情形。
以抓犯人為例，TP是有罪而且被抓到的情形，FN是有罪但沒被抓到的情形，FP是無罪但被誤抓的情形，TN是無罪且未被誤逮的情形


### 精確率和召回率
        p
        r
        e
        c
        i
        s
        i
        o
        n
        =
        
          
            
              T
              P
            
            
              T
              P
              +
              F
              P
            
          
        
        =
        +
        P
        
        
      
    
    {\displaystyle precision={\frac {TP}{TP+FP}}=+P\;\;}
  
 (positive prediction rate)
精確率的分母為兩種判斷為真的情形的總和（范恩圖中完整綠色的部份）

解釋：當辨識結果為FP的代價很高時，F-score應該著重此指標，亦即精確率要很高。
例子：辨識電郵信箱里的垃圾郵件時，如果某封被誤判成垃圾郵件（即FP）時，使用者可能就此錯過重要的通知。

  
    
      
        r
        e
        c
        a
        l
        l
        =
        
          
            
              T
              P
            
            
              T
              P
              +
              F
              N
            
          
        
      
    
    {\displaystyle recall={\frac {TP}{TP+FN}}}
  

召回率的分母為事實上為真的情形的總和（范恩圖中完整紫色的部份）

解釋：當辨識結果為FN的代價很高時，F-score應該著重此指標，亦即召回率要很高。
舉例：一個傳染病診斷辨識系統中，如果某個傳染病患者被誤判成陰性（即FN），當地的社區的居民就落入被傳染的高風險之中。
舉例：真正犯罪的人當中，有多少比例的罪犯被抓到。或，一張照片當中，有多少人臉被偵測到。


#### 精確率和召回率的異同
它們的分子皆為TP。
F-score的精確率和召回率之間存在著權衡的關係，可通過 β 調整更重視的部份。

以警察抓犯人的故事為例：
一位警察很厲害，抓了很多犯人，但是這些犯人當中，只有少部分真正有罪，其他都是被冤枉的。

召回率高，因為該抓與不該抓的犯人都被抓到了。
精確率 低，因為很多都是沒犯罪的人。
「寧可錯抓一百，也不可放過一個」
召回率高，但精確率低。

一個警察非常嚴謹，只逮捕真正有犯罪的人，不抓實在是沒辦法肯定的犯人。

precision 高，因為通常被抓到人的都是有罪的。
召回率低，因為不小心放掉一大群犯人。
「寧可錯放一百，也不可冤枉一個」
精確率高，但召回率低。


## 應用
F-score經常用於評估資訊檢索的結果，如：

圖像檢索
機器學習模型


## 性质
F-score 等價于取回物品集和相关物品集的Dice系数


## 參考
國立台灣大學電信工程學研究所丁建均教授. 高等數位訊號處理.   [2020-07-01]. （原始内容存档于2020-05-08）. 
F度量 （F-measure）. 國家教育研究院雙語詞彙資料庫.   [2020-07-01]. （原始内容存档于2020-07-03）.
