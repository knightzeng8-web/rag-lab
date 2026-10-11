# Claude (语言模型)

Claude是人工智能安全与研究公司Anthropic开发的生成式人工智能聊天机器人，亦指支撑该聊天机器人的一系列大型语言模型。Claude能够处理文本与图像输入，可用于回答问题、总结文档、撰写长篇文本与编写程序代码等任务。用户可通过网页浏览器使用Claude，亦可使用其桌面与移动应用程序。Claude训练方法的一个显著特点在于，Anthropic以一组明文写定的原则（即“宪法”）将人类的价值观与伦理准则纳入模型的训练框架。
按规模与成本，Claude模型分为Haiku、Sonnet与Opus三个层级；2026年起，Anthropic在Opus之上增设能力更强、成本也更高的Mythos级。截至2026年9月，各层级的最新模型分别为Claude Haiku 5.5、Claude Sonnet 5.5、Claude Opus 5.5与Claude Fable 5.1；与Fable 5.1共享同一底层模型、但安全限制较少的Claude Mythos 5.1，则仅向经审核的机构开放。
除聊天应用外，Anthropic还通过Claude API、代理式编程工具Claude Code及面向知识工作的Claude Cowork等产品提供Claude的能力。


## 概述
### 名称
据《纽约时报》报道，Claude之名系向美国数学家、信息论奠基人克劳德·香农（Claude Shannon）致敬。


### 宪法式人工智能
宪法式人工智能（Constitutional AI）是Anthropic于2022年提出的训练方法。该方法以一组书面原则（即“宪法”）为依据，先由模型对自身输出进行批评与修订，再以AI依据这些原则给出的偏好反馈替代部分人工标注，从而训练出无害而有用的模型。2023年，Anthropic公开了Claude所用的宪法，其原则部分取材于《世界人权宣言》等文献。
2026年1月，Anthropic发布新版Claude宪法。与旧版以独立原则罗列为主的写法不同，新版着重阐明各项要求背后的理由，以期模型在陌生情境中亦能作出良好判断。新版将“广泛安全”“广泛合乎伦理”“遵守Anthropic的准则”与“真正有帮助”列为核心属性，并规定在相互冲突时大体按此顺序取舍；对于风险极高的行为，另设若干不可逾越的“硬性约束”。全文以知识共享CC0协议公开。


## 版本历史
### Claude 1与Claude 2
2023年3月14日，Anthropic发布Claude及其更轻量、响应更快的版本Claude Instant，通过聊天界面与API向用户提供。同年7月11日发布的Claude 2在编程、数学与推理方面有所提升，并通过API以及在美国和英国公开测试的claude.ai网站提供。8月9日，Anthropic发布Claude Instant 1.2。11月21日发布的Claude 2.1将上下文窗口扩展至20万token，并降低了幻觉率。


### Claude 3与Claude 3.5
2024年3月4日，Anthropic发布Claude 3系列，包括Haiku、Sonnet与Opus三个型号，三者均支持图像输入。同年6月20日，Anthropic发布Claude 3.5 Sonnet，并以预览形式推出Artifacts功能。10月22日，Anthropic发布升级版Claude 3.5 Sonnet与Claude 3.5 Haiku，并在API中以公测形式推出“计算机使用”功能。


### Claude 3.7
2025年2月24日，Anthropic发布Claude 3.7 Sonnet，同时以研究预览形式推出面向开发者的Claude Code。Anthropic称，Claude 3.7 Sonnet是其首个混合推理模型，既可即时作答，也可进行较长的逐步推理；API用户还可控制模型用于推理的计算量。该模型于2025年10月28日弃用，2026年2月19日退役。


### Claude 4
2025年5月22日，Anthropic发布Claude 4系列，包括Claude Opus 4与Claude Sonnet 4。Anthropic称，Opus 4面向复杂、长时间运行的任务与代理工作流；Sonnet 4则是Claude 3.7 Sonnet的升级版本，着重改进代码生成、推理与指令遵循能力。同日，Claude Code正式推出。两款模型于2026年4月14日弃用，6月15日退役；Anthropic建议分别迁移至Claude Opus 4.8与Claude Sonnet 4.6。


### Claude 4.1
2025年8月5日，Anthropic发布Claude Opus 4.1，称其在代理任务、真实世界的代码编写与推理方面较Claude Opus 4有所改进。该模型面向Claude付费用户、Claude Code、Claude API、Amazon Bedrock与Google Cloud Vertex AI提供，定价与Claude Opus 4相同。该模型于2026年6月5日弃用，8月5日退役；Anthropic建议迁移至Claude Opus 4.8。


### Claude 4.5
2025年9月29日，Anthropic发布Claude Sonnet 4.5，称其在代码编写、代理构建、计算机使用、推理与数学方面较前代有所提升，定价与Claude Sonnet 4相同。10月15日，Anthropic发布Claude Haiku 4.5，将其定位为速度更快、成本更低的小型模型，编程能力接近Claude Sonnet 4。11月24日，Anthropic发布Claude Opus 4.5，其面向编程、代理任务、计算机使用、深度研究以及处理幻灯片与电子表格等日常工作；API同时新增effort参数，供开发者权衡回答质量与token消耗。


### Claude 4.6
2026年2月5日，Anthropic发布Claude Opus 4.6，称其在代码编写、代码审查、调试与长时间代理任务方面较前代有所改进，并首次在Opus系列中以测试版形式提供100万token的上下文窗口。2月17日，Anthropic发布Claude Sonnet 4.6，升级了代码编写、计算机使用、长上下文推理、代理规划、知识工作与设计等方面的能力，同样以测试版形式提供100万token的上下文窗口。


### Claude Mythos Preview
2026年4月7日，Anthropic发布Claude Mythos Preview，并同时启动网络安全计划Project Glasswing。Anthropic认为，该模型发现与利用软件漏洞的能力极强，广泛发布可能带来严重的安全风险，因此不向公众开放，仅供亚马逊云科技、苹果、博通、思科、CrowdStrike、谷歌、摩根大通、Linux基金会、微软、英伟达与Palo Alto Networks等首批合作伙伴，以及另外40余家维护关键软件的机构，用于防御性安全工作。同年6月2日，Anthropic宣布将Mythos Preview的访问权限扩展至约150家新机构，使Project Glasswing的合作机构增至约200家，分布于超过15个国家，并新增此前覆盖不足的电力、供水、医疗、通信与硬件等行业。该模型于2026年6月9日弃用。


### Claude 4.7
2026年4月16日，Anthropic发布Claude Opus 4.7，称其较Claude Opus 4.6在高级软件工程、复杂长任务执行、指令遵循与视觉能力方面有所提升。该模型通过Claude产品、Claude API、Amazon Bedrock、Google Cloud Vertex AI与Microsoft Foundry提供，定价与Claude Opus 4.6相同。


### Claude 4.8
2026年5月28日，Anthropic发布Claude Opus 4.8，称其在Claude Opus 4.7的基础上于各项基准测试中均有改进，定价维持不变。随该模型一同推出的更新包括：Claude.ai与Claude Cowork新增面向所有方案用户的effort控制，供用户调节Claude投入的推理程度；Claude Code以研究预览形式推出“动态工作流”功能，可先规划任务，再在单一会话中并行运行数百个子代理；Opus 4.8的快速模式速度约为常规模式的2.5倍，价格则降至以往模型快速模式的三分之一。


### Claude 5
2026年6月9日，Anthropic发布Claude Fable 5与Claude Mythos 5。Fable 5是Anthropic首款面向公众提供的Mythos级模型。两者共享同一底层模型，区别仅在于安全防护：Fable 5的分类器一旦识别出涉及网络安全、化学与生物或模型蒸馏的请求，便将其转交Claude Opus 4.8处理；Mythos 5则在部分领域放宽了这些限制，仅通过Project Glasswing向经审核的网络防御机构与关键基础设施提供者开放。两款模型的API定价均为每百万输入token 10美元、每百万输出token 50美元。同年6月12日至30日，两款模型因美国出口管制暂停服务（参见下文）。
2026年6月30日，Anthropic发布Claude Sonnet 5，称其为迄今最具代理能力的Sonnet模型，在多项任务上的表现接近Claude Opus 4.8而价格更低。该模型发布后成为Claude免费版与Pro方案的默认模型，并通过Claude Code与Claude API提供。
2026年7月24日，Anthropic发布Claude Opus 5，称其在长时间运行的代理任务、编程与专业工作方面较前代有显著提升，能以Claude Fable 5一半的价格接近后者的前沿智能水平。该模型定价与Claude Opus 4.8相同；在Frontier-Bench与GDPval-AA等编程与知识工作评测中居于领先，但在网络安全任务上仍不及Mythos 5。Opus 5发布后成为Claude Max方案的默认模型。


### Claude 5.1
2026年9月1日，Anthropic发布Claude Fable 5.1与Claude Mythos 5.1，作为Fable 5与Mythos 5的后续版本。两者同样共享底层模型：Fable 5.1为正式开放的版本，Mythos 5.1则仅通过受限访问计划，向经审核的网络安全与生命科学机构开放。Fable 5.1可通过Claude API以及Amazon Web Services、Google Cloud与Microsoft Foundry使用；其输入与输出单价与Fable 5相同，缓存读取价格则由每百万token 1美元降至0.25美元，降幅为75%。Fable 5.1还引入名为“保留思考”（preserved thinking）的反蒸馏防护措施，以阻止API用户篡改Claude此前的上下文、借此提取其推理内容。
2026年10月7日，Claude宣布允許經過驗證的安全專業人士可以申請存取 Claude Mythos 5.1，這是Mythos模型首次對非大型機構開放。


### Claude 5.5
2026年9月22日，Anthropic发布Claude 5.5系列的首款模型Claude Opus 5.5。据Anthropic介绍，该模型在多数工作上的表现与Claude Fable 5.1相当，运行成本比Claude Opus 5低40%；其API定价为每百万输入token 4美元、每百万输出token 20美元（Opus 5为5美元与25美元），输出速度亦较Opus 5提高30%以上。Opus 5.5是Anthropic呼吁放缓前沿人工智能发展节奏之后发布的首款模型；发布前曾交由METR等外部机构测试，并在Anthropic最全面的对齐评估即自动化行为审计中取得该公司迄今最佳成绩。
由于Opus 5.5在生物与网络安全领域的能力与Mythos 5.1相当，Anthropic为其配备了与Fable 5.1相近的防护措施：部分网络安全请求转交Claude Opus 4.8处理，部分生物学请求则转交Claude Opus 5处理。Anthropic亦承认，该模型有时似乎能察觉自身正在接受评估，这使得仅凭基准测试推断其实际行为更为困难。Opus 5.5可通过Claude应用、Claude API以及Amazon Web Services、Google Cloud与Microsoft Foundry使用；随其发布，Anthropic提高了Pro、Max、Team及按席位计费的Enterprise方案的五小时用量上限。
2026年9月28日，Anthropic发布Claude 5.5系列的第二款模型Claude Sonnet 5.5，将其定位为比Opus 5.5更快、成本更低的补充，尤其擅长范围明确的日常任务、修复程序错误以及制作文档、幻灯片与电子表格。该模型定价与Claude Sonnet 5相同，即每百万输入token 2美元、每百万输出token 10美元，缓存读取为每百万token 0.20美元；Anthropic称其输出速度较Sonnet 5提高30%以上，且由于完成同样工作所需的token更少，每项任务的成本最多可降低30%。由于其网络安全能力与Claude Opus 5相当，Sonnet 5.5成为首款配备类似Anthropic最强模型所用网络安全防护与回退机制的Sonnet模型，较高风险的网络安全任务将转交Claude Sonnet 5处理；它也是首款配备阻止推理提取的安全分类器的Sonnet模型，“保留思考”机制亦随之扩展；生物学方面的防护则与Sonnet 5相同。Sonnet 5.5可通过Claude API、Amazon Bedrock、Google Cloud与Microsoft Foundry使用。
2026年10月7日，Anthropic發布Claude Haiku 5.5。官方稱其為該公司迄今最快速、功能最強的小型模型，適用於摘要、子代理與瀏覽器操作等高用量工作。Haiku 5.5的平均運行成本較Haiku 4.5低約75%；提示在10萬token以內時，每百萬輸入與輸出token分別為0.10美元與0.50美元（降幅90%），超過10萬token則為0.50美元與2.50美元（降幅50%）。


## 产品
### Claude应用
Anthropic提供基于Claude模型的聊天服务Claude.ai，涵盖网页端以及适用于Windows、macOS、iOS与Android的应用程序。个人用户可选择免费方案或Pro、Max付费订阅，机构用户则可选择Team与Enterprise方案。


#### Artifacts
2024年6月，Anthropic在发布Claude 3.5 Sonnet的同时，以预览形式推出Artifacts功能：Claude生成的代码、文档等内容在对话旁的独立窗口中呈现，供用户查看并迭代修改。同年8月，该功能向所有Claude.ai用户正式开放。


#### 网络搜索
2025年3月，Anthropic为Claude加入网络搜索功能，率先以预览形式向美国付费用户提供，同年5月扩展至全球。


### Claude Code
Claude Code是Anthropic开发的代理式编程工具，最初以命令行工具形式提供。2025年2月，Anthropic随Claude 3.7 Sonnet以研究预览形式发布该工具，同年5月随Claude 4系列正式推出。用户可通过Claude订阅或Claude API按用量计费的方式使用。


### Claude Cowork
2026年1月12日，Anthropic以研究预览形式推出Claude Cowork，最初仅向macOS桌面应用中的Max订阅用户开放。Cowork与Claude Code构建于相同的技术基础之上，但面向编程以外的工作：用户授权Claude访问电脑上的指定文件夹后，即可委托其完成涉及大量文件的多步骤任务。


### Chrome版Claude
2025年8月26日，Anthropic以研究预览形式推出Chrome浏览器扩展，使Claude能够在浏览器中阅读网页、点击页面元素并在网站间导航。


### Claude教师版
2026年7月14日，Anthropic推出Claude教师版（Claude for Teachers），向经认证的美国K-12教育工作者免费提供Claude的高级功能、教学技能库，以及与美国50个州学术标准相对应的循证课程资源。于2027年6月30日前注册的教师可获得一年免费使用期。


### 计算机使用
2024年10月，Anthropic随升级版Claude 3.5 Sonnet在API中以公测形式推出“计算机使用”功能，使Claude能够通过解读屏幕内容并模拟键盘与鼠标输入来操作计算机。


### 模型上下文协议
2024年11月，Anthropic发布模型上下文协议（Model Context Protocol，MCP），作为连接AI应用与外部工具及数据源的开放标准。2025年12月9日，Anthropic将MCP捐赠给Linux基金会旗下新成立的智能体人工智能基金会（Agentic AI Foundation），该基金会由Anthropic、Block与OpenAI共同创立。Anthropic称，捐赠时已有逾1万个活跃的公共MCP服务器，ChatGPT、Gemini、Microsoft Copilot等产品均已采用该协议。


## 争议
### 网络爬虫
2024年7月，维修信息网站iFixit首席执行官凯尔·维恩斯（Kyle Wiens）称，Anthropic的网络爬虫ClaudeBot在24小时内访问该网站服务器约100万次，违反了网站禁止将其内容用于训练AI的使用条款；自由职业平台Freelancer亦指该爬虫无视其robots.txt协议，在4小时内访问其网站约350万次。维恩斯其后表示，iFixit将ClaudeBot列入robots.txt后，爬取即告停止。


### 图书版权诉讼
2024年8月19日，作家安德烈娅·巴茨（Andrea Bartz）等三人对Anthropic提起集体诉讼（Bartz v. Anthropic），指控其从LibGen、PiLiMi等盗版“影子图书馆”复制受版权保护的图书，用于训练模型。2025年6月23日，联邦地区法官威廉·阿尔苏普（William Alsup）裁定，以合法取得的图书训练模型属于合理使用，但涉及盗版图书的主张不适用合理使用，应进入庭审。同年9月，双方达成金额至少15亿美元的和解：Anthropic须就约48万部作品按每部约3,000美元赔付，并销毁下载的盗版文件。阿尔苏普于2025年9月25日初步批准该和解；2026年7月20日，法官阿拉塞利·马丁内斯-奥尔金（Araceli Martínez-Olguín）最终批准，这是美国历史上金额最大的版权集体诉讼和解。


### 与美国国防部的纠纷
Anthropic通过与数据分析公司Palantir的合作，使Claude得以在美国国防部的机密环境中使用。2026年2月13日，《华尔街日报》援引知情人士报道称，美军同年1月初抓捕委内瑞拉前总统尼古拉斯·马杜罗的行动中使用了Claude。Anthropic回应称，无法就Claude是否被用于任何具体行动置评，但任何使用均须遵守其使用政策。
其后，双方围绕合同条款陷入僵局：Anthropic要求国防部作出有限承诺，保证不将Claude用于对美国民众的大规模监控或完全自主武器；国防部则要求Anthropic允许其不受限制地将Claude用于军事用途。2026年2月27日，特朗普总统在社交平台Truth Social上指示所有联邦机构立即停止使用Anthropic的技术，但给予国防部六个月的过渡期；国防部长皮特·赫格塞思随即宣布将Anthropic认定为“国家安全供应链风险”。国防部于3月初正式作出该认定，要求国防承包商证明其在与军方合作的工作中未使用Claude。据Axios报道，美国政府在对伊朗的军事行动中亦依赖Claude。
3月9日，Anthropic分别向加利福尼亚北区联邦地区法院与哥伦比亚特区巡回上诉法院提起诉讼，主张该认定系对其公开表达政策立场的惩罚，违反《美国宪法第一修正案》，且超越了法定授权。同月下旬，旧金山联邦法院在其中一案中发出初步禁令，禁止政府执行Claude使用禁令；4月8日，哥伦比亚特区巡回上诉法院则驳回了Anthropic暂缓执行该认定的请求。8月27日，联邦地区法官丽塔·林（Rita Lin）裁定，该认定构成违反第一修正案的非法报复，属“任意且反复无常”的行政行为，并剥夺了Anthropic依第五修正案应享有的正当程序，下令撤销该认定。
9月25日，哥伦比亚特区巡回上诉法院在另一案中以2比1裁定维持国防部的认定，允许国防部继续从其系统中移除Claude，并禁止承包商在为国防部开展的工作中使用Anthropic的产品；该判决并不禁止联邦政府整体使用Anthropic的产品。多数意见同时驳回了Anthropic有关言论自由与正当程序受到侵犯的主张；持异议的亨德森（Henderson）法官则认为，承包商事先公开执行使用限制，不属于法律所指可据以将其列入黑名单的供应链风险。Anthropic表示不同意该判决，正在考虑包括寻求进一步审查在内的各种选项。


### Max订阅集体诉讼
2026年6月14日，华盛顿特区居民卡尔·卡恩（Karl Kahn）向加利福尼亚北区联邦地区法院提起拟议集体诉讼，指控Anthropic在Claude Max订阅方案的用量上作出误导性宣传。起诉书称，Max 20x方案实际仅提供Pro方案6至8倍的用量，而非宣传中的20倍；Max 5x方案仅约为Pro方案的3.5倍，而非5倍；此外，月费200美元的Max 20x方案所标榜的“节省50%”亦属不实。


### Clawdbot商标争议
奥地利开发者彼得·施泰因贝格尔（Peter Steinberger）开发的开源个人AI助手项目最初名为Clawdbot。2026年1月，该项目迅速走红后，Anthropic以其名称与“Claude”读音相近为由提出商标方面的异议，项目遂更名为Moltbot，不久又改名为OpenClaw。2026年2月，施泰因贝格尔宣布加入OpenAI，OpenClaw则转入独立的开源基金会运营。


### Claude Code质量下降事件
2026年3月至4月，大量用户反映Claude Code的表现明显下降。4月2日，超威半导体（AMD）AI部门高级总监斯特拉·劳伦佐（Stella Laurenzo）在GitHub上发布基于6,852个会话文件的分析，认为Claude Code已退化到无法被信任执行复杂工程任务的程度。4月23日，Anthropic发布工程事后分析，将问题归因于3月至4月间先后上线的三项产品层变更：3月4日将默认推理投入由“高”下调至“中”；3月26日引入的缓存缺陷导致会话中的思考内容在每一轮都被清除；4月16日为控制输出篇幅而在系统提示中添加的长度限制。Anthropic表示，API与模型权重均未受影响，三项问题已于4月20日前全部修复，并为所有订阅用户重置了用量限额。


### 模型蒸馏指控
2026年2月23日，Anthropic指控中国人工智能公司深度求索（DeepSeek）、月之暗面（Moonshot AI）与稀宇科技（MiniMax）对Claude实施“工业级”蒸馏攻击，称三家公司通过约2.4万个欺诈账号与Claude进行了逾1,600万次交互，借以提取其能力来训练自家模型。
同年6月10日，Anthropic致函美国参议院银行委员会主席蒂姆·斯科特与首席少数党委员伊丽莎白·沃伦，称与阿里巴巴及其通义千问（Qwen）实验室有关联的操作者于4月22日至6月5日间，通过近2.5万个欺诈账号与Claude进行了逾2,880万次交互，主要针对Claude的软件工程与智能体推理能力，并称这是其迄今所知规模最大的蒸馏攻击。阿里巴巴未回应置评请求。


### Fable 5与Mythos 5出口管制
2026年6月12日，即Fable 5与Mythos 5发布三天后，美国政府援引国家安全授权对两款模型实施出口管制，要求Anthropic禁止任何外国国民（无论身处美国境内或境外，包括Anthropic的外籍员工）访问。Anthropic表示，由于该命令即时生效，而公司无法实时核验用户国籍，因此暂停了所有用户对两款模型的访问。据报道，此次管制源于亚马逊研究人员报告的一种可绕过Fable 5网络安全防护的越狱技术：该技术曾诱使模型识别软件漏洞，并在一例中生成了演示漏洞利用方式的代码。
6月26日，经美国政府批准，Anthropic先向部分美国机构恢复Mythos 5的访问。6月30日，美国商务部解除对两款模型的出口管制：Fable 5自7月1日起向全球用户恢复提供，Mythos 5则仍仅向获批的美国机构开放；Anthropic表示将继续与美国政府合作，通过Project Glasswing向更多国内外合作伙伴扩大Mythos 5的访问范围。商务部长霍华德·卢特尼克称，此前两周商务部与Anthropic密切合作，对Fable 5进行了分析与审批。


### 算力规模之争
2026年4月，据CNBC报道，OpenAI在致股东的备忘录中称Anthropic的算力增长“曲线明显更小”。备忘录估计，Anthropic的算力将于2026年底达到3至4吉瓦、2027年底达到7至8吉瓦，而OpenAI计划于2030年前达到30吉瓦。在此之前，Anthropic已于2025年10月宣布与Google签订至多可使用100万颗TPU、容量逾1吉瓦的云协议，11月又宣布与Fluidstack合作投资500亿美元兴建数据中心，并承诺采购300亿美元的Microsoft Azure算力；2026年4月，Anthropic再与Google及博通达成新的大规模TPU算力协议。


## 参见
大型语言模型列表
模型上下文协议


## 脚注
## 参考文献
