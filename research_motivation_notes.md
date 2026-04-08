# Surgical VLP 论文动机与方法草稿

## 研究问题

近年来，弱监督 surgical video-language pretraining 在零样本手术理解中展现出越来越强的潜力。相比依赖大量人工标注的传统方案，这类方法能够利用讲解视频、自动生成的层级文本以及时间戳等弱监督信号，学习具有迁移能力的跨模态表征。然而，弱监督可扩展并不等于监督精确。对于一个视频-文本样本而言，文本在全局语义上可能是成立的，但真正支持该文本的视觉证据往往只存在于局部片段中，并不均匀分布在整个时间窗口内。

这一问题在手术场景中尤其明显。手术视频具有阶段转换细微、动作持续时间不均、视觉外观相似以及文本描述与画面轻微不同步等特点。现有方法通常默认一个 weakly paired clip-text sample 可以作为完整正样本参与对比学习，即把整段视频视为对文本均匀可信的监督来源。但这一假设过强：当文本只对应局部关键片段，而其余部分包含过渡帧、上下文帧甚至弱相关内容时，直接对整段视频做全局池化并与文本强行对齐，容易把无关视觉内容也当作正样本信号，从而污染表征学习过程，削弱模型对细粒度手术语义的捕捉能力。

基于这一观察，我们认为当前弱监督 surgical VLP 的关键问题之一，不只是数据规模或文本层级是否足够丰富，而是样本内部局部证据的不确定性尚未被显式建模。换句话说，问题不在于这个视频-文本 pair 是否完全错误，而在于：即使该 pair 在全局上是合理的，也并不意味着整段视频中的每个局部片段都应被同等地视为正样本。因此，在弱监督对比学习中，更合理的做法应当是从候选时间邻域中识别出真正支持文本的局部视觉证据，并降低不相关片段对训练目标的干扰。

## 方法直觉

在这一思路下，我们的方法直觉是：不再将原始时间戳对应的视频窗口直接作为唯一可信区域，而是对其进行局部扩展，在扩展后的时间邻域中采样多个候选片段；随后利用文本与各候选片段之间的相似度估计其匹配可信度，并据此调整训练中的对齐强度。这样的过程可以理解为一种面向弱监督视频文本预训练的局部证据建模：文本标签对应的不是整个视频窗口，而是窗口中的一个或若干关键片段。

这个视角与多示例学习（MIL）的思想是相通的，即一个弱标签对应的是一个候选片段集合，模型需要从中自动识别更可信的正证据，而不是默认所有实例都 equally positive。我们关注的不是 pair-level validity，而是 weak pair 内部的 local evidence uncertainty。

## 方法定位

- 方法目标不是重新构建数据集，也不是做完整的 temporal grounding。
- 方法目标是在现有弱监督 surgical VLP 框架下，更合理地利用 weak pair 内部的局部监督信号。
- 核心贡献应表述为：面向弱监督对比学习的局部可信度建模，而不是单纯的局部注意力增强。
- 方法与 MIL 的关系在于：一个文本 supervision 对应的是一个候选片段集合，模型需要从中识别真正支持文本的局部证据。

## 论文叙事链

1. 背景：弱监督 surgical VLP 很重要，数据规模和层级越来越强，但弱监督不等于精确对齐。
2. 问题：现有方法通常把 weak pair 当成整段均匀可信的正样本，但真实支持文本的证据常常只在局部片段里，且可能存在时间偏移。
3. 后果：直接把整段视频当成正样本，会把无关片段也纳入对比学习，污染跨模态表征。
4. 方法直觉：在扩展时间邻域中搜索更可信的局部视觉证据，并据此调整训练时的对齐强度。
5. 方法归类：这是 weakly supervised local alignment / MIL-style credibility modeling，而不是完整 temporal grounding。
6. 实验原则：在相同数据、相同 backbone、相同训练框架下，证明局部可信度建模优于直接全局对齐。

## 实验原则

- 必须优先构建 matched baseline：相同数据、相同 backbone、相同训练框架，只改变局部可信度建模部分。
- 不应把主要结论建立在与异构外部方法的直接数值比较上。
- 评价重点应放在 zero-shot surgical evaluation，包括 phase、action、tool、triplet 等细粒度任务。
- 需要证明模型提升来自更合理的弱监督使用方式，而不是来自更强 backbone 或不同训练细节。

## 性能目标与公平比较原则

为了在 zero-shot surgical evaluation 上达到具有竞争力的性能，使用更强的领域先验编码器是合理的选择。尤其是在当前 surgical VLP 已经存在较强基线的情况下，如果仅使用自然场景预训练的视觉与文本编码器，模型性能可能难以达到可与现有方法竞争的水平。因此，采用手术领域预训练的视觉 backbone 与手术领域文本 backbone，本身并不构成不公平比较。

然而，更强的编码器不能替代方法贡献的验证。论文需要同时满足两个目标：一是获得足够强的最终性能，与现有 SOTA 进行比较；二是证明性能提升并非仅仅来自 backbone，而是来自所提出的局部可信度建模方法。因此，实验设计应明确区分“方法验证”和“整体性能比较”两个层次。

第一层是方法验证。在这一层中，应固定预训练数据、视觉 backbone、文本 backbone、训练策略和评估协议，只比较标准对比学习基线与所提出的方法。该层的目标是回答：在完全相同的骨干网络和训练配置下，显式建模 weak pair 内部的局部证据不确定性，是否能够稳定提升零样本迁移性能。

第二层是整体性能比较。在这一层中，可以采用性能最强的配置，例如手术领域预训练的视觉 backbone 与手术文本编码器结合所提出的方法，并与现有 SOTA 进行比较。该层的目标是回答：在实际最优配置下，所提出方法能否在 phase、action、tool、triplet 等下游任务上取得更强的 zero-shot surgical performance。

因此，论文的核心叙事不应是“为了公平只能使用较弱的通用预训练骨干”，而应是“在合理使用强领域先验的同时，通过 matched baseline 证明方法本身有效”。这意味着最终论文至少需要包含两类关键结果：其一，在相同 backbone 下，所提出方法优于标准 hard-pair / uniform alignment 基线；其二，在最强配置下，整体系统达到或超过现有 SOTA。

## 关键表述

- 我们关注的不是 pair-level validity，而是 weak pair 内部的 local evidence uncertainty。
- weak pair 在全局语义上可以成立，但其局部视觉证据并不均匀，且可能发生轻微时间偏移。
- 现有方法通常将整段视频视为均匀可信的正样本，而我们希望识别并强调真正支持文本的局部证据。
- 我们的方法属于 weakly supervised local alignment / MIL-style credibility modeling。

## 后续待补内容

- Introduction 正式英文学术写法
- Method 数学定义与损失函数
- Baseline 设计与公平比较矩阵
- Ablation 设计
- Contribution wording 与标题候选

## 可参考的现有论文写法

在当前阶段，方法本身仍待最终确定，因此更适合先参考已有工作的“方法叙事结构”，而不是过早写死具体公式或模块。下面列出几篇最值得参考的论文，以及各自最适合借鉴的部分。

### 1. SurgVLP：最适合参考基础方法骨架

SurgVLP 的优点在于结构非常清楚，适合作为最基本的 `Method` 模板。它的方法部分先定义视频-文本对的构造方式，再给出双编码器架构，随后说明视觉编码器、文本编码器、对比学习目标以及下游零样本适配方式。它的优势不是方法复杂，而是“问题定义 -> 数据构造 -> 模型结构 -> 损失函数 -> 下游使用方式”这一条线很完整。

对你最有参考价值的部分：

- 如何先把 weakly supervised video-text pair 的训练设置正式化；
- 如何写一个干净的 dual-encoder baseline；
- 如何在方法部分明确视觉编码器、文本编码器、视频采样与池化方式；
- 如何把 zero-shot downstream evaluation 与预训练目标自然衔接起来。

当前你的论文如果还没有开始正式方法设计，最适合先参考 SurgVLP 的整体骨架，再在其上替换成你自己的局部可信度建模思路。

### 2. HecVL：最适合参考“问题很小但讲得很稳”的写法

HecVL 的方法创新幅度并不靠复杂模块取胜，而是把问题收束得非常清楚：现有方法只用 clip-level text，而它要做 hierarchical video-language pretraining。它的方法部分也很工整，先定义 hierarchical pairs，再定义 fine-to-coarse learning strategy，然后是 loss 和 training pipeline。整篇论文最值得学的是：它没有把贡献说得过大，而是把“为什么需要多层级监督”和“为什么单 embedding space 不够”讲得很集中。

对你最有参考价值的部分：

- 如何把一个相对聚焦的方法讲成完整方法论；
- 如何在 `Method` 中先讲 supervision structure，再讲 learning strategy；
- 如何把 fairness 讲清楚，例如它明确说明与 SurgVLP 使用相同 encoder 以保证可比性；
- 如何用一个很清楚的 baseline 对照来支撑方法增益。

如果你的方法最后也是围绕一个核心问题做“小而准”的改进，那么 HecVL 的写法非常值得学。

### 3. PeskaVLP：最适合参考“方法组合较多时如何组织”

PeskaVLP 的方法比较丰富，既有 hierarchical knowledge augmentation，也有 clip-level 与 phase/video-level 两类不同训练目标，还加入 visual self-supervision 和 procedure-aware temporal regularization。它最大的优点是分层写法很清楚：先讲 dataset and contrastive learning，再讲 textual augmentation，然后再讲不同层级下的 pretraining objective。即便方法组件比较多，读者也能知道每个模块是在解决什么问题。

对你最有参考价值的部分：

- 如果你后续的方法不止一个组件，如何避免方法部分写乱；
- 如何把“数据问题”和“模型问题”分开写；
- 如何把 clip-level objective 和 higher-level objective 分开定义；
- 如何让每个模块都明确对应一个具体痛点，而不是堆模块。

如果你后面的方法变成“局部时间扩展 + 候选片段打分 + loss reweighting + hierarchy regularization”这种多组件形式，PeskaVLP 的组织方式最值得参考。

### 4. SurgLaVi / SurgCLIP：最适合参考实验导向的简洁写法

SurgLaVi 的一个重要特点是：作者明确强调自己的核心贡献首先是数据，其次才是一个轻量的 CLIP-style base model。它的写法很适合你现在这个阶段，因为你目前最优先的任务也是先把基线跑稳，而不是急着写很复杂的方法。SurgLaVi 的叙事重点是：使用标准且轻量的 base model，证明更好的数据和更规范的评测可以带来更强 zero-shot transfer。

对你最有参考价值的部分：

- 如何在基线阶段不把方法写得过重；
- 如何把重点放在 evaluation protocol 和 benchmark coverage 上；
- 如何用 phase / step / action / tool / triplet 的多任务评测去证明表示能力；
- 如何在论文里区分“base model”与“真正方法贡献”。

这篇文章尤其适合你用来参考 baseline 阶段的写法：先把 strongest reproducible baseline 讲清楚，再考虑方法创新。

### 5. VidLPRO：最适合参考“当你需要强调训练目标设计”时的写法

VidLPRO 的重点是：它认为现有 surgical VLP 过度依赖单一 contrastive learning，因此加入了 VTC、VTM、MLM 等多个预训练目标。它的方法部分适合参考的地方在于：如何把“为什么原始目标不够”讲清楚，并自然过渡到“因此我们引入额外训练目标”。如果你未来的方法最终不是纯局部重加权，而是会改动 loss 结构，那么 VidLPRO 的这种写法很值得借鉴。

对你最有参考价值的部分：

- 如何从训练目标不足引出方法；
- 如何把多个 loss 的角色区分清楚；
- 如何让方法动机直接落到 objective design 上；
- 如何写“我们的目标不是换 backbone，而是重新定义预训练信号”。

### 6. 弱监督/MIL相关文章：最适合参考问题定义，不一定直接参考实现

在手术领域之外，一些 instructional video 的弱监督对齐工作更适合拿来支撑你的问题定义。例如 DWSA 这类工作把视觉序列和语言序列看成 weakly aligned structure；而 “Look at What I’m Doing” 这类工作则强调 narrated video 中局部证据并非均匀分布，而需要通过 cross-modal mechanism 去识别关键证据。

对你最有参考价值的部分：

- 如何把“局部对齐不精确”写成一个正式研究问题；
- 如何把 weak supervision 和 local evidence selection 联系起来；
- 如何说明你的工作不是完整监督的 grounding，而是针对弱监督信号的更合理建模；
- 如何为 MIL-style 的局部可信度建模提供概念上的支撑。

## 当前最建议采用的方法写作骨架

在你还没有最终确定方法细节之前，最稳的写法不是直接写复杂公式，而是先搭一个足够 solid 的方法骨架。当前最建议采用的骨架如下：

1. Problem Setup
定义 weakly supervised surgical video-language pretraining 的输入形式，以及你的研究关注的是 weak pair 内部的 local evidence uncertainty。

2. Baseline Pretraining Framework
把当前采用的 dual-encoder baseline 先写清楚，包括视频输入、文本输入、编码器、投影层、对比学习目标。

3. Local Evidence Uncertainty Motivation
说明为什么原始时间窗口不应被视为均匀可信监督，为什么需要在局部时间邻域中寻找更可信的视觉证据。

4. Proposed Local Evidence Modeling
在这一节再写你最终的方法细节。即使具体模块仍待定，也可以先保留这个小节的位置。

5. Hierarchical Supervision Usage
单独说明 fine / mid / coarse level 在你的框架里分别扮演什么角色，而不是把它们混成并列文本输入。

6. Training Objective
最后统一给出 loss 设计，说明 baseline loss 与你的改进项之间的关系。

这种结构基本上是：

- 骨架参考 SurgVLP
- 监督结构参考 HecVL
- 多模块组织参考 PeskaVLP
- 实验导向与 baseline 清晰度参考 SurgLaVi
- 若后续涉及 loss redesign，则参考 VidLPRO

## 当前阶段不建议过早写死的部分

- 不建议现在就把最终 loss 公式写得很细，因为方法方向仍可能调整。
- 不建议现在就承诺具体模块名称或复杂结构，否则后面实验不稳定时会被自己绑住。
- 不建议把方法写成完整 temporal grounding，因为当前更合适的定位仍是 local evidence credibility modeling。
- 不建议在方法部分一开始就强调 novelty module，而应先把 baseline 和问题设定写清楚。

## 可改进方向与可缝合模块

当前这篇工作的主线已经比较清楚：不是做更大的数据集，也不是重复层级监督本身，而是研究 weak pair 内部的 local evidence uncertainty 应该如何被建模。因此，后续改进方向不应继续沿着 “更多 hierarchy” 或 “更强 backbone” 本身做文章，而应该聚焦在“如何更可靠地选择和利用局部正证据”。

### 一、最值得做的主方向：从 hard positive 改成 bag-level positive

这是当前最适合你论文主线的改进方向，也是最自然能和现有工作区分开的方向。

现有 surgical VLP 主线工作已经分别覆盖了：

- SurgVLP：用弱监督视频文本对做 contrastive learning；
- HecVL：引入 hierarchical video-text pairs 和 fine-to-coarse contrastive framework；
- PeskaVLP：引入 hierarchical knowledge augmentation、hard negatives 和 DTW-based alignment；
- SurgLaVi：通过更大、更规范的层级数据和标准 base model 提升 zero-shot transfer；
- VidLPRO：指出单一 contrastive objective 不足，并增加 VTM / MLM 等目标。

因此，如果你的方法仍然只是“在单个 pair 上做普通 clip-text contrastive”，或者只是“再多加一层 hierarchy loss”，创新空间已经很小。最值得发展的方向是：把一个 weakly paired sample 看成一个 candidate local evidence bag，而不是一个单一、均匀可信的正样本。

### 二、最推荐直接缝进来的三个核心模块

#### 1. 多候选局部片段 + MIL-style bag supervision

最核心的一步，是把原本一个时间窗对应一个 pooled clip embedding 的训练方式，改成：

- 在原始时间戳附近扩展局部窗口；
- 采样多个候选局部片段或帧；
- 文本对应的是一个候选 bag，而不是 bag 中所有实例都 equally positive；
- 最终损失作用于 bag-level positive，而不是单一 pooled clip。

这部分最值得参考的是 Weakly-Supervised Temporal Article Grounding (WSAG / DualMIL)，因为它明确指出两个不现实的假设：并非所有文本都能被 ground，且文本可能处在不同 semantic scale；其方法核心是 two-level MIL loss 和 sentence-level constraints。这个思想和你的问题高度兼容，只是你要把它从 article/video grounding 改写成 surgical weak pair 内部的 local evidence bag learning。

可借鉴点：

- 正样本不再是单一 clip，而是一个局部候选集合；
- 用 soft bag 或 top-k bag 的方式定义 positive similarity；
- higher-level text 可以作为 bag-level semantic constraint，而不是直接等价于 fine-level caption。

#### 2. 多路径 pseudo matching / confidence scoring

如果你只做局部窗口扩展，然后直接用文本相似度给片段打分，这个方法还是有点薄。最值得借鉴的是 ECCV 2024 的 Multi-Pathway Text-Video Alignment（MPTVA）思路。该工作明确指出 narrated instructional videos 里存在 irrelevant narrations 和 unreliable timestamps，因此不应只依赖单一路径配对，而应融合多条路径生成更可靠的 pseudo matching。

对你最值得缝进来的，不是它完整的 task setup，而是它的 scoring philosophy：

- 时间戳邻近性是一条路径；
- fine-level text 与 local video 的语义相似度是一条路径；
- mid/coarse-level text 与候选片段或候选局部聚合结果的一致性是一条路径；
- 不同路径融合后形成 local evidence confidence。

这会比“单一 cross-modal similarity 作为权重”更有说服力，也更符合你对 weak pair uncertainty 的动机。

#### 3. hierarchy 不再只是多加 supervision，而是做 role decomposition

HecVL 已经证明层级 supervision 是有价值的，但它的主线是 separate embedding spaces 和 fine-to-coarse contrastive learning。你不应简单重复这一点，而应把 hierarchy 的角色重新定义为：

- fine-level text：用于 local evidence selection；
- mid-level text：用于局部语义一致性约束；
- coarse/video-level text：用于全局 procedure context regularization。

这样你借鉴了 HecVL 的 hierarchy 价值，同时避免直接和 HecVL 撞车。你的创新点就会更清楚：不是“我们也有层级”，而是“我们把层级标注用于局部证据可信度建模”。

### 三、第二梯队值得缝的模块

#### 4. procedure-aware hard negatives

PeskaVLP 的一个强点是它不只是做普通 contrastive，还专门构造了 hard negatives，并引入了 procedure-aware 的约束。这个思路很适合缝到你的框架里，尤其是你后面做 zero-shot action / triplet 时。

最值得借鉴的方式不是把它整套搬过来，而是构造更难的 negative：

- 同 procedure 内的不同步骤；
- 同 phase 但不同 action；
- 同 tool 但不同 target；
- 同 center / 同 modality 的相似样本。

这样做的意义在于：你的方法不是只在“找最相关片段”，还要在细粒度语义上把容易混淆的候选片段区分开。这对 action 和 triplet 的提升更有帮助。

#### 5. procedure ordering / weak temporal order regularization

PeskaVLP 还用了 DTW-based loss 去理解 cross-modal procedural alignment。你当前不适合直接做完整 sequence alignment，但可以借它的轻量版思想：

- 相邻局部片段的高置信度位置不应随机跳跃；
- fine-level evidence 的选择应该与 mid-level step order 基本一致；
- 候选局部证据的分布可以加入一个轻量 temporal smoothness 或 ordering regularizer。

这会让你的方法从“局部打分”更进一步，变成“局部打分 + 弱时序约束”，但仍然不至于变成完整 grounding。

#### 6. 轻量 auxiliary matching objective

VidLPRO 的核心论点是：单一 VTC 不够，应该加入 VTM / MLM 等目标来增强 temporal dynamics 和 cross-modal fusion。对你来说，最值得借鉴的是这个判断，而不是把完整的 fusion-heavy 结构搬进来。

更适合你的缝法是：

- 保留 CLIP-like dual encoder 主体；
- 在 top-k 高置信局部片段上增加一个轻量 VTM-style auxiliary loss；
- 只让模型判断 “这个局部 evidence 是否与文本匹配”，而不是全面改成 heavy fusion 架构。

这样能增强局部可信度学习，但不会把方法主线从 CLIP-like pretraining 拉偏。

### 四、可选但更高风险的方向

#### 7. ordered weak alignment

DWSA 的核心思想是：视觉与语言序列 weakly aligned，且允许一部分元素 unmatched。这个思想与你的问题高度一致，但如果你现在直接上 sequence-level differentiable alignment，风险很高，因为：

- 实现复杂；
- 训练不稳定；
- 容易把论文从 surgical VLP 拉成 instructional video localization。

因此更建议把它当作 conceptual support，而不是当前阶段的主方法。

#### 8. LLM-based text denoising / summarization

MPTVA 和 PeskaVLP 都用了 LLM 去过滤、总结、增强文本。这个方向有价值，但当前不建议把它作为主贡献，因为：

- 会稀释你“局部可信度建模”的方法主线；
- reviewer 很容易把提升归因到更好的文本，而不是你的局部建模；
- 还可能引入额外实验变量。

如果后面要加，最多作为可选 preprocessing，而不是主方法部分。

### 五、当前最推荐的方法组合

如果现在要给出一版最值得实现的组合，最推荐的是：

1. 保留 CLIP-like dual encoder 主体；
2. 以 weak timestamp 为中心扩展局部时间邻域；
3. 采样多个候选局部片段，形成 candidate bag；
4. 用多路径 confidence scoring 估计每个局部片段与 fine-level text 的可信度；
5. 用 MIL-style bag loss 替代单一 hard positive alignment；
6. 用 mid/coarse-level text 做 hierarchical consistency regularization；
7. 可选加入 procedure-aware hard negatives；
8. 可选在 top-k 片段上增加轻量 VTM auxiliary loss。

这套组合的优点是：

- 主线非常统一，所有模块都围绕 local evidence uncertainty；
- 能自然兼容 SurgLaVi-Beta 的 full hierarchy；
- 不会和 HecVL、PeskaVLP、VidLPRO 完全撞车；
- 能解释为什么对 action / triplet 这种细粒度 zero-shot task 更有帮助。

### 六、优先级建议

如果按论文推进顺序排序，建议优先级如下：

第一优先级：

- matched baseline 跑稳；
- local candidate bag；
- MIL-style bag supervision；
- hierarchy role decomposition。

第二优先级：

- multi-pathway confidence scoring；
- procedure-aware hard negatives。

第三优先级：

- weak temporal order regularization；
- top-k VTM auxiliary objective。

第四优先级：

- LLM text preprocessing；
- full sequence alignment；
- 更重的 multimodal fusion 结构。

## 当前最推荐的方法蓝图

在当前设定下，backbone 固定为 CLIP-based 模型，因此方法设计的重点不应再放在 encoder 更换上，而应聚焦于“如何定义正样本、如何选择局部证据、如何让 hierarchy 发挥不同作用”。基于现有文献与当前论文目标，最推荐的方法不是单一模块，而是一套围绕 local evidence uncertainty 的有序组合。

### 主方案：最推荐的标准版本

#### 模块 1：Local Temporal Expansion

- 以原始 weak timestamp interval 为中心，向前后扩展一个局部时间邻域；
- 在扩展邻域中采样多个 candidate local clips 或 frames；
- 这些 candidate 构成当前 weak pair 的 local evidence bag。

这一部分的作用是把“单一 clip 正样本”改写为“局部候选证据集合”，是后续所有设计的前提。

#### 模块 2：Multi-Pathway Confidence Scoring

对每个 candidate local clip，不只使用单一路径相似度，而是融合以下几类信息得到 confidence score：

- temporal proximity：候选片段与原始时间戳的距离；
- fine-level similarity：fine-level text 与候选片段的语义相似度；
- mid-level consistency：候选片段与中层 step/phase 语义的一致性；
- coarse-level compatibility：候选片段是否与 procedure-level 语境冲突。

最终每个 candidate 都得到一个 soft confidence，而不是 binary positive/negative label。

#### 模块 3：MIL-style Bag Contrastive Learning

- 文本 supervision 不再只和单个 pooled clip embedding 对齐；
- 而是与一个 local evidence bag 对齐；
- bag 内部的候选片段根据 confidence 被 soft aggregation，或采用 top-k aggregation；
- contrastive loss 作用于 bag-level positive representation。

这是整套方法的核心。它直接把现有 hard positive assumption 改成 uncertainty-aware positive bag learning。

#### 模块 4：Hierarchy Role Decomposition

在 full hierarchy setting 下，不同层级文本不作为并列同权 supervision，而承担不同角色：

- fine-level text：负责 local evidence selection；
- mid-level text：负责 local semantic consistency regularization；
- coarse/video-level text：负责 global procedural context regularization。

这样 hierarchy 就不是简单“多加 supervision”，而是用于稳定 weak pair 内部的证据选择。

#### 模块 5：Procedure-Aware Hard Negatives

在保持 CLIP-style dual encoder 主体不变的前提下，增加更有针对性的 hard negatives：

- 同 procedure、不同 step；
- 同 phase、不同 action；
- 同 tool、不同 target；
- 语义相近但局部证据不匹配的片段。

这一模块的主要作用是提升细粒度区分能力，尤其服务于 zero-shot action / triplet recognition。

### 这一主方案最适合你的原因

- 保持 CLIP-like backbone 不变，符合你的当前设定；
- 主体改动集中在 supervision usage，而不是 encoder architecture；
- 与 SurgVLP / HecVL / PeskaVLP / SurgLaVi / VidLPRO 都有关联，但不会与任一篇完全重合；
- 能自然解释为什么对 phase 之外的 action、tool、triplet 更有帮助；
- 复杂度可控，适合在 baseline 跑稳之后逐步加模块。

## 精简版：最先可实现的版本

如果当前还没有把 SurgLaVi-Beta baseline 跑稳，不建议一开始就上完整主方案。最先可实现的版本应只保留三部分：

1. local temporal expansion；
2. fine-level similarity based soft weighting；
3. MIL-style bag contrastive loss。

这版的特点是：

- 不需要复杂的额外模块；
- 最容易和标准 baseline 做 matched comparison；
- 足够支撑一篇方法论文的最小核心主张。

如果这版已经能在 fixed backbone 下稳定优于 baseline，那么后续再加 hierarchy role decomposition 和 hard negatives 会更稳。

## 增强版：在主方案基础上的可选升级

在主方案稳定有效之后，再考虑以下增强项：

### 增强项 1：Top-k VTM Auxiliary Loss

- 在高 confidence 的 top-k local clips 上增加一个轻量 video-text matching loss；
- 目的不是替代 contrastive learning，而是增强局部匹配判别能力；
- 只建议作为 auxiliary loss，不建议改成重型 multimodal fusion。

### 增强项 2：Weak Temporal Order Regularization

- 对高 confidence 局部证据的时间分布加入轻量时序约束；
- 避免模型在候选 bag 中选择时间上完全跳跃的伪证据；
- 可与 mid-level step ordering 结合。

### 增强项 3：Confidence Refinement

- 用训练初期得到的 confidence 初始化；
- 随训练过程迭代更新 candidate confidence；
- 让伪标签从 heuristic score 逐步过渡到 model-assisted score。

这一项潜力很大，但实现风险也更高，应放在后期。

## 当前不推荐作为主方案的方向

- 不推荐把方法主线写成 full temporal grounding；
- 不推荐一开始就加入 LLM text denoising 作为核心模块；
- 不推荐使用 heavy fusion transformer 替代 CLIP-style dual encoder；
- 不推荐直接上 sequence-level differentiable alignment；
- 不推荐把 hierarchy 继续当作平行多目标 contrastive，而不改变其角色定义。

## 最终建议的实现顺序

如果按真正可落地的开发顺序，最推荐如下：

第一步：

- 复现 strongest possible CLIP-based baseline；
- 在 SurgLaVi-Beta 上达到接近或持平当前可比结果。

第二步：

- 加 local temporal expansion；
- 加 candidate local bag；
- 加 soft confidence weighting；
- 用 bag-level contrastive 替代 hard positive clip-text alignment。

第三步：

- 加 hierarchy role decomposition；
- 加 procedure-aware hard negatives。

第四步：

- 视实验情况再加 top-k VTM auxiliary loss；
- 或加入 weak temporal order regularization。

## 一句话方法定义

如果现在要把你的方法压缩成一句最稳的定义，可以写成：

> We build a CLIP-style surgical VLP framework that replaces uniform hard-pair clip-text alignment with uncertainty-aware local evidence bag learning, where weak timestamps define a candidate temporal neighborhood, multiple local clips are scored by multi-pathway confidence, and hierarchical texts provide role-specific semantic regularization.

## 一版可发表的方法设计

下面给出一版当前最推荐、也最有发表潜力的方法方案。该方案默认 backbone 固定为 CLIP-based surgical VLP framework，方法创新集中在 weak supervision 的使用方式，而不是 encoder architecture 本身。

暂时将该方法记作：

**CABLE-Surg**

全称可理解为：

**Credibility-Aware Bag-Level Local Evidence Learning for Surgical VLP**

这个名字不是最终稿，但足够准确地概括方法核心：不是重新定义 backbone，而是在 weak pair 内部做可信局部证据的 bag-level 学习。

### 1. Problem Setup

给定一个弱监督训练样本：

\[
(V, [t_s, t_e], y_f, y_m, y_c)
\]

其中：

- \(V\) 表示手术视频；
- \([t_s, t_e]\) 表示弱时间戳区间；
- \(y_f\) 表示 fine-level text；
- \(y_m\) 表示 mid-level text；
- \(y_c\) 表示 coarse/video-level text。

CLIP-based 框架提供：

- 视觉编码器 \(g_v(\cdot)\)
- 文本编码器 \(g_t(\cdot)\)

得到文本 embedding：

\[
t_f = g_t(y_f), \quad t_m = g_t(y_m), \quad t_c = g_t(y_c)
\]

为什么这样定义：

- 保持和现有 surgical VLP 主线兼容，不改变数据形式；
- 不要求额外人工标注；
- 论文贡献可以集中在“同样的弱监督，是否能被更合理地利用”。

### 2. Local Temporal Expansion

对于原始 weak timestamp interval，不直接把它当成精确监督窗口，而是在其两侧扩展一个局部邻域：

\[
[t_s', t_e'] = [t_s - \Delta, \; t_e + \Delta]
\]

随后将该扩展窗口划分或采样为 \(N\) 个 candidate local clips：

\[
B = \{c_1, c_2, \dots, c_N\}
\]

每个候选片段经过视觉编码器得到 embedding：

\[
v_i = g_v(c_i), \quad i=1,\dots,N
\]

为什么这样做：

- 弱监督时间戳往往存在轻微偏移，直接信任原始窗口会漏掉真正证据；
- 文本支持的视觉内容通常只出现在局部时间范围，而不是覆盖整个 clip；
- 扩展窗口能覆盖时间误差，candidate bag 则把单一正样本改写为局部候选集合。

这一模块解决的是“证据可能在附近，而不是严格在原区间内”的问题。

### 3. Multi-Pathway Confidence Scoring

对每个 candidate local clip \(c_i\)，不使用单一路径的 clip-text similarity 作为可信度，而是构造一个多路径 confidence score：

\[
a_i = \alpha s_i^{f} + \beta s_i^{m} + \gamma s_i^{c} + \eta p_i
\]

其中：

- \(s_i^{f} = \text{sim}(v_i, t_f)\)：fine-level similarity；
- \(s_i^{m} = \text{sim}(v_i, t_m)\)：mid-level consistency；
- \(s_i^{c} = \text{sim}(v_i, t_c)\)：coarse-level compatibility；
- \(p_i\)：temporal proximity prior，表示 candidate 与原始时间戳中心的距离先验；
- \(\alpha,\beta,\gamma,\eta\) 为可学习或可调系数。

然后通过 softmax 得到 confidence distribution：

\[
q_i = \frac{\exp(a_i / \tau_q)}{\sum_{j=1}^{N}\exp(a_j / \tau_q)}
\]

为什么这样做：

- 如果只依赖 fine-level similarity，模型容易被噪声文本、视觉偶然相似性或局部错误匹配误导；
- fine-level text 提供局部语义证据；
- mid / coarse-level text 提供更高层级的语义约束；
- temporal prior 反映原始 weak timestamp 仍然具有弱可信度。

这个设计的核心逻辑是：局部证据可信度不应由单一信号决定，而应由“时间邻近 + 多层语义一致性”共同决定。

### 4. Top-k Soft Evidence Bag

在得到 confidence distribution 后，不直接对所有 candidate 做均匀池化，而是只保留 top-k 高置信候选片段，并在其中做 soft aggregation：

\[
\tilde{q}_i =
\begin{cases}
\dfrac{q_i}{\sum_{j \in \text{TopK}(q)} q_j}, & i \in \text{TopK}(q) \\
0, & \text{otherwise}
\end{cases}
\]

得到 bag-level local evidence representation：

\[
v_{\text{bag}} = \sum_{i=1}^{N} \tilde{q}_i v_i
\]

同时保留一个全局 clip representation：

\[
v_{\text{global}} = g_v(c_{\text{global}})
\]

其中 \(c_{\text{global}}\) 可以来自原始区间或扩展区间的整体池化表示。

为什么这样做：

- 如果对所有 candidate 做均匀加权，会重新回到原始问题：无关片段污染正样本；
- 如果只取 argmax 单个片段，又太脆弱，容易被早期训练噪声带偏；
- top-k soft bag 是折中方案：既允许多个局部证据共同支持文本，又能抑制无关片段。

这一模块解决的是“证据不是唯一片段，但也绝不是整段平均有效”的问题。

### 5. Bag-Level Contrastive Learning

用 \(v_{\text{bag}}\) 替代传统 pooled clip embedding，与 fine-level text 进行 CLIP-style contrastive learning：

\[
\mathcal{L}_{fine} = \mathcal{L}_{\text{InfoNCE}}(v_{\text{bag}}, t_f)
\]

也可以使用更显式的 bag positive formulation，将 positive similarity 写成 weighted log-sum-exp：

\[
S^{+}(V, y_f) = \log \sum_{i=1}^{N} \exp \left( \frac{\tilde{q}_i \cdot \text{sim}(v_i, t_f)}{\tau} \right)
\]

并将其纳入标准对比学习分母中，与 batch 内负样本进行竞争。

为什么这样做：

这是整个方法最核心的改动。现有 CLIP-style VLP 默认一个 clip-text pair 是单一 hard positive；而这里正样本被改写为一个不确定的局部 evidence bag。这样做的直接好处是：

- 正样本定义更符合 weak supervision 的真实结构；
- 无关局部内容不再被强制拉近文本；
- fine-grained zero-shot transfer，尤其是 action 与 triplet，更容易从真正关键局部证据中学习。

### 6. Hierarchy Role Decomposition

在 full hierarchy setting 下，不同层级文本不共享同一角色，而分别承担不同监督功能：

\[
\mathcal{L}_{hier} = \lambda_m \mathcal{L}_{mid}(v_{\text{bag}}, t_m) + \lambda_c \mathcal{L}_{coarse}(v_{\text{global}}, t_c)
\]

其中：

- fine-level text \(t_f\)：用于 local evidence selection 与 bag-level matching；
- mid-level text \(t_m\)：用于约束局部证据与 step/phase 语义一致；
- coarse-level text \(t_c\)：用于约束全局 procedure context。

为什么这样做：

- 如果把 fine/mid/coarse 都当成同等地位的正文本，只会得到“更多文本 supervision”，但很难真正解决局部可信度问题；
- fine-level 最适合告诉模型“哪里是证据”；
- mid-level 最适合告诉模型“这个局部证据是否仍然属于正确的 step/phase”；
- coarse-level 最适合防止模型过度聚焦局部细节而丢失 procedure-level context。

这一模块让 hierarchy 真正服务于你的核心问题，而不是变成额外但分散的多任务目标。

### 7. Procedure-Aware Hard Negatives

在标准 batch negatives 之外，额外构造 procedure-aware hard negatives：

- 同 procedure、不同 fine label；
- 同 phase、不同 action；
- 同 tool、不同 target；
- 语义相近但局部证据不匹配的候选样本。

它们可被加入 contrastive denominator，或通过额外 margin loss 强化区分。

为什么这样做：

- 普通随机负样本通常过于容易，尤其在手术场景下，很多样本之间共享相似器械、组织背景和阶段上下文；
- hard negatives 能强迫模型学习真正细粒度的局部差异；
- 能避免模型只学到 procedure-level 粗语义；
- 直接提升 action / triplet 这样的高难 zero-shot 任务。

### 8. Temporal Compactness Regularization

由于真正支持文本的证据通常集中在一个相对连续的局部时间范围，而不是离散散落在整个扩展窗口中，因此对 confidence distribution 加一个 temporal compactness regularizer：

\[
\mu_q = \sum_{i=1}^{N} q_i \cdot \tau_i
\]

\[
\mathcal{L}_{compact} = \sum_{i=1}^{N} q_i (\tau_i - \mu_q)^2
\]

其中 \(\tau_i\) 表示第 \(i\) 个 candidate 的时间位置。

为什么这样做：

- 只做 soft weighting 仍可能得到几个分散的伪高分片段；
- 手术中的有效局部证据通常在时间上相对紧凑；
- 该约束可以鼓励模型选择一个更可信、更连续的 evidence region，而不是若干离散巧合片段。

这个正则的优点是轻量、稳定、符合问题先验，而且不会像完整 sequence alignment 那样过重。

### 9. Final Objective

最终损失可写为：

\[
\mathcal{L} = \mathcal{L}_{fine} + \lambda_m \mathcal{L}_{mid} + \lambda_c \mathcal{L}_{coarse} + \lambda_h \mathcal{L}_{hardneg} + \lambda_r \mathcal{L}_{compact}
\]

其中：

- \(\mathcal{L}_{fine}\)：bag-level local contrastive loss；
- \(\mathcal{L}_{mid}\)：mid-level consistency loss；
- \(\mathcal{L}_{coarse}\)：coarse/global context loss；
- \(\mathcal{L}_{hardneg}\)：procedure-aware hard negative term；
- \(\mathcal{L}_{compact}\)：temporal compactness regularizer。

为什么这样设计总损失：

- fine loss 解决核心问题，即 local evidence uncertainty；
- hierarchy losses 负责稳定语义层级；
- hard negative 负责提升判别性；
- compactness regularizer 负责限制伪对齐。

每个损失都服务于同一个主问题，不是松散堆模块。

## 为什么这一版方法最有发表潜力

### 1. 与现有 surgical VLP 主线兼容

- 不要求替换 backbone；
- 不要求重建数据集；
- 不要求额外精标 temporal annotations；
- 可直接在 SurgLaVi-Beta 上实现。

### 2. 与现有论文有明显区分

- 相比 SurgVLP：不再把 weak pair 当成单一 hard positive；
- 相比 HecVL：hierarchy 不只是 fine-to-coarse supervision，而是 role decomposition；
- 相比 PeskaVLP：核心不是 LLM augmentation 或 DTW alignment，而是 bag-level local evidence learning；
- 相比 SurgLaVi：核心不是数据规模，而是 weak supervision usage；
- 相比 VidLPRO：保持 CLIP-style dual encoder，不依赖 heavy fusion objective。

### 3. 能自然解释 downstream gain

这套方法不只对 phase recognition 合理，对更细粒度的 action、tool、triplet 也有直接解释力，因为它明确让模型从局部可信证据中学习，而不是从整个 weak clip 的平均语义中学习。

## 最小可实现版本

如果当前实现资源有限，最小版本可先保留：

- Local Temporal Expansion
- Multi-Pathway Confidence Scoring（至少包含 fine similarity + temporal prior）
- Top-k Soft Evidence Bag
- Bag-Level Contrastive Learning
- Hierarchy Role Decomposition

先不加入 hard negatives 和 compactness regularizer，也能形成一版有完整主张的方法论文。

## 推荐实现顺序

1. 跑稳 strongest CLIP-based baseline；
2. 加 local temporal expansion 和 candidate bag；
3. 加 confidence scoring 与 bag-level contrastive；
4. 加 hierarchy role decomposition；
5. 再加 hard negatives；
6. 最后视情况加 compactness regularizer。

## 当前 baseline 的正式表述

当前可采用如下 baseline 作为后续方法比较的统一起点。

首先，基于 SurgLaVi / SurgLaVi-Beta 的层级标注构建弱监督的 video-text training pairs。每个训练样本由一个视频片段、对应的标注时间区间以及相应层级的文本描述组成。在 baseline 设置下，先使用 fine-level caption 作为主要文本监督，并从对应标注时间段内采样多帧作为视频输入。

在视觉分支中，先使用 ConvNeXt 或 LemonFM 提取逐帧视觉特征，再通过一个 TimeSformer-style temporal aggregation head 将多帧特征聚合为单个视频 embedding。在文本分支中，使用 SurgicBERTa 对 caption 进行编码，得到文本 embedding。随后，视频 embedding 与文本 embedding 被投影到同一共享语义空间中，并通过 CLIP-style bidirectional contrastive learning 进行优化，从而学习跨模态对齐的表征。

这一 baseline 的关键点在于：

- 输入仍然是标准的 weakly paired clip-text samples；
- 不显式建模 pair 内部的局部证据不确定性；
- 视频侧使用多帧输入，但最终仍汇聚成单一 clip representation；
- 文本侧默认 caption 对整个 clip 提供均匀可信的正样本监督。

因此，这一 baseline 正好构成后续方法的最直接对照：如果后续方法有效，其提升应来自更合理地建模 weak pair 内部的局部证据结构，而不是来自 backbone 或整体训练框架的变化。

## baseline 表述时需要注意的点

- 不要把该 baseline 写成“精确配对样本”，而应写成“弱监督的 video-text pairs”。
- 如果使用的是 fine-level caption 作为 baseline 文本输入，需要明确写出，以避免和 full hierarchy 版本混淆。
- 如果 temporal head 只是做时序聚合，不要写得像完整 video transformer backbone，避免 reviewer 误解你更换了主干架构。
- baseline 的核心假设要明确指出：它把标注区间内采样得到的多帧统一压缩为一个 clip embedding，并将其整体与文本对齐。
