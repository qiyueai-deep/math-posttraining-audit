# 大模型算法岗专项对答

这份文档回答「你到底做没做过后训练算法」。项目讲述与统计追问见 `interview_guide.md`。
所有数字来自 `reports/results.jsonl`，由脚本渲染。

## 一、三层验证，开场就讲清楚

第一层是算法机制层：在 CPU 上真实训练一个 7 维 log-linear 策略，
跑通 SFT 的最大似然、DPO 的梯度下降、精确 KL、reward margin、beta 扫描和参考模型锚点对照。
DPO 解析梯度用中心有限差分校验，最大误差在 1e-11 量级。
这一层不是 Qwen 权重，但它验证了目标函数与优化行为，不需要 GPU。
第二层是权重层 CPU pilot：本机有 CPU 版 torch 与完整的 Qwen3-0.6B 权重，
于是在 `configs/model_pilot.yaml` 写明的缩减规模下真跑 LoRA-SFT、
显式冻结 SFT ref 的 DPO、masked loss、三态生成评测与序列级 KL，
数字记在 `model_pilot` 实验，规模与 adapter 哈希记在 `model_pilot_manifest.json`。
第三层才是完整协议：256 token、303 题、四个泛化子集，本机 CPU 跑不完，
所以 9 个 `model_stage` 指标仍是 `pending_model_run`，一个数字没编。

## 二、后训练链路逐环节

| 环节 | 实现 | 能讲的算法点 | 文件 |
|---|---|---|---|
| 数据 | 参数化题族加独立 oracle 双路径交叉验证 | 数据泄露、按族切分、泛化子集设计 | `src/core/data_gen.py`, `splits.py` |
| 评测 | 受限语法精确解析，correct / incorrect / unverifiable 三态 | 为什么不回退子串匹配、coverage 与 precision 取舍 | `src/core/expressions.py`, `verifier.py` |
| SFT | LoRA 微调，只对 assistant completion 算 NLL | chat template、offset_mapping 字符级 mask、padding 与 attention mask | `src/model/train_sft.py`, `teacher_forced.py` |
| 偏好数据 | 三组对照，长度格式匹配只改数学内容 | hard negative、shortcut、反事实组 | `src/core/preferences.py` |
| DPO | 独立基座加载冻结 SFT adapter 作 ref_model | KL 锚点、隐式 reward、梯度形式 | `src/model/train_dpo.py` |
| 诊断 | 采样估计序列级 KL 与按 prompt 聚类的区间 | 为什么真实模型里 KL 不能解析、beta 怎么调 | `src/model/diagnostics.py` |
| 生成评测 | 统一解码协议、256 token、四类泛化子集 | greedy 与 sampling、截断、答案抽取 | `src/model/evaluate.py` |

## 三、白板上能写的推导

SFT 是候选集上的最大似然。log-linear 策略下
$$\nabla_\theta \mathrm{NLL} = \mathbb{E}_{\pi_\theta}[\phi] - \phi(y^+),$$
所以 SFT 把质量特征的系数往上推。

DPO 来自带 KL 约束的 RLHF 最优解 $\pi^* \propto \pi_{\rm ref}\exp(r/\beta)$，
反解出 $r = \beta\log(\pi_\theta/\pi_{\rm ref}) + c(x)$，代入 Bradley-Terry 得到成对分类损失
$$L(\theta)=-\mathbb{E}\log\sigma\big(\beta[\log\tfrac{\pi_\theta(y^+|x)}{\pi_{\rm ref}(y^+|x)}-\log\tfrac{\pi_\theta(y^-|x)}{\pi_{\rm ref}(y^-|x)}]\big).$$
同一 prompt 下配分函数 $Z_\theta(x)$ 抵消，令 $\Delta=\theta\cdot(\phi^+-\phi^-)$，
$$\nabla_\theta L = -\beta(1-\sigma(\beta\Delta))(\phi^+-\phi^-).$$
这就是实验台里跑的梯度，也是有限差分校验的对象。

LoRA 写 $W = W_0 + \frac{\alpha}{r}BA$，$B$ 初始化为零保证训练起点等于基座，
秩 $r$ 决定可训练子空间维度；DPO 里如果 ref_model 置空，TRL 会 disable adapter，
锚点变成未训练的 base，目标函数不再是上面这条。

## 四、算法实验台的关键数字

| 量 | 值 | 说明 |
|---|---:|---|
| 探针题数 | 120 | 2x2 平衡：正确/错误 x 短/长，另设长错误对抗探针 |
| 特征维度 | 7 | 6 个表面特征加 1 个严格验证器给出的正确性特征 |
| SFT 训练样本 | 252 | 真实跑 MLE 的指令条数 |
| DPO 梯度有限差分误差 | 3.44e-11 | 解析梯度与中心差分的最大绝对差 |
| margin/beta 变异系数（固定策略） | 0.0000 | 恒等式，不是实验发现 |
| margin/beta 变异系数（训练后） | 0.5727 | beta 改变优化轨迹，所以不再是常数 |
| 正确锚点的 KL | 0.0513 | ref = SFT 策略 |
| 锚错到 base 的 KL | 0.5179 | policy 仍从 SFT 出发，ref 却是 base |
| 锚点造成的 KL 差 | 0.4666 | 旧项目 ref_model=None 的量化后果 |

## 五、三组偏好对照

| 组 | reward accuracy | reward margin | 长度偏置劫持率 | 质量系数 | 长度系数 |
|---|---:|---:|---:|---:|---:|
| shortcut | 1.0000 | 0.1696 | 0.7241 | 2.0335 | 0.5651 |
| deconfounded | 1.0000 | 0.1165 | 0.1864 | 2.8226 | 0.2470 |
| length_reversed | 1.0000 | 0.1321 | 0.0720 | 2.8178 | -0.0082 |

三组的训练指标 reward accuracy 完全一样，margin 也都上升，
但 shortcut 组把梯度主要花在长度特征上，对抗探针里偏向「更长但错误」的回答（劫持率 0.7241），
去混淆组把梯度花在正确性特征上，劫持率降到 0.1864。
这就是 reward hacking 在目标函数层面的可复现证据，训练曲线见 `algo_lab_curve.svg`。

## 六、算法岗高频追问

**1. 你说做了 DPO，到底跑没跑？**
跑了三层里的前两层。算法机制层在 CPU 上真跑 SFT MLE 与 DPO 梯度下降，
梯度用有限差分校验过，有完整曲线。权重层用真实的 Qwen3-0.6B 加 LoRA 跑了
缩减规模的 CPU pilot，SFT 步数、偏好对数、评测题数与生成长度都写在配置里，
逐条预测和 adapter 哈希都落了盘。第三层完整协议需要 GPU，9 个指标明确标 pending。
我不会把 pilot 数字说成完整协议结果，也不会用解析策略冒充权重训练。

**2. 为什么用 log-linear 策略，不是真 transformer？**
因为要验证的是目标函数层面的性质，不是模型容量。log-linear 策略的配分函数可算，
DPO 梯度与 KL 都有闭式解，能把「beta 是缩放还是改变优化」「ref 锚错差多少」
从口径争论变成可计算的数字。真实模型里 Z 不可算，KL 必须采样估计，
我在 `diagnostics.py` 里给出的就是按 prompt 聚类的 MC 估计与区间。

**3. DPO 和 PPO / GRPO 怎么选？**
DPO 不需要显式 reward model 和在线采样，用离线偏好对直接优化隐式 reward，工程成本最低，
但对偏好数据质量极敏感，这个项目量化的正是这个敏感性。PPO 需要 reward model 与 value net，
GRPO 用组内相对优势省掉 value net，两者都在线采样、能持续探索，代价是算力与实现复杂度。
小样本、弱验证器的场景下先修数据和评测比换算法更划算。

**4. 为什么 ref 必须是 SFT 而不是 base？**
DPO 推导里的参考策略是 SFT 模型。TRL 在 ref_model=None 且 policy 是 PeftModel 时会 disable adapter，
锚点变成未训练 base，KL 惩罚把 SFT adapter 参数当成可直接改的变量。
实验台上锚错使 KL 从 0.0513 升到 0.5179，差 0.4666。
修正版用独立基座实例加载冻结的 SFT adapter，并在训练前逐参数断言 ref 与 policy 初值一致。

**5. beta 怎么调？**
不能看 reward margin。固定策略时 margin 就是 beta 乘 log-ratio 差，
实验台上固定策略的 margin/beta 变异系数是 0.0000，纯恒等式。
真正训练时 beta 会改变梯度尺度与收敛位置，要看序列级 KL 和下游生成准确率，
四个 beta 下 KL 跨度 0.1276，下游准确率跨度 0.0884。

**6. LoRA 的 B 为什么初始化为零？rank 怎么选？**
零初始化保证训练起点与基座完全一致，SFT 的 ref 才有意义。rank 决定低秩子空间的容量，
本项目没有做 rank 消融，简历里不写这一项。真要做，应该固定数据、beta 与步数，
按题目簇给配对区间，而不是只看训练集 loss。

**7. assistant-only loss mask 具体怎么做？**
用 chat template 分别生成完整对话与只到 prompt 的文本，断言前缀关系，
再用 fast tokenizer 的 offset_mapping 在字符级定位边界，把字符起点小于 prompt 长度的 label 设为 -100。
不用 token 级前缀断言，因为 BPE 在交界处可能合并出不同 token。总 NLL 按 token 加权。

**8. 生成评测的协议是什么？**
关闭 thinking，max_new_tokens 至少 256，严格抽取最终答案，解析失败记 unverifiable 而不回退字符串匹配，
按未见参数、同义改写、反事实、未见题型四个子集分别报准确率，并按题目簇给区间。
旧项目用 64 token 截断，把没写完答案全记成数学错误，这条已修掉。

**9. 长度偏置怎么检测？**
两个层面。训练前用只用表面特征的交叉验证 AUC 体检偏好数据，shortcut 组 1.0000，去混淆组 0.5000。
训练后用对抗探针，让「短的正确」与「明显更长的错误」二选一，
shortcut 组劫持率 0.7241，去混淆组 0.1864。

**10. 0.6B 这么小，结论有意义吗？**
算法层验证的是目标函数性质，与模型规模无关。模型层选 0.6B 是为了在 CPU 上也能复现，
而偏好数据里没有真错误、测试集只有 2 道题这类结构性缺陷，换更大的模型不会自动消失。

**11. 给你 GPU，第一件事做什么？**
跑三组 DPO 的下游对照，看去混淆偏好对在真实权重上是否也把梯度从长度转到正确性，
同时把序列级 KL 的 MC 估计与解析实验台的结论对齐。再在 CMATH 或 GSM8K 中文子集上重复整套诊断。

## 七、不能说的话

不要把 algo_lab 的数字说成 Qwen3 的训练结果，不要报任何 pending 指标，
不要说做过 LoRA rank 消融或 GRPO，不要复述旧项目那组无法复现的 loss 与 PPL。
