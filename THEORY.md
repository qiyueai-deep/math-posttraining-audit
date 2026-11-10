# 偏好信号的可识别性

设偏好数据集为 $\mathcal{D}=\{(x_i,y_i^+,y_i^-)\}_{i=1}^n$，标签恒为 $y_i^+$ 优于 $y_i^-$。记 $S_i$ 为一对回答在表面特征上的差，例如字符数差、是否含最终答案标记、LaTeX 标记数差与数字个数差，记 $Q_i$ 为两者在真实数学质量上的差。DPO 的经验风险只通过成对回答的相对对数概率依赖于策略，

$$L(\theta)=-\frac{1}{n}\sum_{i=1}^{n}\log\sigma\Big(\beta\big[\log\tfrac{\pi_\theta(y_i^+|x_i)}{\pi_\theta(y_i^-|x_i)}-\log\tfrac{\pi_{\rm ref}(y_i^+|x_i)}{\pi_{\rm ref}(y_i^-|x_i)}\big]\Big).$$

命题. 若存在可测函数 $f$ 使偏好标签满足 $Y_i=f(S_i)$ 几乎处处，且 $Q_i\perp Y_i\mid S_i$，则 $L(\theta)$ 不识别 $\theta$ 在 $Q$ 方向的分量。因此在使训练集 reward accuracy 达到 1 的策略中，存在真实数学表现任意差的策略。

证明. 记 $\Delta_\theta(x_i)=\log\pi_\theta(y_i^+|x_i)-\log\pi_\theta(y_i^-|x_i)$。$L$ 只通过 $\Delta_\theta$ 依赖于 $\theta$。给定任意 $\theta$，取 $\theta'$ 使对每个 $x_i$ 与 $\mathcal D$ 中出现的两个回答有 $\log\pi_{\theta'}(y|x_i)=\log\pi_\theta(y|x_i)+c_i$，常数 $c_i$ 由把 $\mathcal D$ 之外回答上的概率质量重新归一得到。于是 $\Delta_{\theta'}=\Delta_\theta$，从而 $L(\theta')=L(\theta)$。而 $\mathcal D$ 之外回答上的概率分布未被数据约束，可以让 $\theta'$ 在与 $Q$ 相关的回答上取任意允许的概率，因而两者的真实数学表现可以不同。等价地从统计可识别性看，把 $\theta$ 分解为控制 $\mathcal D$ 内相对对数概率的 $\phi$ 与控制 $\mathcal D$ 外、与 $Q$ 相关行为的 $\psi$，似然只依赖 $\phi$，关于 $\psi$ 的 Fisher 信息为零，$\psi$ 不可识别。

实验上把这个条件做成可操作的量。只用 $S$ 的逻辑回归按偏好对分组做 5 折交叉验证，定义 $SI=2\,AUC-1$ 为表面线索能解释的偏好可分性，残差可识别性为 $1-SI$。复刻旧缺陷的 shortcut 组测得 AUC 为 1.0000，说明 $Y$ 被 $S$ 确定性解释，长度与格式匹配的 deconfounded 组测得 0.5000，说明线索已被切断。数值上随机重排 $Q$ 的分配 50 次，$L$ 逐位不变，而令 $\Delta_\theta$ 同时依赖 $Q$ 时 $L$ 发生变化，这是上述观测等价性的直接佐证。旧项目 240 条偏好对的 AUC 同样是 1.0000，因此它报告的 reward accuracy 为 1.0 只说明模型学会了长度与格式，不构成关于数学能力的证据。

同一个恒等式也解释了旧项目 β 扫描的无效。隐式 reward 为 $r_\theta=\beta\log(\pi_\theta/\pi_{\rm ref})+c(x)$，reward margin 与 β 成正比是定义使然，实测 margin/β 的变异系数为 0.0016，而测试集 loss 在四个 β 之间只变化 0.005。要度量策略偏离参考模型的程度，应直接估计 $D_{\rm KL}(\pi_\theta\|\pi_{\rm ref})$，本项目用从 $\pi_\theta$ 采样的 $\log\pi_\theta-\log\pi_{\rm ref}$ 均值作为无偏估计，并按 prompt 为聚类单位给 bootstrap 区间。

条件独立假设在真实语料里未必严格成立，去混淆偏好对就是对它的实验性保证，KL 锚点只限制解的范数，不改变不可识别的结论。

算法实验台里把策略限制成候选集上的 log-linear 模型，$\log\pi_\theta(y|x)=\theta\cdot\phi(x,y)-\log Z_\theta(x)$，其中 $Z_\theta(x)=\sum_{y\in Y(x)}\exp(\theta\cdot\phi(x,y))$ 可以精确计算。同一 prompt 下配分函数在成对比较里抵消，令 $\Delta_i=(\theta-\theta_{\rm ref})\cdot(\phi_i^+-\phi_i^-)$，则 $L=-\frac1n\sum_i\log\sigma(\beta\Delta_i)$，其梯度为 $-\frac\beta n\sum_i(1-\sigma(\beta\Delta_i))(\phi_i^+-\phi_i^-)$，而 $D_{\rm KL}(\pi_\theta\|\pi_{\rm ref})$ 在候选集上是有限求和，无需 Monte Carlo。于是上述命题的条件可以直接落成可计算的量：当所有 $\phi_i^+-\phi_i^-$ 只在表面特征方向非零时，梯度在质量方向恒为零，去混淆组则让该方向成为唯一非零分量。固定 $\theta$ 时 $\mathrm{margin}/\beta=\frac1n\sum_i\Delta_i$ 与 $\beta$ 无关，实测变异系数为 0.0000；真正训练时 $\beta$ 通过梯度尺度改变收敛位置，此时应报告精确 KL 与下游准确率，实测四个 $\beta$ 的 KL 跨度为 0.1276。真实语言模型里 $Z_\theta(x)$ 不可算，配分函数只能靠采样近似，所以模型层改用按 prompt 聚类的序列级 KL 估计，两层的差别正是解析实验台能验证目标函数、却不能替代权重训练的原因。
