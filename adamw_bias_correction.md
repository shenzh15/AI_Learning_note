# AdamW 优化器中的学习率修正项（bias correction）详解

在 Adam 和 AdamW 优化器中，更新参数时常常出现如下形式的修正项：

$$\alpha_t = \alpha \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}$$

这一项称为 **偏差修正项（bias correction term）**，是为了纠正在训练初期由于动量初始为 0 所带来的偏差。

---

## 一、AdamW 参数更新公式

下面是标准的 AdamW 参数更新公式（来自论文）：

$$\begin{aligned}
&\text{初始化：} \\
&\quad \theta_0 \leftarrow \text{初始参数} \\
&\quad m_0 \leftarrow 0 \quad (\text{一阶动量}) \\
&\quad v_0 \leftarrow 0 \quad (\text{二阶动量}) \\
\\
&\text{对每个 step } t = 1, \dots, T: \\
&\quad g_t = \nabla_\theta \ell(\theta_t; B_t) \quad \text{(计算梯度)} \\
&\quad m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(更新一阶动量)} \\
&\quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(更新二阶动量)} \\
&\quad \hat{\alpha}_t = \alpha \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t} \quad \text{(学习率修正)} \\
&\quad \theta_t \leftarrow \theta_{t-1} - \hat{\alpha}_t \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} \quad \text{(梯度更新)} \\
&\quad \theta_t \leftarrow \theta_t - \alpha \lambda \theta_t \quad \text{(权重衰减）}
\end{aligned}$$

---

## 二、为什么要有 bias correction？

Adam 的一阶和二阶动量都是用指数滑动平均（EMA）计算的：

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

由于初始时 $m_0 = 0$，所以初期的 $m_t$ 和 $v_t$ 会偏小。

你可以数学推导得到：

$$\mathbb{E}[m_t] \approx (1 - \beta_1^t) \cdot \text{true\_mean}$$
$$\mathbb{E}[v_t] \approx (1 - \beta_2^t) \cdot \text{true\_var}$$

所以我们需要修正：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

这就是所谓的 bias correction。避免 early steps 更新太小，训练进展缓慢。

---

## 三、指数滑动平均为什么叫“指数”？

以一阶动量为例：

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

我们展开几步可以得到：

$$m_t = (1 - \beta_1) \cdot \left(g_t + \beta_1 g_{t-1} + \beta_1^2 g_{t-2} + \cdots + \beta_1^{t-1} g_1\right)$$

可以看出每个历史梯度 $g_{t-k}$ 的权重是 $\beta_1^k$，也就是**指数衰减**。越早的梯度，权重越小，呈现指数下降。这种结构让：

- 历史信息仍然保留（长期记忆）；
- 近期信息更重要（短期适应）；
- 更新过程更加平滑。

这就是“指数滑动平均”名称的由来。

---

为了方便，很多实现（包括 PyTorch）直接把这个 bias correction 融合到学习率里：

$$\alpha_t = \alpha \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}$$

这样就不用手动除以：$1 - \beta_1^t$ 和 $1 - \beta_2^t$ 代码更简洁。

---

## 五、总结

- Adam 和 AdamW 使用动量估计 $m, v$ 来加速收敛。
- 由于动量初始化为 0，前几步的估计会偏小。
- 所以用如下修正因子：$1 - \beta_1^t, \quad 1 - \beta_2^t$
- 这就是 bias correction。
- 在代码中常把它融合进学习率 $\alpha_t$。
- EMA 是指数加权的滑动平均，因此具有良好平滑性。

这项修正是 **Adam 收敛稳定的关键**，尤其在训练初期和 batch size 较小时影响更明显。

