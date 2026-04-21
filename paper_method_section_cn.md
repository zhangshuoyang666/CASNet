# 3. 方法

## 3.1 总体架构

我们提出了一种基于DeepLabV3+的增强型语义分割框架，称为**SA-BNet**（Spatial-Attention Boundary-aware Network，空间注意力边界感知网络），该框架在一个统一的端到端架构中同时解决多尺度上下文聚合、注意力引导的特征重校准、跨层级特征融合以及显式边界监督等问题。总体流程如图2所示。

**编码器。** 给定输入图像 $\mathbf{I} \in \mathbb{R}^{3 \times H \times W}$，我们采用在ImageNet上预训练的ResNet-101作为特征编码骨干网络。遵循密集预测任务的通行做法，我们采用输出步幅（Output Stride, OS）为16的设置，即将最后一个残差阶段中的stride-2卷积替换为空洞卷积，从而在不牺牲所学表征深度的前提下保留更高的空间分辨率。具体而言，当OS = 8时，layer3和layer4的最后两个残差块分别使用空洞率 $\{1, 2\}$ 和 $\{1, 2, 4\}$；当OS = 16时，仅对layer4进行膨胀操作。

骨干网络通过中间层提取机制，在三个不同的语义层级上输出特征图层次：

$$\mathbf{F}_{\text{low}} = \phi_{\text{layer1}}(\mathbf{I}) \in \mathbb{R}^{C_l \times \frac{H}{4} \times \frac{W}{4}} \tag{1}$$

$$\mathbf{F}_{\text{mid}} = \phi_{\text{layer2}}(\mathbf{I}) \in \mathbb{R}^{C_m \times \frac{H}{8} \times \frac{W}{8}} \tag{2}$$

$$\mathbf{F}_{\text{high}} = \phi_{\text{layer4}}(\mathbf{I}) \in \mathbb{R}^{C_h \times \frac{H}{16} \times \frac{W}{16}} \tag{3}$$

其中，对于ResNet-101，$C_l = 256$，$C_m = 512$，$C_h = 2048$。$\phi_{\text{layer}k}$ 表示从输入图像到第 $k$ 个残差阶段输出的复合映射。与原始DeepLabV3+仅利用低层特征 $\mathbf{F}_{\text{low}}$ 和高层特征 $\mathbf{F}_{\text{high}}$ 不同，我们的框架**显式地从第二个残差阶段提取中层特征** $\mathbf{F}_{\text{mid}}$。该中间表征能够捕获介于低层视觉线索与高层语义抽象之间的结构和几何模式——例如边缘、角点和纹理边界——为后续的特征融合提供了关键的信息桥梁。

**多尺度上下文聚合。** 高层特征 $\mathbf{F}_{\text{high}}$ 被送入空洞空间金字塔池化（Atrous Spatial Pyramid Pooling, ASPP）模块，以捕获多尺度上下文信息。ASPP模块由五个并行分支组成，同时对 $\mathbf{F}_{\text{high}}$ 进行操作。每个分支 $b_k$ 在不同的尺度上提取特征：

$$b_0(\mathbf{F}_{\text{high}}) = \delta\left(\text{BN}\left(\mathbf{W}_0 * \mathbf{F}_{\text{high}}\right)\right), \quad \mathbf{W}_0 \in \mathbb{R}^{256 \times C_h \times 1 \times 1} \tag{4}$$

$$b_k(\mathbf{F}_{\text{high}}) = \delta\left(\text{BN}\left(\mathbf{W}_k *_{r_k} \mathbf{F}_{\text{high}}\right)\right), \quad k = 1, 2, 3 \tag{5}$$

$$b_4(\mathbf{F}_{\text{high}}) = \text{Up}\left(\delta\left(\text{BN}\left(\mathbf{W}_4 * \text{GAP}(\mathbf{F}_{\text{high}})\right)\right)\right) \tag{6}$$

其中，$*_{r_k}$ 表示空洞率为 $r_k \in \{6, 12, 18\}$ 的 $3 \times 3$ 空洞卷积，$\delta(\cdot)$ 表示ReLU激活函数，$\text{BN}(\cdot)$ 表示批归一化，$\text{GAP}(\cdot)$ 为自适应全局平均池化（将空间维度压缩至 $1 \times 1$），$\text{Up}(\cdot)$ 通过双线性插值将池化表征上采样回输入空间尺寸。$\mathbf{W}_0$ 至 $\mathbf{W}_4$ 为可学习的卷积核，每个分支输出256个通道。五个分支的输出沿通道维度进行拼接：

$$\mathbf{F}_{\text{cat}} = \text{Concat}\left[b_0(\mathbf{F}_{\text{high}}),\; b_1(\mathbf{F}_{\text{high}}),\; b_2(\mathbf{F}_{\text{high}}),\; b_3(\mathbf{F}_{\text{high}}),\; b_4(\mathbf{F}_{\text{high}})\right] \in \mathbb{R}^{1280 \times \frac{H}{16} \times \frac{W}{16}} \tag{7}$$

三个空洞卷积分支的有效感受野为 $3 + 2(r_k - 1)$，当 $r_k \in \{6, 12, 18\}$ 时，有效卷积核大小分别为 $\{13, 25, 37\}$ 像素。结合 $1 \times 1$ 局部分支和全局池化分支，ASPP模块的感受野覆盖范围从单个像素跨越至整个特征图。

**注意力引导的特征重校准。** 拼接后的多尺度特征 $\mathbf{F}_{\text{cat}}$ 包含来自重叠感受野的丰富但可能冗余的信息。我们将所提出的**空间增强CBAM（Spatial-CBAM）**注意力模块应用于 $\mathbf{F}_{\text{cat}}$，沿通道和空间两个维度进行自适应重校准。注意力精炼后的特征通过 $1 \times 1$ 卷积投影到紧凑的256维表征，并以0.1的概率进行Dropout：

$$\mathbf{F}_{\text{aspp}} = \text{Dropout}\left(\delta\left(\text{BN}\left(\mathbf{W}_{\text{proj}} * \text{Spatial-CBAM}(\mathbf{F}_{\text{cat}})\right)\right)\right) \in \mathbb{R}^{256 \times \frac{H}{16} \times \frac{W}{16}} \tag{8}$$

其中 $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{256 \times 1280 \times 1 \times 1}$ 为投影卷积核。

**跨层级特征融合。** 随后，$\mathbf{F}_{\text{aspp}}$ 通过所提出的**浅层-高级特征融合（Shallow-Advanced Feature Fusion, SAFF）**模块与骨干网络中层特征 $\mathbf{F}_{\text{mid}}$ 进行融合，生成语义丰富且空间精确的中间表征：

$$\mathbf{F}_{\text{fused}} = \text{SAFF}(\mathbf{F}_{\text{mid}}, \mathbf{F}_{\text{aspp}}) \in \mathbb{R}^{256 \times \frac{H}{8} \times \frac{W}{8}} \tag{9}$$

**解码器。** 融合特征 $\mathbf{F}_{\text{fused}}$ 经双线性上采样至低层特征分辨率（$\frac{H}{4} \times \frac{W}{4}$），并与投影后的低层特征进行拼接。低层投影将 $\mathbf{F}_{\text{low}}$ 从 $C_l = 256$ 通道降至48通道，以平衡信息贡献，防止高通道数的低层特征主导融合表征：

$$\mathbf{F}_{\text{low}}' = \delta\left(\text{BN}\left(\mathbf{W}_{\text{low}} * \mathbf{F}_{\text{low}}\right)\right) \in \mathbb{R}^{48 \times \frac{H}{4} \times \frac{W}{4}}, \quad \mathbf{W}_{\text{low}} \in \mathbb{R}^{48 \times C_l \times 1 \times 1} \tag{10}$$

$$\mathbf{F}_{\text{dec}} = \text{Concat}\left[\mathbf{F}_{\text{low}}',\; \text{Up}_{2\times}(\mathbf{F}_{\text{fused}})\right] \in \mathbb{R}^{304 \times \frac{H}{4} \times \frac{W}{4}} \tag{11}$$

解码器随后通过两个连续的 $3 \times 3$ 卷积块（每个包含Conv-BN-ReLU）对拼接特征进行细化，最后通过 $1 \times 1$ 分类卷积生成分割预测：

$$\hat{\mathbf{F}}_{\text{dec}} = \delta\left(\text{BN}\left(\mathbf{W}_2 * \delta\left(\text{BN}\left(\mathbf{W}_1 * \mathbf{F}_{\text{dec}}\right)\right)\right)\right) \in \mathbb{R}^{256 \times \frac{H}{4} \times \frac{W}{4}} \tag{12}$$

$$\hat{\mathbf{Y}}_{\text{seg}} = \text{Up}_{4\times}\left(\mathbf{W}_{\text{cls}} * \hat{\mathbf{F}}_{\text{dec}}\right) \in \mathbb{R}^{K \times H \times W} \tag{13}$$

其中 $\mathbf{W}_1 \in \mathbb{R}^{256 \times 304 \times 3 \times 3}$，$\mathbf{W}_2 \in \mathbb{R}^{256 \times 256 \times 3 \times 3}$，$\mathbf{W}_{\text{cls}} \in \mathbb{R}^{K \times 256 \times 1 \times 1}$，$K$ 为语义类别数。$\text{Up}_{4\times}$ 表示 $4\times$ 双线性上采样，以恢复原始输入分辨率。

**边界辅助监督。** 与主分割路径并行，一个轻量级的**边界辅助头（Boundary Auxiliary Head）**从解码器特征 $\hat{\mathbf{F}}_{\text{dec}}$ 分支产生显式的边界预测 $\hat{\mathbf{Y}}_{\text{edge}} \in \mathbb{R}^{1 \times H \times W}$，在训练期间提供额外的几何监督信号以锐化目标边界。

**联合训练目标。** 总体训练损失整合了分割监督和边界监督：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}}(\hat{\mathbf{Y}}_{\text{seg}}, \mathbf{Y}) + \lambda \cdot \mathcal{L}_{\text{boundary}}(\hat{\mathbf{Y}}_{\text{edge}}, \mathbf{Y}_{\text{edge}}) \tag{14}$$

其中 $\mathcal{L}_{\text{seg}}$ 为语义分割的标准交叉熵损失，$\mathcal{L}_{\text{boundary}}$ 结合了二元交叉熵损失和Dice损失用于边界预测，$\mathbf{Y}$ 和 $\mathbf{Y}_{\text{edge}}$ 分别为分割和边界的真值标注，$\lambda$ 为平衡超参数。

**本文的三大核心创新——用于增强多尺度特征重校准的Spatial-CBAM注意力、用于跨层级语义-空间融合的SAFF模块、以及用于显式边缘监督的边界辅助分支——将在以下各小节中详细阐述。**

---

## 3.2 空间增强CBAM注意力模块

尽管ASPP模块通过并行的空洞卷积有效地捕获了多尺度上下文信息，但其拼接产生的特征 $\mathbf{F}_{\text{cat}} \in \mathbb{R}^{1280 \times \frac{H}{16} \times \frac{W}{16}}$ 不可避免地包含来自重叠感受野的分支间冗余信息和通道-空间噪声。标准的通道注意力机制（如SE-Net）仅建模通道间依赖而忽略空间结构；相反，纯空间注意力机制未能利用通道间的相关性。原始CBAM通过顺序应用通道注意力和空间注意力来解决这一问题，但我们认为，对于高维多尺度的ASPP特征而言，单次空间注意力不足以完全消解空间歧义——尤其是在包含重叠目标或复杂背景纹理的杂乱场景中。

为此，我们提出了**空间增强CBAM（Spatial-CBAM）**模块，该模块在标准CBAM的基础上增加了一个额外的空间精炼阶段，形成三阶段级联注意力流水线：（1）通道注意力，用于通道间重校准；（2）空间注意力，用于初始空间定位；（3）空间精炼，用于细粒度空间校正。详细结构如图3所示。

### 3.2.1 通道注意力

通道注意力子模块建模通道间依赖关系，以识别1280维ASPP特征中"什么"（what）是语义上重要的。给定输入特征张量 $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$（此处 $C = 1280$），我们首先通过两种互补的全局池化操作压缩空间维度。全局平均池化通过计算平均激活值来捕获整体的通道级统计量，而全局最大池化则捕获每个通道上最显著的激活值：

$$\mathbf{z}_{\text{avg}}^c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{F}(c, i, j), \quad c = 1, \ldots, C \tag{15}$$

$$\mathbf{z}_{\text{max}}^c = \max_{1 \leq i \leq H,\, 1 \leq j \leq W} \mathbf{F}(c, i, j), \quad c = 1, \ldots, C \tag{16}$$

得到两个通道描述符向量 $\mathbf{z}_{\text{avg}}, \mathbf{z}_{\text{max}} \in \mathbb{R}^{C \times 1 \times 1}$。两个描述符编码了互补的统计视角：$\mathbf{z}_{\text{avg}}$ 反映了通道间的整体能量分布，而 $\mathbf{z}_{\text{max}}$ 捕获峰值响应幅度，对判别性特征更为敏感。

两个描述符通过一个具有瓶颈架构的**共享两层MLP**进行前向传播，以建模非线性的通道间交互。瓶颈结构以缩减比 $r = 16$ 降低通道维度，在限制参数开销的同时保留表征能力：

$$\text{MLP}(\mathbf{z}) = \mathbf{W}_2 \cdot \delta\left(\mathbf{W}_1 \cdot \mathbf{z}\right) \tag{17}$$

其中 $\mathbf{W}_1 \in \mathbb{R}^{\lfloor C/r \rfloor \times C}$ 为压缩权重，将 $C$ 个通道投影到 $\lfloor C/r \rfloor = 80$ 个通道；$\mathbf{W}_2 \in \mathbb{R}^{C \times \lfloor C/r \rfloor}$ 为扩展权重，恢复原始通道维度；$\delta(\cdot)$ 为ReLU激活。隐藏维度强制设定下界为8以防止信息瓶颈：$\lfloor C/r \rfloor = \max(8, \lfloor C/r \rfloor)$。至关重要的是，**权重 $\mathbf{W}_1$ 和 $\mathbf{W}_2$ 在两条池化路径之间共享**，这不仅减少了参数量，还迫使MLP学习一个在不同空间聚合策略间具有泛化能力的统一通道重要性模型。

通道注意力图通过对两个MLP输出进行逐元素求和并施加sigmoid激活得到，产生位于 $[0, 1]$ 范围内的逐通道缩放因子：

$$\mathbf{M}_c = \sigma\left(\text{MLP}(\mathbf{z}_{\text{avg}}) + \text{MLP}(\mathbf{z}_{\text{max}})\right) \in \mathbb{R}^{C \times 1 \times 1} \tag{18}$$

$$\mathbf{F}' = \mathbf{M}_c \otimes \mathbf{F} \tag{19}$$

其中 $\sigma(\cdot)$ 为sigmoid函数，$\otimes$ 表示通道级广播乘法。$\mathbf{F}$ 的每个通道被 $\mathbf{M}_c$ 中对应的注意力权重进行缩放，有效地放大了信息通道（与目标语义类别相关的通道），同时抑制了不太相关的通道（捕获背景噪声或冗余多尺度信息的通道）。

### 3.2.2 空间注意力

通道注意力识别了"什么"特征是重要的，而空间注意力子模块则通过确定"哪里"（where）需要关注来进行互补。在通道精炼后的特征 $\mathbf{F}' \in \mathbb{R}^{C \times H \times W}$ 上，我们通过沿通道轴池化生成两个空间描述符图：

$$\mathbf{s}_{\text{avg}}(i, j) = \frac{1}{C} \sum_{c=1}^{C} \mathbf{F}'(c, i, j) \in \mathbb{R}^{1 \times H \times W} \tag{20}$$

$$\mathbf{s}_{\text{max}}(i, j) = \max_{1 \leq c \leq C} \mathbf{F}'(c, i, j) \in \mathbb{R}^{1 \times H \times W} \tag{21}$$

平均描述符 $\mathbf{s}_{\text{avg}}$ 反映了所有通道上逐像素的平均能量，指示空间上均匀激活的区域；而 $\mathbf{s}_{\text{max}}$ 捕获每个空间位置上的峰值响应，突出至少有一个通道强烈响应的像素——这些像素通常对应于语义显著区域或边界邻近区域。

两个空间描述符拼接形成2通道特征图，通过一个大的 $7 \times 7$ 卷积核（填充 $p = 3$ 以保持空间尺寸不变）卷积生成空间注意力图：

$$\mathbf{M}_s = \sigma\left(\mathbf{W}_s * [\mathbf{s}_{\text{avg}};\; \mathbf{s}_{\text{max}}]\right) \in \mathbb{R}^{1 \times H \times W}, \quad \mathbf{W}_s \in \mathbb{R}^{1 \times 2 \times 7 \times 7} \tag{22}$$

选择 $7 \times 7$ 而非更小的卷积核（如 $3 \times 3$）是经过深思熟虑的设计：它为每个像素提供了足够大的空间邻域（$7 \times 7 = 49$ 像素）来聚合上下文，以确定其相对空间重要性。鉴于特征图在 $\frac{H}{16} \times \frac{W}{16}$ 处已经是降低的空间分辨率，$7 \times 7$ 卷积核覆盖了特征图的相当比例。

空间加权特征通过广播乘法获得：

$$\mathbf{F}'' = \mathbf{M}_s \otimes \mathbf{F}' \tag{23}$$

其中每个空间位置 $(i, j)$ 处所有通道被同一标量 $\mathbf{M}_s(i, j) \in [0, 1]$ 缩放。

### 3.2.3 空间精炼

Spatial-CBAM相对于标准CBAM的**核心创新**在于引入了一个额外的**空间精炼**阶段。我们通过以下分析来阐述该设计的动机：ASPP拼接特征 $\mathbf{F}_{\text{cat}}$ 聚合了来自五个感受野差异巨大的并行分支（从 $1 \times 1$ 到全局）的信息。在通道注意力去除冗余通道、第一次空间注意力提供粗略空间先验之后，残余的空间噪声可能仍然存在——特别是在多个ASPP分支产生冲突空间激活的区域（例如，在目标边界附近，局部和全局上下文可能存在分歧）。单次空间注意力受限于单个 $7 \times 7$ 卷积和一个输出通道的有限容量，难以解决此类多尺度空间冲突。

因此，我们级联了第二个空间注意力模块，使用**独立的可学习参数集** $\mathbf{W}_{s'}$ 来进一步精炼空间响应。空间精炼阶段在已经过注意力处理的特征 $\mathbf{F}''$ 上执行与第一次空间注意力相同的架构操作：

$$\mathbf{s}_{\text{avg}}'(i, j) = \frac{1}{C} \sum_{c=1}^{C} \mathbf{F}''(c, i, j) \tag{24}$$

$$\mathbf{s}_{\text{max}}'(i, j) = \max_{1 \leq c \leq C} \mathbf{F}''(c, i, j) \tag{25}$$

$$\mathbf{M}_s' = \sigma\left(\mathbf{W}_{s'} * [\mathbf{s}_{\text{avg}}';\; \mathbf{s}_{\text{max}}']\right) \in \mathbb{R}^{1 \times H \times W}, \quad \mathbf{W}_{s'} \in \mathbb{R}^{1 \times 2 \times 7 \times 7} \tag{26}$$

$$\mathbf{F}_{\text{out}} = \mathbf{M}_s' \otimes \mathbf{F}'' \tag{27}$$

这种级联设计形成了**由粗到精**的空间聚焦机制。在第一遍中，空间注意力图 $\mathbf{M}_s$ 识别广泛相关的区域（例如前景与背景的区分）。在第二遍中，精炼注意力图 $\mathbf{M}_s'$ 在更干净的特征空间（经过通道和初始空间过滤之后）上操作，可以聚焦于更细粒度的区分——例如区分相邻目标边界、小尺度结构或第一遍可能欠关注的细长区域。值得注意的是，两个空间注意力模块共享相同的架构（$7 \times 7$ 卷积，$\text{padding} = 3$，无偏置），但学习**不同的参数** $\mathbf{W}_s \neq \mathbf{W}_{s'}$，使得每个阶段能够特化其空间门控行为。

完整的Spatial-CBAM变换可形式化为复合函数：

$$\mathbf{F}_{\text{out}} = \underbrace{\text{SA}_{\theta_{s'}}}_{\text{空间精炼}} \circ \underbrace{\text{SA}_{\theta_s}}_{\text{空间注意力}} \circ \underbrace{\text{CA}_{\theta_c}}_{\text{通道注意力}} (\mathbf{F}) \tag{28}$$

其中 $\theta_c = \{\mathbf{W}_1, \mathbf{W}_2\}$，$\theta_s = \{\mathbf{W}_s\}$，$\theta_{s'} = \{\mathbf{W}_{s'}\}$ 分别表示各阶段的可学习参数。Spatial-CBAM的总额外参数量为：

$$|\theta_{\text{S-CBAM}}| = \underbrace{2 \times C \times \lfloor C/r \rfloor}_{\text{通道MLP}} + \underbrace{2 \times (2 \times 7 \times 7)}_{\text{两个空间卷积}} = 2 \times 1280 \times 80 + 2 \times 98 = 204{,}996 \tag{29}$$

相对于骨干网络参数而言，这仅占不到0.5%的可忽略开销，但如消融实验所示，能带来显著的性能提升。

**为了进一步弥合经注意力重校准的高层特征与空间丰富的中层特征之间的语义鸿沟，我们引入了一个专用的跨层级融合模块，如下节所述。**

---

## 3.3 浅层-高级特征融合（SAFF）模块

### 3.3.1 动机

在标准DeepLabV3+中，编码器到解码器的路径仅创建一个跳跃连接：高层ASPP特征（$1/16$ 尺度）经上采样后与低层特征（$1/4$ 尺度）在解码器中进行拼接。这种设计引入了显著的语义鸿沟——来自第一残差阶段的低层特征捕获了细粒度的空间细节（边缘、纹理），但缺乏语义理解；而ASPP输出编码了丰富的语义信息，但空间分辨率粗糙。来自第二残差阶段的中间层特征（$1/8$ 尺度）自然地弥合了这两个极端，编码了结构和几何模式（角点、目标部件、中尺度纹理），然而在标准流水线中被完全丢弃。

我们观察到，直接拼接高层和低层特征迫使解码器在单步中调和 $4\times$ 的分辨率差距和巨大的通道维度不匹配（256 vs. 48），给仅有的两层解码器卷积带来了过重的负担。为解决这一问题，我们提出了**浅层-高级特征融合（SAFF）**模块，在ASPP输出和骨干网络中层特征之间引入一个显式的 $1/8$ 尺度中间融合步骤。SAFF的架构如图4所示。

### 3.3.2 架构与公式

SAFF模块接收两个输入：来自骨干网络的中层特征 $\mathbf{F}_{\text{mid}} \in \mathbb{R}^{C_m \times \frac{H}{8} \times \frac{W}{8}}$ 和经注意力精炼的ASPP输出 $\mathbf{F}_{\text{aspp}} \in \mathbb{R}^{256 \times \frac{H}{16} \times \frac{W}{16}}$。

**中层投影路径。** 中层特征通过 $1 \times 1$ 卷积从其原始通道维度 $C_m = 512$ 投影到统一的256维，随后进行批归一化和ReLU激活：

$$\mathbf{F}_{\text{mid}}' = \delta\left(\text{BN}\left(\mathbf{W}_{\text{f1}} * \mathbf{F}_{\text{mid}}\right)\right) \in \mathbb{R}^{256 \times \frac{H}{8} \times \frac{W}{8}}, \quad \mathbf{W}_{\text{f1}} \in \mathbb{R}^{256 \times C_m \times 1 \times 1} \tag{30}$$

该 $1 \times 1$ 投影具有双重作用：（1）通道维度对齐以匹配ASPP输出；（2）通道级特征选择，丢弃低层噪声的同时保留来自中层表征的结构性信息激活。

**ASPP精炼路径。** ASPP输出位于 $\frac{H}{16} \times \frac{W}{16}$ 分辨率，需要在空间上与 $\frac{H}{8} \times \frac{W}{8}$ 的中层特征对齐。我们首先进行 $2\times$ 双线性插值上采样以提升空间维度：

$$\tilde{\mathbf{F}}_{\text{aspp}} = \text{Upsample}_{2\times}(\mathbf{F}_{\text{aspp}}) \in \mathbb{R}^{256 \times \frac{H}{8} \times \frac{W}{8}} \tag{31}$$

然而，朴素的双线性上采样会引入空间模糊，且无法将语义特征自适应地调整到更精细的空间网格上。因此，我们在上采样之后应用一个 $3 \times 3$ 空洞卷积（空洞率 $d = 3$，填充 $p = 3$ 以保持空间维度不变）作为后上采样精炼：

$$\mathbf{F}_{\text{aspp}}' = \delta\left(\text{BN}\left(\mathbf{W}_{\text{f2}} *_3 \tilde{\mathbf{F}}_{\text{aspp}}\right)\right) \in \mathbb{R}^{256 \times \frac{H}{8} \times \frac{W}{8}}, \quad \mathbf{W}_{\text{f2}} \in \mathbb{R}^{256 \times 256 \times 3 \times 3} \tag{32}$$

其中 $*_3$ 表示空洞率为 $d = 3$ 的卷积。该空洞卷积的有效感受野为：

$$k_{\text{eff}} = k + (k - 1)(d - 1) = 3 + (3 - 1)(3 - 1) = 7 \tag{33}$$

这个 $7 \times 7$ 的有效感受野具有至关重要的双重作用。第一，它提供了足够大的空间上下文来重新调整插值后的特征值至其新的空间位置，补偿双线性上采样的不精确性。第二，空洞操作保持了来自原始ASPP输出的全局语义上下文——在 $1/8$ 尺度上的每个 $7 \times 7$ 有效窗口对应原始图像中 $56 \times 56$ 像素的区域，确保了上采样特征在更高分辨率下仍保持广泛的上下文感知能力。

**加法融合与输出投影。** 两个在空间和通道上已对齐的特征通过逐元素加法进行融合：

$$\mathbf{F}_{\text{sum}} = \mathbf{F}_{\text{mid}}' + \mathbf{F}_{\text{aspp}}' \in \mathbb{R}^{256 \times \frac{H}{8} \times \frac{W}{8}} \tag{34}$$

我们采用逐元素加法而非拼接，基于以下设计考量：（i）加法在共享潜在空间中强制特征对齐，鼓励网络从两个层级学习互补表征——中层分支侧重空间精度，而ASPP分支贡献语义上下文；（ii）加法不引入通道维度扩展，保持256通道宽度恒定，避免后续的通道降维操作；（iii）加法公式可被解释为残差连接，其中 $\mathbf{F}_{\text{aspp}}'$ 作为对空间丰富的 $\mathbf{F}_{\text{mid}}'$ 的语义"校正"。

最后，通过 $1 \times 1$ 卷积配合BN和ReLU作为非线性输出投影：

$$\mathbf{F}_{\text{fused}} = \delta\left(\text{BN}\left(\mathbf{W}_{\text{out}} * \mathbf{F}_{\text{sum}}\right)\right) \in \mathbb{R}^{256 \times \frac{H}{8} \times \frac{W}{8}}, \quad \mathbf{W}_{\text{out}} \in \mathbb{R}^{256 \times 256 \times 1 \times 1} \tag{35}$$

该投影实现了非线性特征重组，选择性地强调最具判别力的融合特征，同时抑制两条路径之间的残余对齐误差。SAFF的完整变换可以紧凑形式写为：

$$\text{SAFF}(\mathbf{F}_{\text{mid}}, \mathbf{F}_{\text{aspp}}) = g_{\text{out}}\left(g_{\text{f1}}(\mathbf{F}_{\text{mid}}) + g_{\text{f2}}\left(\text{Up}_{2\times}(\mathbf{F}_{\text{aspp}})\right)\right) \tag{36}$$

其中 $g_{\text{f1}}$、$g_{\text{f2}}$ 和 $g_{\text{out}}$ 分别表示三个Conv-BN-ReLU模块。

### 3.3.3 三级特征层次

通过在ASPP编码器和解码器之间插入SAFF模块，我们建立了一个渐进式的三级特征融合层次。总体解码器数据流可形式化为：

$$\mathbf{F}_{\text{fused}} = \text{SAFF}\left(\mathbf{F}_{\text{mid}},\; \text{Proj}\left(\text{S-CBAM}\left(\text{ASPP}(\mathbf{F}_{\text{high}})\right)\right)\right) \tag{37}$$

$$\mathbf{F}_{\text{dec}} = \text{Concat}\left[\text{Proj}_{\text{low}}(\mathbf{F}_{\text{low}}),\; \text{Up}_{2\times}(\mathbf{F}_{\text{fused}})\right] \tag{38}$$

$$\hat{\mathbf{Y}}_{\text{seg}} = \text{Up}_{4\times}\left(\text{Classifier}(\mathbf{F}_{\text{dec}})\right) \tag{39}$$

这构建了一条平滑的信息流：语义信息在 $1/16$ 尺度注入（ASPP），与结构信息在 $1/8$ 尺度融合（SAFF），再与细粒度空间细节在 $1/4$ 尺度结合（解码器）。每个阶段仅缩小 $2\times$ 的分辨率差距，使得解码器的任务与原始DeepLabV3+中 $4\times$ 差距相比大大简化。

**通过为中层特征建立显式的信息通路，SAFF模块构建了三级特征层次，逐步整合递增语义层级的信息。为进一步改善最终分割输出中的边界描绘精度，我们引入了边界辅助监督分支，如下节所述。**

---

## 3.4 边界辅助分支

### 3.4.1 动机

语义分割网络在目标边界附近通常会产生模糊或不一致的预测。这一现象源于多个因素：（i）编码器中的池化操作和步幅卷积逐步侵蚀了细粒度的边界信息；（ii）标准交叉熵损失对所有像素一视同仁，不考虑其与边界的空间邻近性，为数量上处于少数的边界像素提供了不足的梯度信号；（iii）解码器中的双线性上采样引入空间平滑效应，进一步降低了边界清晰度。尽管我们的SAFF模块通过中层空间线索丰富了特征表征，但它在 $1/8$ 分辨率上运行，无法完全恢复精确分割所需的像素级边界精度。

为直接应对上述挑战，我们引入了一个轻量级的**边界辅助分支（Boundary Auxiliary Branch）**，为边界邻近像素提供显式的监督信号。核心思想在于：通过强制一个平行的解码器分支预测目标边界，共享的特征骨干被鼓励在整个网络中保留和增强边缘判别性表征——不仅有利于边界预测，还通过梯度反向传播惠及主分割任务。模块架构和训练流程如图5所示。

### 3.4.2 边界辅助头

边界头在解码器特征 $\hat{\mathbf{F}}_{\text{dec}} \in \mathbb{R}^{C_d \times \frac{H}{4} \times \frac{W}{4}}$ 上运行（使用传统解码器时 $C_d = 304$，使用改进解码器时 $C_d = 256$）。该头部由一个紧凑的三层全卷积网络组成，通道维度逐步递减：

**第一层（特征压缩）：** $3 \times 3$ 卷积将高维解码器特征压缩为紧凑的64通道表征，同时提取局部的边缘导向模式：

$$\mathbf{H}_1 = \delta\left(\text{BN}\left(\mathbf{W}_{\text{b1}} * \hat{\mathbf{F}}_{\text{dec}}\right)\right) \in \mathbb{R}^{64 \times \frac{H}{4} \times \frac{W}{4}}, \quad \mathbf{W}_{\text{b1}} \in \mathbb{R}^{64 \times C_d \times 3 \times 3} \tag{40}$$

**第二层（边界精炼）：** 第二个 $3 \times 3$ 卷积进一步精炼边界表征，提供额外的非线性容量以区分真实边界与纹理边缘：

$$\mathbf{H}_2 = \delta\left(\text{BN}\left(\mathbf{W}_{\text{b2}} * \mathbf{H}_1\right)\right) \in \mathbb{R}^{64 \times \frac{H}{4} \times \frac{W}{4}}, \quad \mathbf{W}_{\text{b2}} \in \mathbb{R}^{64 \times 64 \times 3 \times 3} \tag{41}$$

两个 $3 \times 3$ 卷积层提供了 $5 \times 5$ 像素的组合有效感受野（在 $1/4$ 分辨率下），对应原始图像中 $20 \times 20$ 像素的区域。这足以捕获局部边界上下文，同时保持头部的轻量性。

**第三层（逻辑值预测）：** 带偏置的 $1 \times 1$ 卷积输出单通道边界逻辑值图：

$$\mathbf{E} = \mathbf{W}_{\text{b3}} * \mathbf{H}_2 + \mathbf{b}_{\text{b3}} \in \mathbb{R}^{1 \times \frac{H}{4} \times \frac{W}{4}}, \quad \mathbf{W}_{\text{b3}} \in \mathbb{R}^{1 \times 64 \times 1 \times 1} \tag{42}$$

注意，与前面无偏置的层不同，最后一层使用了偏置项 $\mathbf{b}_{\text{b3}} \in \mathbb{R}$，使得网络能够学习一个初始的边界/非边界偏移量，帮助sigmoid函数从训练伊始就在合适的范围内运行。

边缘逻辑值图经双线性上采样恢复到原始分辨率：

$$\hat{\mathbf{Y}}_{\text{edge}} = \text{Up}_{4\times}(\mathbf{E}) \in \mathbb{R}^{1 \times H \times W} \tag{43}$$

边界头的完整变换可紧凑写为：

$$\hat{\mathbf{Y}}_{\text{edge}} = \text{Up}_{4\times}\left(\mathbf{W}_{\text{b3}} * \delta\left(\text{BN}\left(\mathbf{W}_{\text{b2}} * \delta\left(\text{BN}\left(\mathbf{W}_{\text{b1}} * \hat{\mathbf{F}}_{\text{dec}}\right)\right)\right)\right) + \mathbf{b}_{\text{b3}}\right) \tag{44}$$

边界头的总参数量为：

$$|\theta_{\text{boundary}}| = C_d \times 64 \times 9 + 64 \times 64 \times 9 + 64 \times 1 + 1 \approx 211\text{K} \tag{45}$$

不足骨干网络参数的0.5%，是一个极其高效的添加。

### 3.4.3 形态学边界标签生成

边界监督的一个关键方面是真值边界标签的质量和形式。与依赖预计算边缘检测算法（如Canny或Sobel，它们是类别无关的且对纹理边缘敏感）不同，我们通过**类别感知的形态学膨胀-腐蚀方案**直接从语义真值标签生成边界监督目标。

对于真值语义掩码 $\mathbf{Y} \in \{0, 1, \ldots, K-1\}^{H \times W}$，我们首先提取每个语义类别 $c$ 的二值掩码：

$$\mathbf{Y}_c(i, j) = \mathbb{1}[\mathbf{Y}(i, j) = c] \in \{0, 1\}^{H \times W} \tag{46}$$

对于每个类别二值掩码，我们使用半径为 $\rho$ 的方形结构元素进行形态学膨胀和腐蚀操作来计算边界带：

$$\text{Dilate}(\mathbf{Y}_c, \rho)(i, j) = \max_{|p-i| \leq \rho, |q-j| \leq \rho} \mathbf{Y}_c(p, q) \tag{47}$$

$$\text{Erode}(\mathbf{Y}_c, \rho)(i, j) = \min_{|p-i| \leq \rho, |q-j| \leq \rho} \mathbf{Y}_c(p, q) \tag{48}$$

在实现中，膨胀操作通过核大小为 $k = 2\rho + 1$ 的最大池化实现，腐蚀操作通过对取反掩码进行取反最大池化实现，从而避免了对显式形态学操作库的依赖：

$$\text{Dilate}(\mathbf{Y}_c, \rho) = \text{MaxPool}_{2\rho+1}(\mathbf{Y}_c) \tag{49}$$

$$\text{Erode}(\mathbf{Y}_c, \rho) = -\text{MaxPool}_{2\rho+1}(-\mathbf{Y}_c) \tag{50}$$

每类边界定义为位于膨胀区域内但不在腐蚀区域内的像素集合——即横跨每个类别边界、宽度为 $2\rho$ 的条带：

$$\mathbf{B}_c = \mathbb{1}\left[\text{Dilate}(\mathbf{Y}_c, \rho) - \text{Erode}(\mathbf{Y}_c, \rho) > 0\right] \in \{0, 1\}^{H \times W} \tag{51}$$

最终的边界标签是所有类别边界带的并集，并进一步通过有效像素区域（排除忽略标签像素）进行掩码：

$$\mathbf{Y}_{\text{edge}}(i, j) = \mathbb{1}\left[\max_{c \in \{0, \ldots, K-1\}} \mathbf{B}_c(i, j) > 0\right] \cdot \mathbb{1}[\mathbf{Y}(i, j) \neq \text{ignore}] \tag{52}$$

默认设置 $\rho = 3$ 时，生成的边界带宽度为6像素（真实边界两侧各3像素），这提供了两个关键优势：（1）更宽的条带包含更多像素，与单像素细线边界标注相比提供了更丰富的梯度信号；（2）条带自然地考虑了标注噪声和真值标注中的微小对齐误差，提高了训练稳定性。

### 3.4.4 边界损失函数设计

边界预测本质上是一个严重类别不平衡的二分类任务——边界像素通常仅占图像面积的5%–15%。为应对这一挑战，我们设计了结合两个互补目标的复合边界损失 $\mathcal{L}_{\text{boundary}}$。

**二元交叉熵（BCE）损失** 提供逐像素监督，具有稳定的梯度特性：

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N_{\text{valid}}}\sum_{i \in \Omega_{\text{valid}}}\left[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\right] \tag{53}$$

其中 $\hat{p}_i = \sigma(\hat{e}_i)$ 为通过对原始逻辑值 $\hat{e}_i$ 施加sigmoid得到的预测边界概率，$y_i \in \{0, 1\}$ 为边界真值，$\Omega_{\text{valid}}$ 为有效像素集合（排除忽略标签区域），$N_{\text{valid}} = |\Omega_{\text{valid}}|$。

**Dice损失** 直接优化预测与真值边界区域之间的集合级重叠度，提供对类别不平衡鲁棒的优化目标：

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_{i \in \Omega_{\text{valid}}} \hat{p}_i \cdot y_i + \epsilon}{\sum_{i \in \Omega_{\text{valid}}} \hat{p}_i + \sum_{i \in \Omega_{\text{valid}}} y_i + \epsilon} \tag{54}$$

其中 $\epsilon = 1 \times 10^{-5}$ 为防止除零的平滑常数。Dice损失通过相对于预测和真值边界面积（而非总像素数）进行归一化，有效地提升了少数边界像素的贡献权重。这与被多数非边界类别主导的BCE损失形成互补。

复合边界损失为：

$$\mathcal{L}_{\text{boundary}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}} \tag{55}$$

### 3.4.5 联合优化

总体训练目标将主分割损失与加权边界损失相整合：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\hat{\mathbf{Y}}_{\text{seg}}, \mathbf{Y}) + \lambda \cdot \mathcal{L}_{\text{boundary}}(\hat{\mathbf{Y}}_{\text{edge}}, \mathbf{Y}_{\text{edge}}) \tag{56}$$

其中 $\mathcal{L}_{\text{CE}}$ 为 $K$ 类语义分割的标准逐像素交叉熵损失：

$$\mathcal{L}_{\text{CE}} = -\frac{1}{N_{\text{valid}}}\sum_{i \in \Omega_{\text{valid}}} \sum_{k=1}^{K} \mathbb{1}[\mathbf{Y}(i) = k] \log \frac{\exp(\hat{\mathbf{Y}}_{\text{seg}}^k(i))}{\sum_{k'=1}^{K} \exp(\hat{\mathbf{Y}}_{\text{seg}}^{k'}(i))} \tag{57}$$

$\lambda$ 为平衡超参数，控制边界监督的相对贡献。来自 $\mathcal{L}_{\text{boundary}}$ 的梯度通过边界头反向传播到共享的解码器特征 $\hat{\mathbf{F}}_{\text{dec}}$，进一步流入SAFF模块、ASPP编码器直至骨干网络。这一梯度通路产生了有益的正则化效应：边界损失显式地鼓励整个网络在编码器-解码器流水线的每个阶段保留边缘判别性特征，与主要驱动类别区分的逐像素交叉熵损失形成互补。

边界分支的一个重要实用特性是其**推理时的灵活性**：由于边界头是一个独立模块，不影响分割头的前向计算，因此在部署时可以选择性地丢弃，而不影响分割精度。这使得边界分支成为一个**仅在训练时使用的正则化器**，在不增加任何推理开销的前提下改善所学特征表征。或者，当应用需要显式的边缘图（如实例轮廓提取、全景分割后处理）时，边界输出可作为互补预测结果。

**三个核心组件——Spatial-CBAM、SAFF和边界辅助分支——之间的协同效应构成了一个在分割框架不同维度上运作的综合增强流水线。Spatial-CBAM在通道和空间两个维度上重校准多尺度ASPP特征，抑制冗余并突出判别性模式；SAFF桥接多层级编码器特征，缩小编码器-解码器路径中的语义鸿沟；边界辅助分支通过梯度级正则化提供显式的几何监督，锐化决策边界。三个组件协同工作，在保持计算效率的同时显著提升了分割精度，尤其是在目标边界和小尺度结构处的表现。**

---

*图2：所提SA-BNet框架的总体架构。骨干网络提取三个层级的特征（$\mathbf{F}_{\text{low}}$、$\mathbf{F}_{\text{mid}}$、$\mathbf{F}_{\text{high}}$），依次经过带Spatial-CBAM注意力的ASPP处理、SAFF模块融合，以及带边界辅助监督的解码器。*

*图3：空间增强CBAM（Spatial-CBAM）注意力模块的架构。三阶段级联（通道注意力 → 空间注意力 → 空间精炼）从通道级重要性到粗粒度空间定位再到细粒度空间校正，逐步重校准多尺度特征。*

*图4：浅层-高级特征融合（SAFF）模块的架构。中层特征 $\mathbf{F}_{\text{mid}}$ 通过 $1 \times 1$ 卷积投影，ASPP输出经上采样和空洞卷积精炼。两个对齐的特征通过加法融合并投影生成 $\mathbf{F}_{\text{fused}}$。*

*图5：边界辅助分支的架构和训练流程。（A）边界头通过轻量级卷积网络处理解码器特征以预测边缘逻辑值；（B）真值边界标签通过类别感知的形态学操作从语义标签生成；（C）联合损失将分割交叉熵与加权边界BCE+Dice损失相结合。*
