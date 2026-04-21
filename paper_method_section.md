# 3. Method

## 3.1 Overall Architecture

We propose an enhanced semantic segmentation framework built upon DeepLabV3+, termed **SA-BNet** (Spatial-Attention Boundary-aware Network), which jointly addresses multi-scale context aggregation, attention-guided feature recalibration, cross-level feature fusion, and explicit boundary supervision within a unified end-to-end architecture. The overall pipeline is illustrated in Fig. 2.

**Encoder.** Given an input image $\mathbf{I} \in \mathbb{R}^{3 \times H \times W}$, we employ a ResNet-101 backbone pre-trained on ImageNet as the feature encoder. Following standard practice in dense prediction tasks, we adopt an output stride (OS) of 16 by replacing the stride-2 convolution in the last residual stage with dilated convolutions, thereby preserving higher spatial resolution without sacrificing the depth of learned representations. Concretely, the last two residual blocks of layer3 and layer4 employ dilation rates of $\{1, 2\}$ and $\{1, 2, 4\}$, respectively, when OS = 8; under OS = 16, only layer4 is dilated.

The backbone produces a hierarchy of feature maps at three distinct semantic levels through intermediate layer extraction:

$$\mathbf{F}_{\text{low}} = \phi_{\text{layer1}}(\mathbf{I}) \in \mathbb{R}^{C_l \times \frac{H}{4} \times \frac{W}{4}} \tag{1}$$

$$\mathbf{F}_{\text{mid}} = \phi_{\text{layer2}}(\mathbf{I}) \in \mathbb{R}^{C_m \times \frac{H}{8} \times \frac{W}{8}} \tag{2}$$

$$\mathbf{F}_{\text{high}} = \phi_{\text{layer4}}(\mathbf{I}) \in \mathbb{R}^{C_h \times \frac{H}{16} \times \frac{W}{16}} \tag{3}$$

where $C_l = 256$, $C_m = 512$, and $C_h = 2048$ for ResNet-101. $\phi_{\text{layer}k}$ denotes the composite mapping from the input image to the output of the $k$-th residual stage. Unlike the vanilla DeepLabV3+, which only utilizes the low-level feature $\mathbf{F}_{\text{low}}$ and the high-level feature $\mathbf{F}_{\text{high}}$, our framework explicitly extracts the mid-level feature $\mathbf{F}_{\text{mid}}$ from the second residual stage. This intermediate representation captures structural and geometric patterns—such as edges, corners, and texture boundaries—that lie between low-level visual cues and high-level semantic abstractions, providing a crucial information bridge for subsequent fusion.

**Multi-Scale Context Aggregation.** The high-level feature $\mathbf{F}_{\text{high}}$ is fed into the Atrous Spatial Pyramid Pooling (ASPP) module to capture multi-scale contextual information. The ASPP module consists of five parallel branches operating on $\mathbf{F}_{\text{high}}$ simultaneously. Each branch $b_k$ extracts features at a different scale:

$$b_0(\mathbf{F}_{\text{high}}) = \delta\left(\text{BN}\left(\mathbf{W}_0 * \mathbf{F}_{\text{high}}\right)\right), \quad \mathbf{W}_0 \in \mathbb{R}^{256 \times C_h \times 1 \times 1} \tag{4}$$

$$b_k(\mathbf{F}_{\text{high}}) = \delta\left(\text{BN}\left(\mathbf{W}_k *_{r_k} \mathbf{F}_{\text{high}}\right)\right), \quad k = 1, 2, 3 \tag{5}$$

$$b_4(\mathbf{F}_{\text{high}}) = \text{Up}\left(\delta\left(\text{BN}\left(\mathbf{W}_4 * \text{GAP}(\mathbf{F}_{\text{high}})\right)\right)\right) \tag{6}$$

where $*_{r_k}$ denotes the $3 \times 3$ atrous convolution with dilation rate $r_k \in \{6, 12, 18\}$, $\delta(\cdot)$ represents the ReLU activation, $\text{BN}(\cdot)$ is batch normalization, $\text{GAP}(\cdot)$ is adaptive global average pooling that compresses spatial dimensions to $1 \times 1$, and $\text{Up}(\cdot)$ upsamples the pooled representation back to the input spatial size via bilinear interpolation. $\mathbf{W}_0$ through $\mathbf{W}_4$ are learnable convolutional kernels, each projecting to 256 output channels. The five branch outputs are concatenated along the channel dimension:

$$\mathbf{F}_{\text{cat}} = \text{Concat}\left[b_0(\mathbf{F}_{\text{high}}),\; b_1(\mathbf{F}_{\text{high}}),\; b_2(\mathbf{F}_{\text{high}}),\; b_3(\mathbf{F}_{\text{high}}),\; b_4(\mathbf{F}_{\text{high}})\right] \in \mathbb{R}^{1280 \times \frac{H}{16} \times \frac{W}{16}} \tag{7}$$

The effective receptive fields of the three atrous branches are $3 + 2(r_k - 1)$ for $r_k \in \{6, 12, 18\}$, yielding effective kernel sizes of $\{13, 25, 37\}$ pixels, respectively. Together with the $1 \times 1$ local branch and the global pooling branch, the ASPP module covers receptive fields ranging from a single pixel to the entire feature map.

**Attention-Guided Recalibration.** The concatenated multi-scale feature $\mathbf{F}_{\text{cat}}$ contains abundant yet potentially redundant information from overlapping receptive fields. We apply the proposed **Spatial-Enhanced CBAM (Spatial-CBAM)** attention module to adaptively recalibrate $\mathbf{F}_{\text{cat}}$ along both channel and spatial dimensions. The attention-refined feature is then projected to a compact 256-dimensional representation via a $1 \times 1$ convolution followed by dropout with rate 0.1:

$$\mathbf{F}_{\text{aspp}} = \text{Dropout}\left(\delta\left(\text{BN}\left(\mathbf{W}_{\text{proj}} * \text{Spatial-CBAM}(\mathbf{F}_{\text{cat}})\right)\right)\right) \in \mathbb{R}^{256 \times \frac{H}{16} \times \frac{W}{16}} \tag{8}$$

where $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{256 \times 1280 \times 1 \times 1}$ is the projection kernel.

**Cross-Level Feature Fusion.** Subsequently, $\mathbf{F}_{\text{aspp}}$ is combined with the mid-level backbone feature $\mathbf{F}_{\text{mid}}$ through the proposed **Shallow-Advanced Feature Fusion (SAFF)** module, producing a semantically enriched yet spatially precise intermediate representation:

$$\mathbf{F}_{\text{fused}} = \text{SAFF}(\mathbf{F}_{\text{mid}}, \mathbf{F}_{\text{aspp}}) \in \mathbb{R}^{256 \times \frac{H}{8} \times \frac{W}{8}} \tag{9}$$

**Decoder.** The fused feature $\mathbf{F}_{\text{fused}}$ is bilinearly upsampled to the low-level feature resolution ($\frac{H}{4} \times \frac{W}{4}$) and concatenated with the projected low-level feature. The low-level projection reduces $\mathbf{F}_{\text{low}}$ from $C_l = 256$ channels to 48 channels to balance the information contribution and prevent the high-channel-count low-level feature from dominating the fused representation:

$$\mathbf{F}_{\text{low}}' = \delta\left(\text{BN}\left(\mathbf{W}_{\text{low}} * \mathbf{F}_{\text{low}}\right)\right) \in \mathbb{R}^{48 \times \frac{H}{4} \times \frac{W}{4}}, \quad \mathbf{W}_{\text{low}} \in \mathbb{R}^{48 \times C_l \times 1 \times 1} \tag{10}$$

$$\mathbf{F}_{\text{dec}} = \text{Concat}\left[\mathbf{F}_{\text{low}}',\; \text{Up}_{2\times}(\mathbf{F}_{\text{fused}})\right] \in \mathbb{R}^{304 \times \frac{H}{4} \times \frac{W}{4}} \tag{11}$$

The decoder then applies two successive $3 \times 3$ convolution blocks (each comprising Conv-BN-ReLU) to refine the concatenated feature, followed by a $1 \times 1$ classification convolution:

$$\hat{\mathbf{F}}_{\text{dec}} = \delta\left(\text{BN}\left(\mathbf{W}_2 * \delta\left(\text{BN}\left(\mathbf{W}_1 * \mathbf{F}_{\text{dec}}\right)\right)\right)\right) \in \mathbb{R}^{256 \times \frac{H}{4} \times \frac{W}{4}} \tag{12}$$

$$\hat{\mathbf{Y}}_{\text{seg}} = \text{Up}_{4\times}\left(\mathbf{W}_{\text{cls}} * \hat{\mathbf{F}}_{\text{dec}}\right) \in \mathbb{R}^{K \times H \times W} \tag{13}$$

where $\mathbf{W}_1 \in \mathbb{R}^{256 \times 304 \times 3 \times 3}$, $\mathbf{W}_2 \in \mathbb{R}^{256 \times 256 \times 3 \times 3}$, $\mathbf{W}_{\text{cls}} \in \mathbb{R}^{K \times 256 \times 1 \times 1}$, and $K$ is the number of semantic classes. $\text{Up}_{4\times}$ denotes $4\times$ bilinear upsampling to recover the original input resolution.

**Boundary Auxiliary Supervision.** In parallel to the main segmentation path, a lightweight **Boundary Auxiliary Head** branches off from the decoder feature $\hat{\mathbf{F}}_{\text{dec}}$ to produce an explicit boundary prediction $\hat{\mathbf{Y}}_{\text{edge}} \in \mathbb{R}^{1 \times H \times W}$, which provides additional geometric supervisory signals to sharpen object boundaries during training.

**Joint Training Objective.** The total training loss integrates segmentation and boundary supervision:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}}(\hat{\mathbf{Y}}_{\text{seg}}, \mathbf{Y}) + \lambda \cdot \mathcal{L}_{\text{boundary}}(\hat{\mathbf{Y}}_{\text{edge}}, \mathbf{Y}_{\text{edge}}) \tag{14}$$

where $\mathcal{L}_{\text{seg}}$ is the standard cross-entropy loss for semantic segmentation, $\mathcal{L}_{\text{boundary}}$ combines binary cross-entropy and Dice loss for boundary prediction, $\mathbf{Y}$ and $\mathbf{Y}_{\text{edge}}$ are the segmentation and boundary ground truth, respectively, and $\lambda$ is a balancing hyperparameter.

**The three core innovations—Spatial-CBAM attention for enhanced multi-scale feature recalibration, SAFF for cross-level semantic-spatial fusion, and the boundary auxiliary branch for explicit edge supervision—are detailed in the following subsections.**

---

## 3.2 Spatial-Enhanced CBAM Attention Module

While the ASPP module effectively captures multi-scale contextual information through parallel atrous convolutions, the resulting concatenated feature $\mathbf{F}_{\text{cat}} \in \mathbb{R}^{1280 \times \frac{H}{16} \times \frac{W}{16}}$ inevitably contains inter-branch redundancy and channel-spatial noise from overlapping receptive fields. Standard channel attention mechanisms (e.g., SE-Net) only model channel-wise dependencies but ignore spatial structure; conversely, pure spatial attention mechanisms fail to exploit inter-channel correlations. The original CBAM addresses this by sequentially applying channel and spatial attention, but we argue that for the high-dimensional, multi-scale ASPP feature, a single-pass spatial attention is insufficient to fully resolve spatial ambiguities—particularly in cluttered scenes with overlapping objects or complex background textures.

To this end, we propose the **Spatial-Enhanced CBAM (Spatial-CBAM)** module, which extends the standard CBAM with an additional spatial refinement stage, forming a three-stage cascaded attention pipeline: (1) Channel Attention for inter-channel recalibration, (2) Spatial Attention for initial spatial localization, and (3) Spatial Refinement for fine-grained spatial correction. The detailed structure is illustrated in Fig. 3.

### 3.2.1 Channel Attention

The channel attention sub-module models inter-channel dependencies to identify "what" is semantically important across the 1280-dimensional ASPP feature. Given an input feature tensor $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$ (here $C = 1280$), we first compress the spatial dimensions using two complementary global pooling operations. Global average pooling captures the holistic channel-wise statistics by computing the mean activation, while global max pooling captures the most salient activations per channel:

$$\mathbf{z}_{\text{avg}}^c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{F}(c, i, j), \quad c = 1, \ldots, C \tag{15}$$

$$\mathbf{z}_{\text{max}}^c = \max_{1 \leq i \leq H,\, 1 \leq j \leq W} \mathbf{F}(c, i, j), \quad c = 1, \ldots, C \tag{16}$$

yielding two channel descriptor vectors $\mathbf{z}_{\text{avg}}, \mathbf{z}_{\text{max}} \in \mathbb{R}^{C \times 1 \times 1}$. Both descriptors encode complementary statistical perspectives: $\mathbf{z}_{\text{avg}}$ reflects the overall energy distribution across channels, while $\mathbf{z}_{\text{max}}$ captures the peak response magnitude and is more sensitive to discriminative features.

Both descriptors are forwarded through a shared two-layer MLP with a bottleneck architecture to model nonlinear inter-channel interactions. The bottleneck reduces the channel dimension by a ratio $r = 16$ to limit parameter overhead while retaining representational capacity:

$$\text{MLP}(\mathbf{z}) = \mathbf{W}_2 \cdot \delta\left(\mathbf{W}_1 \cdot \mathbf{z}\right) \tag{17}$$

where $\mathbf{W}_1 \in \mathbb{R}^{\lfloor C/r \rfloor \times C}$ is the compression weight that projects from $C$ channels to $\lfloor C/r \rfloor = 80$ channels, $\mathbf{W}_2 \in \mathbb{R}^{C \times \lfloor C/r \rfloor}$ is the expansion weight that restores the original channel dimension, and $\delta(\cdot)$ is the ReLU activation. A lower bound of 8 is enforced on the hidden dimension to prevent information bottleneck: $\lfloor C/r \rfloor = \max(8, \lfloor C/r \rfloor)$. Crucially, the weights $\mathbf{W}_1$ and $\mathbf{W}_2$ are shared between the two pooling paths, which not only reduces parameters but also forces the MLP to learn a unified channel importance model that generalizes across different spatial aggregation strategies.

The channel attention map is obtained by element-wise summing the two MLP outputs and applying a sigmoid activation to produce per-channel scaling factors in $[0, 1]$:

$$\mathbf{M}_c = \sigma\left(\text{MLP}(\mathbf{z}_{\text{avg}}) + \text{MLP}(\mathbf{z}_{\text{max}})\right) \in \mathbb{R}^{C \times 1 \times 1} \tag{18}$$

$$\mathbf{F}' = \mathbf{M}_c \otimes \mathbf{F} \tag{19}$$

where $\sigma(\cdot)$ is the sigmoid function and $\otimes$ denotes channel-wise broadcasting multiplication. Each channel of $\mathbf{F}$ is scaled by its corresponding attention weight in $\mathbf{M}_c$, effectively amplifying informative channels (those related to the target semantic classes) while suppressing less relevant ones (those capturing background noise or redundant multi-scale information).

### 3.2.2 Spatial Attention

While channel attention identifies "what" features are important, the spatial attention sub-module complements it by determining "where" to focus. Operating on the channel-refined feature $\mathbf{F}' \in \mathbb{R}^{C \times H \times W}$, we generate two spatial descriptor maps by pooling along the channel axis:

$$\mathbf{s}_{\text{avg}}(i, j) = \frac{1}{C} \sum_{c=1}^{C} \mathbf{F}'(c, i, j) \in \mathbb{R}^{1 \times H \times W} \tag{20}$$

$$\mathbf{s}_{\text{max}}(i, j) = \max_{1 \leq c \leq C} \mathbf{F}'(c, i, j) \in \mathbb{R}^{1 \times H \times W} \tag{21}$$

The average descriptor $\mathbf{s}_{\text{avg}}$ reflects the per-pixel mean energy across all channels, indicating spatially uniform activation regions, while $\mathbf{s}_{\text{max}}$ captures the peak response at each spatial location, highlighting pixels where at least one channel responds strongly—typically corresponding to semantically salient or boundary-adjacent regions.

The two spatial descriptors are concatenated to form a 2-channel feature map and convolved with a large $7 \times 7$ kernel (with padding $p = 3$ to preserve spatial size) to produce the spatial attention map:

$$\mathbf{M}_s = \sigma\left(\mathbf{W}_s * [\mathbf{s}_{\text{avg}};\; \mathbf{s}_{\text{max}}]\right) \in \mathbb{R}^{1 \times H \times W}, \quad \mathbf{W}_s \in \mathbb{R}^{1 \times 2 \times 7 \times 7} \tag{22}$$

The use of a $7 \times 7$ kernel, rather than smaller kernels such as $3 \times 3$, is deliberate: it provides each pixel with a sufficiently large spatial neighborhood ($7 \times 7 = 49$ pixels) to aggregate context for determining its relative spatial importance. Given the already-reduced spatial resolution at $\frac{H}{16} \times \frac{W}{16}$, a $7 \times 7$ kernel covers a meaningful proportion of the feature map.

The spatially weighted feature is obtained via broadcasting multiplication:

$$\mathbf{F}'' = \mathbf{M}_s \otimes \mathbf{F}' \tag{23}$$

where each spatial location $(i, j)$ in all channels is scaled by the same scalar $\mathbf{M}_s(i, j) \in [0, 1]$.

### 3.2.3 Spatial Refinement

The key innovation of Spatial-CBAM over standard CBAM lies in the introduction of an additional **spatial refinement** stage. We motivate this design through the following analysis: the ASPP concatenated feature $\mathbf{F}_{\text{cat}}$ aggregates information from five parallel branches with vastly different receptive fields (from $1 \times 1$ to global). After channel attention removes redundant channels and the first spatial attention provides a coarse spatial prior, residual spatial noise may still persist—particularly in regions where multiple ASPP branches produce conflicting spatial activations (e.g., near object boundaries where local and global context disagree). A single spatial attention pass, constrained to a single $7 \times 7$ convolution with one output channel, has limited capacity to resolve such multi-scale spatial conflicts.

Therefore, we cascade a second spatial attention module with an independent set of learned parameters ($\mathbf{W}_{s'}$) to further refine the spatial response. The spatial refinement stage applies the same architectural operations as the first spatial attention but on the already-attended feature $\mathbf{F}''$:

$$\mathbf{s}_{\text{avg}}'(i, j) = \frac{1}{C} \sum_{c=1}^{C} \mathbf{F}''(c, i, j) \tag{24}$$

$$\mathbf{s}_{\text{max}}'(i, j) = \max_{1 \leq c \leq C} \mathbf{F}''(c, i, j) \tag{25}$$

$$\mathbf{M}_s' = \sigma\left(\mathbf{W}_{s'} * [\mathbf{s}_{\text{avg}}';\; \mathbf{s}_{\text{max}}']\right) \in \mathbb{R}^{1 \times H \times W}, \quad \mathbf{W}_{s'} \in \mathbb{R}^{1 \times 2 \times 7 \times 7} \tag{26}$$

$$\mathbf{F}_{\text{out}} = \mathbf{M}_s' \otimes \mathbf{F}'' \tag{27}$$

This cascaded design forms a **coarse-to-fine** spatial focusing mechanism. In the first pass, the spatial attention map $\mathbf{M}_s$ identifies broadly relevant regions (e.g., foreground vs. background). In the second pass, the refinement map $\mathbf{M}_s'$ operates on a cleaner feature space (after channel and initial spatial filtering) and can focus on finer-grained distinctions—such as discriminating between adjacent object boundaries, small-scale structures, or thin elongated regions that the first pass may under-attend. Notably, the two spatial attention modules share identical architectures ($7 \times 7$ convolution with $\text{padding} = 3$, no bias) but learn **distinct** parameters $\mathbf{W}_s \neq \mathbf{W}_{s'}$, allowing each stage to specialize its spatial gating behavior.

The complete Spatial-CBAM transformation can be formalized as a composite function:

$$\mathbf{F}_{\text{out}} = \underbrace{\text{SA}_{\theta_{s'}}}_{\text{Spatial Refine}} \circ \underbrace{\text{SA}_{\theta_s}}_{\text{Spatial Attn}} \circ \underbrace{\text{CA}_{\theta_c}}_{\text{Channel Attn}} (\mathbf{F}) \tag{28}$$

where $\theta_c = \{\mathbf{W}_1, \mathbf{W}_2\}$, $\theta_s = \{\mathbf{W}_s\}$, and $\theta_{s'} = \{\mathbf{W}_{s'}\}$ denote the learnable parameters of each stage. The total additional parameter count of Spatial-CBAM is:

$$|\theta_{\text{S-CBAM}}| = \underbrace{2 \times C \times \lfloor C/r \rfloor}_{\text{Channel MLP}} + \underbrace{2 \times (2 \times 7 \times 7)}_{\text{Two spatial convs}} = 2 \times 1280 \times 80 + 2 \times 98 = 204{,}996 \tag{29}$$

This represents a negligible overhead (<0.5%) relative to the backbone parameters, yet delivers substantial performance gains as demonstrated in our ablation study.

**To further bridge the semantic gap between the attention-recalibrated high-level features and the spatial-rich mid-level features, we introduce a dedicated cross-level fusion module described next.**

---

## 3.3 Shallow-Advanced Feature Fusion (SAFF) Module

### 3.3.1 Motivation

In standard DeepLabV3+, the encoder-to-decoder pathway creates only a single skip connection: high-level ASPP features (at $1/16$ scale) are upsampled and concatenated with low-level features (at $1/4$ scale) in the decoder. This design introduces a significant semantic gap—the low-level feature from the first residual stage captures fine-grained spatial details (edges, textures) but lacks semantic understanding, while the ASPP output encodes rich semantics but at coarse spatial resolution. The intermediate mid-level features from the second residual stage ($1/8$ scale), which naturally bridge these two extremes by encoding structural and geometric patterns (corners, object parts, medium-scale textures), are entirely discarded in the standard pipeline.

We observe that directly concatenating high-level and low-level features forces the decoder to reconcile a 4$\times$ resolution gap and a large channel-dimension mismatch (256 vs. 48) in a single step, placing excessive burden on the two decoder convolution layers. To address this, we propose the **Shallow-Advanced Feature Fusion (SAFF)** module, which introduces an explicit intermediate fusion step at the $1/8$ scale between the ASPP output and the mid-level backbone feature. The architecture of SAFF is shown in Fig. 4.

### 3.3.2 Architecture and Formulation

The SAFF module takes two inputs: the mid-level backbone feature $\mathbf{F}_{\text{mid}} \in \mathbb{R}^{C_m \times \frac{H}{8} \times \frac{W}{8}}$ and the attention-refined ASPP output $\mathbf{F}_{\text{aspp}} \in \mathbb{R}^{256 \times \frac{H}{16} \times \frac{W}{16}}$.

**Mid-level projection path.** The mid-level feature is projected from its original channel dimension $C_m = 512$ to a unified dimension of 256 using a $1 \times 1$ convolution followed by batch normalization and ReLU activation:

$$\mathbf{F}_{\text{mid}}' = \delta\left(\text{BN}\left(\mathbf{W}_{\text{f1}} * \mathbf{F}_{\text{mid}}\right)\right) \in \mathbb{R}^{256 \times \frac{H}{8} \times \frac{W}{8}}, \quad \mathbf{W}_{\text{f1}} \in \mathbb{R}^{256 \times C_m \times 1 \times 1} \tag{30}$$

This $1 \times 1$ projection serves two purposes: (1) channel dimension alignment to match the ASPP output, and (2) channel-wise feature selection that discards low-level noise while retaining structurally informative activations from the mid-level representation.

**ASPP refinement path.** The ASPP output, residing at $\frac{H}{16} \times \frac{W}{16}$ resolution, must be spatially aligned to the mid-level feature at $\frac{H}{8} \times \frac{W}{8}$. We first apply $2\times$ bilinear interpolation to upsample the spatial dimensions:

$$\tilde{\mathbf{F}}_{\text{aspp}} = \text{Upsample}_{2\times}(\mathbf{F}_{\text{aspp}}) \in \mathbb{R}^{256 \times \frac{H}{8} \times \frac{W}{8}} \tag{31}$$

However, naive bilinear upsampling introduces spatial blurring and cannot adapt the semantic feature to the finer spatial grid. Therefore, we apply a $3 \times 3$ dilated convolution with dilation rate $d = 3$ and padding $p = 3$ (to preserve spatial dimensions) as a post-upsampling refinement:

$$\mathbf{F}_{\text{aspp}}' = \delta\left(\text{BN}\left(\mathbf{W}_{\text{f2}} *_3 \tilde{\mathbf{F}}_{\text{aspp}}\right)\right) \in \mathbb{R}^{256 \times \frac{H}{8} \times \frac{W}{8}}, \quad \mathbf{W}_{\text{f2}} \in \mathbb{R}^{256 \times 256 \times 3 \times 3} \tag{32}$$

where $*_3$ denotes convolution with dilation $d = 3$. The effective receptive field of this dilated convolution is:

$$k_{\text{eff}} = k + (k - 1)(d - 1) = 3 + (3 - 1)(3 - 1) = 7 \tag{33}$$

This $7 \times 7$ effective receptive field serves a critical dual purpose. First, it provides a sufficiently large spatial context to re-adapt the interpolated feature values to their new spatial positions, compensating for the imprecision of bilinear upsampling. Second, the dilation maintains global semantic context from the original ASPP output—each $7 \times 7$ effective window at $1/8$ scale corresponds to a $56 \times 56$ pixel region in the original image, ensuring that the upsampled feature retains broad contextual awareness despite operating at higher resolution.

**Additive fusion and output projection.** The two spatially and channel-aligned features are fused via element-wise addition:

$$\mathbf{F}_{\text{sum}} = \mathbf{F}_{\text{mid}}' + \mathbf{F}_{\text{aspp}}' \in \mathbb{R}^{256 \times \frac{H}{8} \times \frac{W}{8}} \tag{34}$$

We adopt element-wise addition rather than concatenation for the following design considerations: (i) addition enforces feature alignment in a shared latent space, encouraging the network to learn complementary representations from both levels—the mid-level branch focuses on spatial precision while the ASPP branch contributes semantic context; (ii) addition introduces no channel dimension expansion, maintaining a constant 256-channel width and avoiding the need for subsequent channel reduction; (iii) the additive formulation can be interpreted as a residual connection where $\mathbf{F}_{\text{aspp}}'$ serves as a semantic "correction" to the spatial-rich $\mathbf{F}_{\text{mid}}'$.

A final $1 \times 1$ convolution with BN and ReLU is applied as a nonlinear output projection:

$$\mathbf{F}_{\text{fused}} = \delta\left(\text{BN}\left(\mathbf{W}_{\text{out}} * \mathbf{F}_{\text{sum}}\right)\right) \in \mathbb{R}^{256 \times \frac{H}{8} \times \frac{W}{8}}, \quad \mathbf{W}_{\text{out}} \in \mathbb{R}^{256 \times 256 \times 1 \times 1} \tag{35}$$

This projection enables nonlinear feature recombination that selectively emphasizes the most discriminative fused features while suppressing any residual misalignment between the two paths. The complete SAFF transformation can be written in compact form as:

$$\text{SAFF}(\mathbf{F}_{\text{mid}}, \mathbf{F}_{\text{aspp}}) = g_{\text{out}}\left(g_{\text{f1}}(\mathbf{F}_{\text{mid}}) + g_{\text{f2}}\left(\text{Up}_{2\times}(\mathbf{F}_{\text{aspp}})\right)\right) \tag{36}$$

where $g_{\text{f1}}$, $g_{\text{f2}}$, and $g_{\text{out}}$ denote the three Conv-BN-ReLU blocks.

### 3.3.3 Three-Tier Feature Hierarchy

By inserting the SAFF module between the ASPP encoder and the decoder, we establish a progressive three-tier feature fusion hierarchy. The overall decoder data flow can be formalized as:

$$\mathbf{F}_{\text{fused}} = \text{SAFF}\left(\mathbf{F}_{\text{mid}},\; \text{Proj}\left(\text{S-CBAM}\left(\text{ASPP}(\mathbf{F}_{\text{high}})\right)\right)\right) \tag{37}$$

$$\mathbf{F}_{\text{dec}} = \text{Concat}\left[\text{Proj}_{\text{low}}(\mathbf{F}_{\text{low}}),\; \text{Up}_{2\times}(\mathbf{F}_{\text{fused}})\right] \tag{38}$$

$$\hat{\mathbf{Y}}_{\text{seg}} = \text{Up}_{4\times}\left(\text{Classifier}(\mathbf{F}_{\text{dec}})\right) \tag{39}$$

This creates a smooth information flow: semantic information is injected at $1/16$ scale (ASPP), fused with structural information at $1/8$ scale (SAFF), and combined with fine-grained spatial details at $1/4$ scale (decoder). Each stage reduces the resolution gap by only $2\times$, making the decoder's task substantially easier compared to the standard $4\times$ gap in vanilla DeepLabV3+.

**By establishing an explicit information pathway for mid-level features, the SAFF module constructs a three-tier feature hierarchy that progressively integrates information at increasing semantic levels. To further improve boundary delineation in the final segmentation output, we introduce a boundary auxiliary supervision branch described below.**

---

## 3.4 Boundary Auxiliary Branch

### 3.4.1 Motivation

Semantic segmentation networks commonly produce blurred or inconsistent predictions near object boundaries. This phenomenon arises from multiple sources: (i) pooling operations and strided convolutions in the encoder progressively erode fine-grained boundary information; (ii) the standard cross-entropy loss treats all pixels equally regardless of their spatial proximity to boundaries, providing insufficient gradient signals for the numerically minority boundary pixels; (iii) bilinear upsampling in the decoder introduces spatial smoothing that further degrades boundary sharpness. While our SAFF module enriches features with mid-level spatial cues, it operates at $1/8$ resolution and cannot fully recover the pixel-level boundary precision required for accurate segmentation.

To directly address these challenges, we introduce a lightweight **Boundary Auxiliary Branch** that provides explicit supervisory signals for boundary-adjacent pixels. The key insight is that by forcing a parallel decoder branch to predict object boundaries, the shared feature backbone is encouraged to preserve and enhance edge-discriminative representations throughout the network—benefiting not only the boundary prediction but also the main segmentation task through gradient back-propagation. The module architecture and training pipeline are illustrated in Fig. 5.

### 3.4.2 Boundary Auxiliary Head

The boundary head operates on the decoder feature $\hat{\mathbf{F}}_{\text{dec}} \in \mathbb{R}^{C_d \times \frac{H}{4} \times \frac{W}{4}}$ (where $C_d = 304$ when using the legacy decoder or $C_d = 256$ when using the improved decoder). The head consists of a compact three-layer fully convolutional network with progressively reducing channel dimensions:

**Layer 1 (Feature Compression):** A $3 \times 3$ convolution compresses the high-dimensional decoder feature to a compact 64-channel representation while extracting local edge-oriented patterns:

$$\mathbf{H}_1 = \delta\left(\text{BN}\left(\mathbf{W}_{\text{b1}} * \hat{\mathbf{F}}_{\text{dec}}\right)\right) \in \mathbb{R}^{64 \times \frac{H}{4} \times \frac{W}{4}}, \quad \mathbf{W}_{\text{b1}} \in \mathbb{R}^{64 \times C_d \times 3 \times 3} \tag{40}$$

**Layer 2 (Boundary Refinement):** A second $3 \times 3$ convolution further refines the boundary representation, providing additional nonlinear capacity to distinguish true boundaries from texture edges:

$$\mathbf{H}_2 = \delta\left(\text{BN}\left(\mathbf{W}_{\text{b2}} * \mathbf{H}_1\right)\right) \in \mathbb{R}^{64 \times \frac{H}{4} \times \frac{W}{4}}, \quad \mathbf{W}_{\text{b2}} \in \mathbb{R}^{64 \times 64 \times 3 \times 3} \tag{41}$$

The two $3 \times 3$ convolution layers provide a combined effective receptive field of $5 \times 5$ pixels at $1/4$ resolution, corresponding to a $20 \times 20$ pixel region in the original image. This is sufficient to capture local boundary context while keeping the head lightweight.

**Layer 3 (Logit Prediction):** A $1 \times 1$ convolution with bias produces a single-channel boundary logit map:

$$\mathbf{E} = \mathbf{W}_{\text{b3}} * \mathbf{H}_2 + \mathbf{b}_{\text{b3}} \in \mathbb{R}^{1 \times \frac{H}{4} \times \frac{W}{4}}, \quad \mathbf{W}_{\text{b3}} \in \mathbb{R}^{1 \times 64 \times 1 \times 1} \tag{42}$$

Note that this final layer uses a bias term $\mathbf{b}_{\text{b3}} \in \mathbb{R}$ (unlike the preceding layers that are bias-free) to allow the network to learn an initial boundary/non-boundary offset, which helps the sigmoid function operate in an appropriate range from the beginning of training.

The edge logit map is bilinearly upsampled to the original resolution:

$$\hat{\mathbf{Y}}_{\text{edge}} = \text{Up}_{4\times}(\mathbf{E}) \in \mathbb{R}^{1 \times H \times W} \tag{43}$$

The complete boundary head transformation can be compactly written as:

$$\hat{\mathbf{Y}}_{\text{edge}} = \text{Up}_{4\times}\left(\mathbf{W}_{\text{b3}} * \delta\left(\text{BN}\left(\mathbf{W}_{\text{b2}} * \delta\left(\text{BN}\left(\mathbf{W}_{\text{b1}} * \hat{\mathbf{F}}_{\text{dec}}\right)\right)\right)\right) + \mathbf{b}_{\text{b3}}\right) \tag{44}$$

The total parameter count of the boundary head is:

$$|\theta_{\text{boundary}}| = C_d \times 64 \times 9 + 64 \times 64 \times 9 + 64 \times 1 + 1 \approx 211{,}K \tag{45}$$

representing less than 0.5% of the backbone parameters, making it an extremely efficient addition.

### 3.4.3 Morphological Boundary Target Generation

A critical aspect of boundary supervision is the quality and form of the ground truth boundary target. Rather than relying on pre-computed edge detection algorithms (e.g., Canny or Sobel), which are class-agnostic and sensitive to texture edges, we generate boundary supervision targets directly from the semantic ground truth labels through a class-aware morphological dilation–erosion scheme.

For a ground truth semantic mask $\mathbf{Y} \in \{0, 1, \ldots, K-1\}^{H \times W}$, we first extract the binary mask for each semantic class $c$:

$$\mathbf{Y}_c(i, j) = \mathbb{1}[\mathbf{Y}(i, j) = c] \in \{0, 1\}^{H \times W} \tag{46}$$

For each class binary mask, we compute the morphological boundary band using dilation and erosion operations with a square structuring element of radius $\rho$:

$$\text{Dilate}(\mathbf{Y}_c, \rho)(i, j) = \max_{|p-i| \leq \rho, |q-j| \leq \rho} \mathbf{Y}_c(p, q) \tag{47}$$

$$\text{Erode}(\mathbf{Y}_c, \rho)(i, j) = \min_{|p-i| \leq \rho, |q-j| \leq \rho} \mathbf{Y}_c(p, q) \tag{48}$$

In implementation, the dilation is realized via max-pooling with kernel size $k = 2\rho + 1$ and the erosion via negated max-pooling on the negated mask, which avoids explicit morphological operation libraries:

$$\text{Dilate}(\mathbf{Y}_c, \rho) = \text{MaxPool}_{2\rho+1}(\mathbf{Y}_c) \tag{49}$$

$$\text{Erode}(\mathbf{Y}_c, \rho) = -\text{MaxPool}_{2\rho+1}(-\mathbf{Y}_c) \tag{50}$$

The per-class boundary is the set of pixels that lie in the dilated region but not in the eroded region—i.e., the band of width $2\rho$ straddling each class boundary:

$$\mathbf{B}_c = \mathbb{1}\left[\text{Dilate}(\mathbf{Y}_c, \rho) - \text{Erode}(\mathbf{Y}_c, \rho) > 0\right] \in \{0, 1\}^{H \times W} \tag{51}$$

The final boundary target is the union of all per-class boundary bands, further masked by the valid pixel region (excluding ignore-labeled pixels):

$$\mathbf{Y}_{\text{edge}}(i, j) = \mathbb{1}\left[\max_{c \in \{0, \ldots, K-1\}} \mathbf{B}_c(i, j) > 0\right] \cdot \mathbb{1}[\mathbf{Y}(i, j) \neq \text{ignore}] \tag{52}$$

With $\rho = 3$ (default), the resulting boundary band is 6 pixels wide (3 pixels on each side of the true boundary), which provides two key advantages: (1) a thicker band encompasses more pixels, providing richer gradient signals compared to thin single-pixel boundary annotations; (2) the band naturally accounts for labeling noise and minor misalignments in ground truth annotations, improving training stability.

### 3.4.4 Boundary Loss Formulation

The boundary prediction is a binary classification task with severe class imbalance—boundary pixels typically constitute only 5–15% of the image area. To address this, we design a composite boundary loss $\mathcal{L}_{\text{boundary}}$ that combines two complementary objectives.

**Binary Cross-Entropy (BCE) Loss** provides per-pixel supervision with a stable gradient landscape:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N_{\text{valid}}}\sum_{i \in \Omega_{\text{valid}}}\left[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\right] \tag{53}$$

where $\hat{p}_i = \sigma(\hat{e}_i)$ is the predicted boundary probability obtained by applying sigmoid to the raw logit $\hat{e}_i$, $y_i \in \{0, 1\}$ is the boundary ground truth, $\Omega_{\text{valid}}$ is the set of valid pixels (excluding ignore-labeled regions), and $N_{\text{valid}} = |\Omega_{\text{valid}}|$.

**Dice Loss** directly optimizes the set-level overlap between predicted and ground truth boundary regions, providing a class-imbalance-robust objective:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_{i \in \Omega_{\text{valid}}} \hat{p}_i \cdot y_i + \epsilon}{\sum_{i \in \Omega_{\text{valid}}} \hat{p}_i + \sum_{i \in \Omega_{\text{valid}}} y_i + \epsilon} \tag{54}$$

where $\epsilon = 1 \times 10^{-5}$ is a smoothing constant to prevent division by zero. The Dice loss effectively up-weights the contribution of minority boundary pixels by normalizing against the predicted and ground truth boundary area, rather than the total number of pixels. This complements the BCE loss, which is dominated by the majority non-boundary class.

The composite boundary loss is:

$$\mathcal{L}_{\text{boundary}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}} \tag{55}$$

### 3.4.5 Joint Optimization

The total training objective integrates the main segmentation loss with the weighted boundary loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\hat{\mathbf{Y}}_{\text{seg}}, \mathbf{Y}) + \lambda \cdot \mathcal{L}_{\text{boundary}}(\hat{\mathbf{Y}}_{\text{edge}}, \mathbf{Y}_{\text{edge}}) \tag{56}$$

where $\mathcal{L}_{\text{CE}}$ is the standard pixel-wise cross-entropy loss for $K$-class semantic segmentation:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{N_{\text{valid}}}\sum_{i \in \Omega_{\text{valid}}} \sum_{k=1}^{K} \mathbb{1}[\mathbf{Y}(i) = k] \log \frac{\exp(\hat{\mathbf{Y}}_{\text{seg}}^k(i))}{\sum_{k'=1}^{K} \exp(\hat{\mathbf{Y}}_{\text{seg}}^{k'}(i))} \tag{57}$$

and $\lambda$ is a balancing hyperparameter that controls the relative contribution of boundary supervision. The gradients from $\mathcal{L}_{\text{boundary}}$ back-propagate through the boundary head and into the shared decoder feature $\hat{\mathbf{F}}_{\text{dec}}$, further flowing into the SAFF module, the ASPP encoder, and ultimately the backbone. This gradient pathway creates a beneficial regularization effect: the boundary loss explicitly encourages the entire network to preserve edge-discriminative features at every stage of the encoder-decoder pipeline, complementing the pixel-wise cross-entropy loss that primarily drives class discrimination.

An important practical property of the boundary branch is its **flexibility at inference time**: since the boundary head is a separate module with no influence on the segmentation head's forward computation, it can be optionally discarded during deployment without affecting segmentation accuracy. This makes the boundary branch a **training-only regularizer** that improves the learned feature representations without adding any inference overhead. Alternatively, when applications require explicit edge maps (e.g., instance contour extraction, panoptic segmentation post-processing), the boundary output can serve as a complementary prediction.

**The synergy among the three proposed components—Spatial-CBAM, SAFF, and the boundary auxiliary branch—constitutes a comprehensive enhancement pipeline that operates across different dimensions of the segmentation framework. Spatial-CBAM recalibrates the multi-scale ASPP features in both channel and spatial dimensions to suppress redundancy and highlight discriminative patterns. SAFF bridges multi-level encoder features to reduce the semantic gap in the encoder-decoder pathway. The boundary auxiliary branch provides explicit geometric supervision that sharpens decision boundaries through gradient-level regularization. Together, these three components work synergistically to improve segmentation accuracy, particularly at object boundaries and for small-scale structures, while maintaining computational efficiency.**

---

*Figure 2: Overall architecture of the proposed SA-BNet framework. The backbone extracts three levels of features ($\mathbf{F}_{\text{low}}$, $\mathbf{F}_{\text{mid}}$, $\mathbf{F}_{\text{high}}$), which are processed through ASPP with Spatial-CBAM attention, fused via the SAFF module, and decoded with boundary auxiliary supervision.*

*Figure 3: Architecture of the Spatial-Enhanced CBAM (Spatial-CBAM) attention module. The three-stage cascade (Channel Attention → Spatial Attention → Spatial Refinement) progressively recalibrates the multi-scale feature from channel-wise importance to coarse spatial localization to fine-grained spatial correction.*

*Figure 4: Architecture of the Shallow-Advanced Feature Fusion (SAFF) module. The mid-level feature $\mathbf{F}_{\text{mid}}$ is projected via $1 \times 1$ convolution, while the ASPP output is upsampled and refined via dilated convolution. The two aligned features are additively fused and projected to produce $\mathbf{F}_{\text{fused}}$.*

*Figure 5: Architecture and training pipeline of the Boundary Auxiliary Branch. (A) The boundary head processes decoder features through a lightweight convolutional network to predict edge logits. (B) Ground truth boundary targets are generated from semantic labels via class-aware morphological operations. (C) The joint loss combines segmentation cross-entropy with weighted boundary BCE+Dice loss.*
