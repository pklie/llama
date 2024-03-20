# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama
from typing import List

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts_comp: List[str] = [
        """
        Summarize this in 1000 words:
Preprint
EMERNERF: EMERGENT SPATIAL-TEMPORAL SCENE
DECOMPOSITION VIA SELF-SUPERVISION
Jiawei Yang∗,¶, Boris Ivanovic¶, Or Litany†,¶, Xinshuo Weng¶, Seung Wook Kim¶, Boyi Li¶,
Tong Che¶, Danfei Xu$,¶, Sanja Fidler§,¶, Marco Pavone‡,¶, Yue Wang∗,¶
∗ {yangjiaw,yue.w}@usc.edu, University of Southern California
$ danfei@gatech.edu, Georgia Institute of Technology
§ fidler@cs.toronto.edu, University of Toronto
‡ pavone@stanford.edu, Stanford University
† orlitany@gmail.com, Technion
¶ {bivanovic,xweng,seungwookk,boyil,tongc}@nvidia.com, NVIDIA Research
ABSTRACT
We present EmerNeRF, a simple yet powerful approach for learning spatial-
temporal representations of dynamic driving scenes. Grounded in neural fields,
EmerNeRF simultaneously captures scene geometry, appearance, motion, and
semantics via self-bootstrapping.
EmerNeRF hinges upon two core compo-
nents: First, it stratifies scenes into static and dynamic fields. This decomposition
emerges purely from self-supervision, enabling our model to learn from general,
in-the-wild data sources. Second, EmerNeRF parameterizes an induced flow field
from the dynamic field and uses this flow field to further aggregate multi-frame
features, amplifying the rendering precision of dynamic objects. Coupling these
three fields (static, dynamic, and flow) enables EmerNeRF to represent highly-
dynamic scenes self-sufficiently, without relying on ground truth object annota-
tions or pre-trained models for dynamic object segmentation or optical flow esti-
mation. Our method achieves state-of-the-art performance in sensor simulation,
significantly outperforming previous methods when reconstructing static (+2.93
PSNR) and dynamic (+3.70 PSNR) scenes. In addition, to bolster EmerNeRF’s
semantic generalization, we lift 2D visual foundation model features into 4D
space-time and address a general positional bias in modern Transformers, signif-
icantly boosting 3D perception performance (e.g., 37.50% relative improvement
in occupancy prediction accuracy on average). Finally, we construct a diverse
and challenging 120-sequence dataset to benchmark neural fields under extreme
and highly-dynamic settings. See the project page for code, data, and request
pre-trained models: https://emernerf.github.io
1
INTRODUCTION
Perceiving, representing, and reconstructing dynamic scenes are critical for autonomous agents to
understand and interact with their environments. Current approaches predominantly build custom
pipelines with components dedicated to identifying and tracking static obstacles and dynamic ob-
jects (Yang et al., 2023; Guo et al., 2023). However, such approaches require training each com-
ponent with a large amount of labeled data and devising complex mechanisms to combine out-
puts across components. To represent static scenes, approaches leveraging neural radiance fields
(NeRFs) (Mildenhall et al., 2021) have witnessed a Cambrian explosion in computer graphics,
robotics, and autonomous driving, owing to their strong performance in estimating 3D geometry
and appearance (Rematas et al., 2022; Tancik et al., 2022; Wang et al., 2023b; Guo et al., 2023).
However, without explicit supervision, NeRFs struggle with dynamic environments filled with fast-
moving objects, such as vehicles and pedestrians in urban scenarios. In this work, we tackle this
long-standing challenge and develop a self-supervised technique for building 4D (space-time) rep-
resentations of dynamic scenes.
1
arXiv:2311.02077v1  [cs.CV]  3 Nov 2023
Preprint
(a) GT RGB
(b) Rendered RGB
(c) Decomposed Static RGB
(d) Decomposed Static Depth
(e) Decomposed Dynamic RGB
(f) Decomposed Dynamic Depth
(g) Emerged Scene Flow
(h) Reconstructed DINOv2 Features
(i) Decomposed PE-Free DINOV2 Features
(j) Decomposed PE-Features
Figure 1: EmerNeRF effectively reconstructs photo-realistic dynamic scenes (b), separating them
into explicit static (c-d) and dynamic (e-f) elements, all via self-supervision. Notably, (g) scene flows
emerge from EmerNeRF without any explicit flow supervision. Moreover, EmerNeRF can address
detrimental positional embedding (PE) patterns observed in vision foundation models (h, j), and lift
clean, PE-free features into 4D space (i). Additional visualizations can be found in Appendix C.4.
We consider a common setting where a mobile robot equipped with multiple sensors (e.g., cam-
eras, LiDAR) navigates through a large dynamic environment (e.g., a neighborhood street). The
fundamental underlying challenge is to construct an expansive 4D representation with only sparse,
transient observations, that is, to reconstruct an entire 4D space-time volume from a single traversal
of the volume. Unfortunately, this setting challenges NeRF’s multi-view consistency assumption—
each point in the space-time volume will only be observed once. Recent works (Yang et al., 2023;
Ost et al., 2021) seek to simplify the problem by modeling static scene components (e.g., buildings
and trees) and dynamic objects separately, using a combination of neural fields and mesh repre-
sentations. This decomposition enables exploiting multi-timestep observations to supervise static
components, but it often requires costly ground-truth annotations to segment and track dynamic ob-
jects. Moreover, no prior works in this field have explicitly modeled temporal correspondence for
2
Preprint
dynamic objects, a prerequisite for accumulating sparse supervision over time. Overall, learning 4D
representations of dynamic scenes remains a formidable challenge.
Towards this end, we present EmerNeRF, a self-supervised approach for constructing 4D neural
scene representations. As shown in Fig. 1, EmerNeRF decouples static and dynamic scene com-
ponents and estimates 3D scene flows — remarkably, all from self-supervision. At a high level,
EmerNeRF builds a hybrid static-dynamic world representation via a density-regularized objec-
tive, generating density for dynamic objects only as necessary (i.e., when points intersect dynamic
objects). This representation enables our approach to capture dynamic components and exploit
multi-timestep observations to self-supervise static scene elements. To address the lack of cross-
observation consistency for dynamic components, we task EmerNeRF to predict 3D scene flows
and use them to aggregate temporally-displaced features. Intriguingly, EmerNeRF’s capability to
estimate scene flow emerges naturally from this process, without any explicit flow supervision. Fi-
nally, to enhance scene comprehension, we “lift” features from pre-trained 2D visual foundation
models (e.g., DINOv1 (Caron et al., 2021), DINOv2 (Oquab et al., 2023)) to 4D space-time. In do-
ing so, we observe and rectify a challenge tied to Transformer-based foundation models: positional
embedding (PE) patterns (Fig. 1 (h)). As we will show in §4.3, effectively utilizing such general
features greatly improves EmerNeRF’s semantic understanding and enables few-shot auto-labeling.
We evaluate EmerNeRF on sensor sequences collected by autonomous vehicles (AVs) traversing
through diverse urban environments. A critical challenge is that current autonomous driving datasets
are heavily imbalanced, containing many simple scenarios with few dynamic objects. To facilitate
a focused empirical study and bolster future research on this topic, we present the NeRF On-The-
Road (NOTR) benchmark, a balanced subsample of 120 driving sequences from the Waymo Open
Dataset (Sun et al., 2020) containing diverse visual conditions (lighting, weather, and exposure) and
challenging dynamic scenarios. On this benchmark, EmerNeRF significantly outperforms previous
state-of-the-art NeRF-based approaches (Park et al., 2021b; Wu et al., 2022; M¨uller et al., 2022; Guo
et al., 2023) on scene reconstruction by 2.93 and 3.70 PSNR on static and dynamic scenes, respec-
tively, and by 2.91 PSNR on dynamic novel view synthesis. For scene flow estimation, EmerNeRF
excels over Li et al. (2021a) by 42.16% in metrics of interest. Additionally, removing PE patterns
brings an average improvement of 37.50% relative to using the original, PE pattern-laden features
on semantic occupancy prediction. Contributions. Our key contributions are fourfold: (1) We in-
troduce EmerNeRF, a novel 4D neural scene representation framework that excels in challenging
autonomous driving scenarios. EmerNeRF performs static-dynamic decomposition and scene flow
estimation, all through self-supervision. (2) A streamlined method to tackle the undesired effects of
positional embedding patterns from Vision Transformers, which is immediately applicable to other
tasks. (3) We introduce the NOTR dataset to assess neural fields in diverse conditions and facili-
tate future development in the field. (4) EmerNeRF achieves state-of-the-art performance in scene
reconstruction, novel view synthesis, and scene flow estimation.
2
RELATED WORK
Dynamic scene reconstruction with NeRFs. Recent works adopt NeRFs (Mildenhall et al., 2021;
M¨uller et al., 2022) to accommodate dynamic scenes (Li et al., 2021b; Park et al., 2021b; Wu et al.,
2022). Earlier methods (Bansal et al., 2020; Li et al., 2022; Wang et al., 2022; Fang et al., 2022) for
dynamic view synthesis rely on multiple synchronized videos recorded from different viewpoints,
restricting their use for real-world applications in autonomous driving and robotics. Recent meth-
ods, such as Nerfies (Park et al., 2021a) and HyperNeRF (Park et al., 2021b), have managed to
achieve dynamic view synthesis using a single camera. However, they rely on a strict assumption
that all observations can be mapped via deformation back to a canonical reference space, usually
constructed from the first timestep. This assumption does not hold in driving because objects might
not be fully present in any single frame and can constantly enter and exit the scene.
Of particular relevance to our work are methods like D2NeRF (Wu et al., 2022), SUDS (Turki et al.,
2023), and NeuralGroundplans (Sharma et al., 2022). These methods also partition a 4D scene
into static and dynamic components. However, D2NeRF underperforms significantly for outdoor
scenes due to its sensitivity to hyperparameters and insufficient capacity; NeuralGroundplan relies
on synchronized videos from different viewpoints to reason about dynamics; and SUDS, designed
for multi-traversal driving logs, largely relies on accurate optical flows derived by pre-trained models
3
Preprint
and incurs high computational costs due to its expensive flow-based warping losses. In contrast, our
approach can reconstruct an accurate 4D scene representation from a single-traversal log captured
by sensors mounted on a self-driving vehicle. Freed from the constraints of pre-trained flow models,
EmerNeRF exploits and refines its own intrinsic flow predictions, enabling a self-improving loop.
NeRFs for AV data. Creating high-fidelity neural simulations from collected driving logs is crucial
for the autonomous driving community, as it facilitates the closed-loop training and testing of various
algorithms. Beyond SUDS (Turki et al., 2023), there is a growing interest in reconstructing scenes
from driving logs. In particular, recent methods excel with static scenes but face challenges with
dynamic objects (Guo et al., 2023). While approaches like UniSim (Yang et al., 2023) and NSG (Ost
et al., 2021) handle dynamic objects, they depend on ground truth annotations, making them less
scalable due to the cost of obtaining such annotations. In contrast, our method achieves high-fidelity
simulation results purely through self-supervision, offering a scalable solution.
Augmenting NeRFs. NeRF methods are commonly augmented with external model outputs to
incorporate additional information. For example, approaches that incorporate scene flow often rely
on existing optical flow models for supervision (Li et al., 2021b; Turki et al., 2023; Li et al., 2023b).
They usually require cycle-consistency tests to filter out inconsistent flow estimations; otherwise,
the optimization process is prone to failure (Wang et al., 2023a). The Neural Scene Flow Prior
(NSFP) (Li et al., 2021a), a state-of-the-art flow estimator, optimizes a neural network to estimate
the scene flow at each timestep (minimizing the Chamfer Loss (Fan et al., 2017)). This per-timestep
optimization makes NSFP prohibitively expensive. In contrast, our EmerNeRF bypasses the need
for either pre-trained optical flow models or holistic geometry losses. Instead, our flow field is
supervised only by scene reconstruction losses and the flow estimation capability emerges on its
own. Most recently, 2D signals such as semantic labels or foundation model feature vectors have
been distilled into 3D space (Kobayashi et al., 2022; Kerr et al., 2023; Tsagkas et al., 2023; Shafiullah
et al., 2022), enabling semantic understanding tasks. In this work, we similarly lift visual foundation
model features into 4D space and show their potential for few-shot perception tasks.
3
SELF-SUPERVISED SPATIAL-TEMPORAL NEURAL FIELDS
Learning a spatial-temporal representation of a dynamic environment with a multi-sensor robot is
challenging due to the sparsity of observations and costs of obtaining ground truth annotations. To
this end, our design choices stem from the following key principles: (1) Learn a scene decompo-
sition entirely through self-supervision and avoid using any ground-truth annotations or pre-trained
models for dynamic object segmentation or optical flow. (2) Model dynamic element correspon-
dences across time via scene flow. (3) Obtain a mutually reinforcing representation: static-dynamic
decomposition and flow estimation can benefit from each other. (4) Improve the semantics of scene
representations by leveraging feature lifting and distillation, enabling a range of perception tasks.
Having established several design principles, we are now equipped to describe EmerNeRF, a self-
supervised approach for efficiently representing both static and dynamic scene components. First,
§3.1 details how EmerNeRF builds a hybrid world representation with a static and dynamic field.
Then, §3.2 explains how EmerNeRF leverages an emergent flow field to aggregate temporal features
over time, further improving its representation of dynamic components. §3.3 describes the lifting
of semantic features from pre-trained 2D models to 4D space-time, enhancing EmerNeRF’s scene
understanding. Finally, §3.4 discusses the loss function that is minimized during training.
3.1
SCENE REPRESENTATIONS
Scene decomposition. To enable efficient scene decomposition, we design EmerNeRF to be a
hybrid spatial-temporal representation. It decomposes a 4D scene into a static field S and a dynamic
field D, both of which are parameterized by learnable hash grids (M¨uller et al., 2022) Hs and
Hd, respectively. This decoupling offers a flexible and compact 4D scene representation for time-
independent features hs = Hs(x) and time-varying features hd = Hd(x, t), where x = (x, y, z) is
the 3D location of a query point and t denotes its timestep. These features are further transformed
into gs and gd by lightweight MLPs (gs and gd) and used to predict per-point density σs and σd:
gs, σs = gs(Hs(x))
gd, σd = gd(Hd(x, t))
(1)
4
Preprint
Bilinear 
Interpolation
NeRF
Learnable  
2D Feature maps
+
x, y, z, t
u, v, (t)
Decomposed Features (remove 
2D-position-related patterns)
2D-position-related 
Output features
Shape: (C, h, w). I used (32, 80, 120)
u, v: pixel coordinates in image plane
NeRF
(x, t)
(x, t)
x
(x, t)
Σ
Feature 
 Head
(u, v)
d
+
cd
csky
ρ
σd
σscs
gs
(x, t)
PE Head
Pixel 
Coordinate
PE Patterns
PE-Free Features
(a)
(b)
Static Field
Dynamic Field
Final Prediction
gt−1
d
gt
d
gt+1
d
Color 
Head
Shadow 
Head
g′ d
Flow Field
Sky Head
Learnable PE Map
vf
vb
Figure 2: EmerNeRF Overview. (a) EmerNeRF consists of a static, dynamic, and flow field
(S, D, V). These fields take as input either a spatial query x or spatial-temporal query (x, t) to
generate a static (feature gs, density σs) pair or a dynamic (feature g′
d, density σd) pair. Of note,
we use the forward and backward flows (vf and vb) to generate temporally-aggregated features g′
d
from nearby temporal features gt−1
d
, gt
d, and gt+1
d
(a slight abuse of notation w.r.t. Eq. (8)). These
features (along with the view direction d) are consumed by the shared color head which indepen-
dently predicts the static and dynamic colors cs and cd. The shadow head predicts a shadow ratio ρ
from the dynamic features. The sky head decodes a per-ray color csky for sky pixels from the view
direction d. (b) EmerNeRF renders the aggregated features to 2D and removes undesired positional
encoding patterns (via a learnable PE map followed by a lightweight PE head).
Multi-head prediction. EmerNeRF uses separate heads for color, sky, and shadow predictions. To
maximize the use of dense self-supervision from the static branch, the static and dynamic branches
share the same color head MLPcolor. This color head takes (gs, d) and (gd, d) as input, and outputs
per-point color cs and cd for static and dynamic items, where d is the normalized view direction.
Since the depth of the sky is ill-defined, we follow Rematas et al. (2022) and use a separate sky
head to predict the sky’s color from the frequency-embedded view direction γ(d) , where γ(·) is
a frequency-based positional embedding function, as in Mildenhall et al. (2021). Lastly, as in Wu
et al. (2022), we use a shadow head MLPshadow to depict the shadows of dynamic objects. It outputs
a scalar ρ ∈ [0, 1] for dynamic objects, modulating the color intensity predicted by the static field.
Collectively, we have:
cs = MLPcolor(gs, γ(d))
cd = MLPcolor(gd, γ(d))
(2)
csky = MLPcolor sky(γ(d))
ρ = MLPshadow(gd)
(3)
Rendering. To enable highly-efficient rendering, we use density-based weights to combine results
from the static field and dynamic field:
c =
σs
σs + σd
· (1 − ρ) · cs +
σd
σs + σd
· cd
(4)
To render a pixel, we use K discrete samples {x1, . . . , xK} along its ray to estimate the integral of
color. The final outputs are given by:
ˆC =
K
X
i=1
Tiαici +
 
1 −
K
X
i=1
Tiαi
!
csky
(5)
where Ti = Qi−1
j=1(1 − αj) is the accumulated transmittance and αi = 1 − exp(−σi(xi+1 − xi)) is
the piece-wise opacity.
Dynamic density regularization. To facilitate static-dynamic decomposition, we leverage the fact
that our world is predominantly static. We regularize dynamic density by minimizing the expectation
of the dynamic density σd, which prompts the dynamic field to produce density values only as
needed:
Lσd = E(σd)
(6)
3.2
EMERGENT SCENE FLOW
Scene flow estimation. To capture explicit correspondences between dynamic objects and provide
a link by which to aggregate temporally-displaced features, we introduce an additional scene flow
5
Preprint
field consisting of a hash grid V := Hv(x, t) and a flow predictor MLPv. This flow field maps a
spatial-temporal query point (x, t) to a flow vector v ∈ R3, which transforms the query point to its
position in the next timestep, given by:
v = MLPv(Hv(x, t))
x′ = x + v
(7)
In practice, our flow field predicts both a forward flow vf and a backward flow vb, resulting in a
6-dimensional flow vector for each point.
Multi-frame feature integration. Next, we use the link provided by the predicted scene flow to
integrate features from nearby timesteps, using a simple weighted summation:
g′
d = 0.25 · gd(Hd(x + vb, t − 1)) + 0.5 · gd(Hd(x, t)) + 0.25 · gd(Hd(x + vf, t + 1))
(8)
If not otherwise specified, g′
d is used by default when the flow field is enabled (instead of gd in
Eqs. (2) and (3)). This feature aggregation module achieves three goals: 1) It connects the flow field
to scene reconstruction losses (e.g., RGB loss) for supervision, 2) it consolidates features, denoising
temporal attributes for accurate predictions, and 3) each point is enriched through the shared gradient
of its temporally-linked features, enhancing the quality of individual points via shared knowledge.
Emergent abilities. We do not use any explicit flow supervision to guide EmerNeRF’s flow estima-
tion process. Instead, this capability emerges from our temporal aggregation step while optimizing
scene reconstruction losses (§3.4). Our hypothesis is that only temporally-consistent features benefit
from multi-frame feature integration, and this integration indirectly drives the scene flow field to-
ward optimal solutions — predicting correct flows for all points. Our subsequent ablation studies in
Appendix C.2 confirm this: when the temporal aggregation is disabled or gradients of these nearby
features are stopped, the flow field fails to learn meaningful results.
3.3
VISION TRANSFORMER FEATURE LIFTING
While NeRFs excel at generating high-fidelity color and density fields, they lack in conveying se-
mantic content, constraining their utility for semantic scene comprehension. To bridge this gap,
we lift 2D foundation model features to 4D, enabling crucial autonomous driving perception tasks
such as semantic occupancy prediction. Although previous works might suggest a straightforward
approach (Kerr et al., 2023; Kobayashi et al., 2022), directly lifting features from state-of-the-art
vision transformer (ViT) models has hidden complexities due to positional embeddings (PEs) in
transformer models (Fig. 1 (h-j)). In the following sections, we detail how we enhance EmerNeRF
with a feature reconstruction head, uncover detrimental PE patterns in transformer models, and sub-
sequently mitigate these issues.
Feature reconstruction head. Analogous to the color head, we incorporate a feature head MLPfeat
and a feature sky head MLPfeat sky to predict per-point features f and sky features fsky, given by:
f∗ = MLPfeat(g∗), where ∗ ∈ {s, d}
fsky = MLPfeat sky(γ(d)).
(9)
Similar to the color head, we share the feature head among the static and dynamic branches. Ren-
dering these features similarly follows Eq. (5), given by:
ˆF =
K
X
i=1
Tiαifi +
 
1 −
K
X
i=1
Tiαi
!
fsky
(10)
Positional embedding patterns. We observe pronounced and undesired PE patterns when using
current state-of-the-art foundation models, notably DINOv2 (Oquab et al., 2023) (Fig. 1 (h)). These
patterns remain fixed in images, irrespective of 3D viewpoint changes, breaking 3D multi-view
consistency. Our experiments (§4.3) reveal that these patterns not only impair feature synthesis
results, but also cause a substantial reduction in 3D perception performance.
Shared learnable additive prior. We base our solution on the observation that ViTs extract feature
maps image-by-image and these PE patterns appear (almost) consistently across all images. This
suggests that a single PE feature map might be sufficient to capture this shared artifact. Accord-
ingly, we assume an additive noise model for the PE patterns; that is, they can be independently
subtracted from the original features to obtain PE-free features. With this assumption, we construct
a learnable and globally-shared 2D feature map U to compensate for these patterns. This process
6
Preprint
is depicted in Fig. 2 (b). For a target pixel coordinate (u, v), we first volume-render a PE-free fea-
ture as in Eq. (10). Then, we bilinearly interpolate U and decode the interpolated feature using a
single-layer MLPPE to obtain the PE pattern feature, which is then added to the PE-free feature.
Formally:
ˆF =
K
X
i=1
Tiαifi +
 
1 −
k
X
i=1
Tiαi
!
fsky
|
{z
}
Volume-rendered PE-free feature
+ MLPPE (interp ((u, v) , U))
|
{z
}
PE feature
(11)
The grouped terms render “PE-free” features (Fig. 1 (i)) and “PE” patterns (Fig. 1 (j)), respectively,
with their sum producing the overall “PE-containing” features (Fig. 1 (h)).
3.4
OPTIMIZATION
Loss functions. Our method decouples pixel rays and LiDAR rays to account for sensor asynchro-
nization. For pixel rays, we use an L2 loss for colors Lrgb (and optional semantic features Lfeat), a
binary cross entropy loss for sky supervision Lsky, and a shadow sparsity loss Lshadow. For LiDAR
rays, we combine an expected depth loss with a line-of-sight loss Ldepth, as proposed in Rematas
et al. (2022). This line-of-sight loss promotes an unimodal distribution of density weights along
a ray, which we find is important for clear static-dynamic decomposition. For dynamic regular-
ization, we use a density-based regularization (Eq. 6) to encourage the dynamic field to produce
density values only when absolutely necessary. This dynamic regularization loss is applied to both
pixel rays (Lσd(pixel)) and LiDAR rays (Lσd(LiDAR)). Lastly, we regularize the flow field with a cycle
consistency loss Lcycle. See Appendix A.1 for details. In summary, we minimize:
L = Lrgb + Lsky + Lshadow + Lσd(pixel) + Lcycle + Lfeat
|
{z
}
for pixel rays
+ Ldepth + Lσd(LiDAR)
|
{z
}
for LiDAR rays
(12)
Implementation details. All model implementation details can be found in Appendix A.
4
EXPERIMENTS
In this section, we benchmark the reconstruction capabilities of EmerNeRF against prior meth-
ods, focusing on static and dynamic scene reconstruction, novel view synthesis, scene flow esti-
mation, and foundation model feature reconstruction. Further ablation studies and a discussion of
EmerNeRF’s limitations can be found in Appendices C.2 and C.3, respectively.
Dataset. While there exist many public datasets with AV sensor data (Caesar et al., 2020; Sun et al.,
2020; Caesar et al., 2021), they are heavily imbalanced, containing many simple scenarios with few
to no dynamic objects. To remedy this, we introduce NeRF On-The-Road (NOTR), a balanced
and diverse benchmark derived from the Waymo Open Dataset (Sun et al., 2020). NOTR features
120 unique, hand-picked driving sequences, split into 32 static (the same split as in StreetSurf (Guo
et al., 2023)), 32 dynamic, and 56 diverse scenes across seven challenging conditions: ego-static,
high-speed, exposure mismatch, dusk/dawn, gloomy, rainy, and night. We name these splits Static-
32, Dynamic-32, and Diverse-56, respectively. This dataset not only offers a consistent benchmark
for static and dynamic object reconstruction, it also highlights the challenges of training NeRFs
on real-world AV data. Beyond simulation, our benchmark offers 2D bounding boxes for dynamic
objects, ground truth 3D scene flow, and 3D semantic occupancy—all crucial for driving perception
tasks. Additional details can be found in Appendix B.
4.1
RENDERING
Setup. To analyze performance across various driving scenarios, we test EmerNeRF’s scene recon-
struction and novel view synthesis capabilities on different NOTR splits. For scene reconstruction,
all samples in a log are used for training. This setup probes the upper bound of each method.
For novel view synthesis, we omit every 10th timestep, resulting in 10% novel temporal views for
evaluation. Our metrics include peak signal-to-noise ratio (PSNR) and structural similarity index
(SSIM). For dynamic scenes, we further leverage ground truth bounding boxes and velocity data to
7
Preprint
Table 1: Dynamic and static scene reconstruction performance.
(a) Dynamic-32 Split
Methods
Scene Reconstruction
Novel View Synthesis
Full Image
Dynamic-Only
Full Image
Dynamic-Only
PSNR↑
SSIM↑
PSNR↑
SSIM↑
PSNR↑
SSIM↑
DPSNR↑
SSIM↑
D2NeRF
24.35
0.645
21.78
0.504
24.17
0.642
21.44
0.494
HyperNeRF
25.17
0.688
22.93
0.569
24.71
0.682
22.43
0.554
Ours
28.87
0.814
26.19
0.736
27.62
0.792
24.18
0.670
(b) Static-32 Split
Methods
Static Scene Reconstruction
PSNR↑
SSIM↑
iNGP
24.46
0.694
StreetSurf
26.15
0.753
Ours
29.08
0.803
Table 2: Scene flow estimation on the NOTR Dynamic-32 split.
Methods
EPE3D (m) ↓
Acc5(%) ↑
Acc10(%) ↑
θ (rad) ↓
NSFP (Li et al., 2021a)
0.365
51.76
67.36
0.84
Ours
0.014
93.92
96.27
0.64
identify dynamic objects and compute “dynamic-only” metrics; and we benchmark against Hyper-
NeRF (Park et al., 2021b) and D2NeRF (Wu et al., 2022), two state-of-the-art methods for mod-
eling dynamic scenes. Due to their prohibitive training cost, we only compare against them in the
Dynamic-32 split. On the Static-32 split, we disable our dynamic and flow branches, and compare
against StreetSurf (Guo et al., 2023) and iNGP (M¨uller et al., 2022) (as implemented by Guo et al.
(2023)). We use the official codebases released by these methods, and adapt them to NOTR. To en-
sure a fair comparison, we augment all methods with LiDAR depth supervision and sky supervision,
and disable our feature field. Further details can be found in Appendix A.2.
Dynamic scene comparisons. Table 1 (a) shows that our approach consistently outperforms others
on scene reconstruction and novel view synthesis. We refer readers to Appendix C.1 for qualitative
comparisons. In them, we can see that HyperNeRF (Park et al., 2021b) and D2NeRF (Wu et al.,
2022) tend to produce over-smoothed renderings and struggle with dynamic object representation.
In contrast, EmerNeRF excels in reconstructing high-fidelity static background and dynamic fore-
ground objects, while preserving high-frequency details (evident from its high SSIM and PSNR
values). Despite D2NeRF’s intent to separate static and dynamic elements, it struggles in com-
plex driving contexts and produces poor dynamic object segmentation (as shown in Fig. C.4). Our
method outperforms them both quantitatively and qualitatively. Static scene comparisons. While
static scene representation is not our main focus, EmerNeRF excels in this aspect too, as evidenced
in Table 1 (b). It outperforms state-of-the-art StreetSuRF (Guo et al., 2023) which is designed for
static outdoor scenes. With the capability to model both static and dynamic components, EmerNeRF
can accurately represent more general driving scenes.
4.2
FLOW ESTIMATION
Setup. We assess EmerNeRF on all frames of the Dynamic-32 split, benchmarking against the prior
state-of-the-art, NSFP (Li et al., 2021a). Using the Waymo dataset’s ground truth scene flows, we
compute metrics consistent with Li et al. (2021a): 3D end-point error (EPE3D), calculated as the
mean L2 distance between predictions and ground truth for all points; Acc5, representing the frac-
tion of points with EPE3D less than 5cm or a relative error under 5%; Acc10, indicating the fraction
of points with EPE3D under 10cm or a relative error below 10%; and θ, the average angle error be-
tween predictions and ground truths. When evaluating NSFP (Li et al., 2021a), we use their official
implementation and remove ground points (our approach does not require such preprocessing).
Results. As shown in Table 2, our approach outperforms NSFP across all metrics, with significant
leads in EPE3D, Acc5, and Acc10. While NSFP (Li et al., 2021a) employs the Chamfer distance loss
Fan et al. (2017) to solve scene flow, EmerNeRF achieves significantly better results without any
explicit flow supervision. These properties naturally emerge from our temporal aggregation step.
Appendix C.2 contains additional ablation studies regarding the emergence of flow estimation.
4.3
LEVERAGING FOUNDATION MODEL FEATURES
To investigate the impact of ViT PE patterns on 3D perception and feature synthesis, we instantiate
versions of EmerNeRF with and without our proposed PE decomposition module.
8
Preprint
Table 3: Few-shot semantic occupancy prediction evaluation. We investigate the influence of
positional embedding (PE) patterns on 4D features by evaluating semantic occupancy prediction
performance. We report sample-averaged micro-accuracy and class-averaged macro-accuracy.
PE removed?
ViT model
Static-32
Dynamic-32
Diverse-56
Average of 3 splits
Micro Acc
Macro Acc
Micro Acc
Macro Acc
Micro Acc
Macro Acc
Micro Acc
Macro Acc
No
DINOv1
43.12%
52.71%
47.51%
54.46%
43.19%
51.11%
44.60%
52.76%
Yes
DINOv1
55.02%
57.13%
57.65%
57.77%
54.56%
55.13%
55.74%
56.67%
Relative Improvement
+27.60%
+8.38%
+21.35%
+6.07%
+26.32%
+7.87%
+24.95%
+7.42%
No
DINOv2
38.73%
50.30%
51.43%
57.03%
45.22%
54.37%
45.13%
53.90%
Yes
DINOv2
63.21%
59.41%
65.08%
60.82%
57.86%
59.00%
62.05%
59.74%
Relative Improvement
+63.22%
+18.11%
+26.53%
+6.65%
+27.95%
+8.51%
+37.50%
+10.84%
Table 4: Feature synthesis results. We report the feature-PNSR values under different settings.
PE removed?
ViT model
Static-32
Dynamic-32
Diverse-56
No
DINOv1
23.35
23.37
23.78
Yes
DINOv1
23.57 (+0.23)
23.52 (+0.15)
23.92 (+0.14)
No
DINOv2
21.87
22.34
22.79
Yes
DINOv2
22.70 (+0.83)
22.80 (+0.45)
23.21 (+0.42)
Setup. We evaluate EmerNeRF’s few-shot perception capabilities using the Occ3D dataset (Tian
et al., 2023). Occ3D provides 3D semantic occupancy annotations for the Waymo dataset (Sun
et al., 2020) in voxel sizes of 0.4m and 0.1m (we use 0.1m). For each sequence, we annotate every
10th frame with ground truth information, resulting in 10% labeled data. Occupied coordinates are
input to pre-trained EmerNeRF models to compute feature centroids per class. Features from the
remaining 90% of frames are then queried and classified based on their nearest feature centroid.
We report both micro (sample-averaged) and macro (class-averaged) classification accuracies. All
models are obtained from the scene reconstruction setting, i.e., all views are used for training.
Results. Table 3 compares the performance of PE-containing 4D features to their PE-free counter-
parts. Remarkably, EmerNeRF with PE-free DINOv2 (Oquab et al., 2023) features sees a maximum
relative improvement of 63.22% in micro-accuracy and an average increase of 37.50% over its PE-
containing counterpart. Intriguingly, although the DINOv1 (Caron et al., 2021) model might appear
visually unaffected (Fig. C.5), our results indicate that directly lifting PE-containing features to 4D
space-time is indeed problematic. With our decomposition, PE-free DINOv1 features witness an
average relative boost of 24.95% in micro-accuracy. As another illustration of PE patterns’ impact,
by eliminating PE patterns, the improved performance of DINOv2 over DINOv1 carries over to 3D
perception (e.g., Static32 micro-accuracy).
Feature synthesis results. Table 4 compares the feature-PSNR of PE-containing and PE-free mod-
els, showing marked improvements in feature synthesis quality when using our proposed PE decom-
position method, especially for DINOv2 (Oquab et al., 2023). While DINOv1 (Caron et al., 2021)
appears to be less influenced by PE patterns, our method unveils their presence, further showing that
even seemingly unaffected models can benefit from PE pattern decomposition.
5
CONCLUSION
In this work, we present EmerNeRF, a simple yet powerful approach for learning 4D neural repre-
sentations of dynamic scenes. EmerNeRF effectively captures scene geometry, appearance, motion,
and any additional semantic features by decomposing scenes into static and dynamic fields, learning
an emerged flow field, and optionally lifting foundation model features to a resulting hybrid world
representation. EmerNeRF additionally removes problematic positional embedding patterns that
appear when employing Transformer-based foundation model features. Notably, all of these tasks
(save for foundation model feature lifting) are learned in a self-supervised fashion, without relying
on ground truth object annotations or pre-trained models for dynamic object segmentation or optical
flow estimation. When evaluated on NOTR, our carefully-selected subset of 120 challenging driv-
ing scenes from the Waymo Open Dataset (Sun et al., 2020), EmerNeRF achieves state-of-the-art
performance in sensor simulation, significantly outperforming previous methods on both static and
9
Preprint
dynamic scene reconstruction, novel view synthesis, and scene flow estimation. Exciting areas of
future work include further exploring capabilities enabled or significantly improved by harnessing
foundation model features: few-shot, zero-shot, and auto-labeling via open-vocabulary detection.
ETHICS STATEMENT
This work primarily focuses on autonomous driving data representation and reconstruction. Ac-
cordingly, we use open datasets captured in public spaces that strive to preserve personal privacy
by leveraging state-of-the-art object detection techniques to blur people’s faces and vehicle license
plates. However, these are instance-level characteristics. What requires more effort to manage (and
could potentially lead to greater harm) is maintaining a diversity of neighborhoods, and not only
in terms of geography, but also population distribution, architectural diversity, and data collection
times (ideally repeated traversals uniformly distributed throughout the day and night, for example).
We created the NOTR dataset with diversity in mind, hand-picking scenarios from the Waymo Open
Dataset (Sun et al., 2020) to ensure a diversity of neighborhoods and scenario types (e.g., static,
dynamic). However, as in the parent Waymo Open Dataset, the NOTR dataset contains primarily
urban geographies, collected from only a handful of cities in the USA.
REPRODUCIBILITY STATEMENT
We present our method in §3, experiments and results in §4, implementation details and abla-
tion studies in Appendix A. We benchmark previous approaches and our proposed method us-
ing publicly-available data and include details of the derived dataset in Appendix B. Additional
visualizations, code, models, and data are anonymously available either in the appendix or at
https://emernerf.github.io.
REFERENCES
Shir Amir, Yossi Gandelsman, Shai Bagon, and Tali Dekel. Deep ViT features as dense visual
descriptors. arXiv preprint arXiv:2112.05814, 2021.
Aayush Bansal, Minh Vo, Yaser Sheikh, Deva Ramanan, and Srinivasa Narasimhan. 4d visualiza-
tion of dynamic events from unconstrained multi-view videos. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 5366–5375, 2020.
Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and
Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields.
In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5855–5864,
2021.
Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf:
Anti-aliased grid-based neural radiance fields. arXiv preprint arXiv:2304.06706, 2023.
Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush
Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A multimodal dataset for au-
tonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2020.
Holger Caesar, Juraj Kabzan, Kok Seang Tan, Whye Kit Fong, Eric Wolff, Alex Lang, Luke Fletcher,
Oscar Beijbom, and Sammy Omari. nuPlan: A closed-loop ml-based planning benchmark for
autonomous vehicles. In IEEE/CVF Conference on Computer Vision and Pattern Recognition
Workshop on Autonomous Driving: Perception, Prediction and Planning, 2021.
Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and
Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of
the IEEE/CVF international conference on computer vision, pp. 9650–9660, 2021.
Haoqiang Fan, Hao Su, and Leonidas J Guibas. A point set generation network for 3d object recon-
struction from a single image. In Proceedings of the IEEE conference on computer vision and
pattern recognition, pp. 605–613, 2017.
10
Preprint
Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Matthias
Nießner, and Qi Tian. Fast dynamic radiance fields with time-aware neural voxels. In SIGGRAPH
Asia 2022 Conference Papers, pp. 1–9, 2022.
Jianfei Guo, Nianchen Deng, Xinyang Li, Yeqi Bai, Botian Shi, Chiyu Wang, Chenjing Ding,
Dongliang Wang, and Yikang Li. Streetsurf: Extending multi-view implicit surface reconstruction
to street views. arXiv preprint arXiv:2306.04988, 2023.
Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Lan-
guage embedded radiance fields. In International Conference on Computer Vision (ICCV), 2023.
Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitzmann. Decomposing nerf for editing via
feature field distillation. In Advances in Neural Information Processing Systems, pp. 23311–
23330, 2022.
Ruilong Li, Hang Gao, Matthew Tancik, and Angjoo Kanazawa. Nerfacc: Efficient sampling accel-
erates nerfs. arXiv preprint arXiv:2305.04966, 2023a.
Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim,
Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video
synthesis from multi-view video. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 5521–5531, 2022.
Xueqian Li, Jhony Kaesemodel Pontes, and Simon Lucey. Neural scene flow prior. Advances in
Neural Information Processing Systems, 34:7838–7851, 2021a.
Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural scene flow fields for space-
time view synthesis of dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 6498–6508, 2021b.
Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker, and Noah Snavely. Dynibar: Neu-
ral dynamic image-based rendering. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 4273–4284, 2023b.
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and
Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. Communications
of the ACM, 65(1):99–106, 2021.
Thomas M¨uller.
tiny-cuda-nn, April 2021.
URL https://github.com/NVlabs/
tiny-cuda-nn.
Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics prim-
itives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG), 41(4):1–15,
2022.
Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov,
Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning
robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
Julian Ost, Fahim Mannan, Nils Thuerey, Julian Knodt, and Felix Heide. Neural scene graphs for
dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 2856–2865, 2021.
Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M
Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In Proceedings of
the IEEE/CVF International Conference on Computer Vision, pp. 5865–5874, 2021a.
Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman,
Ricardo Martin-Brualla, and Steven M Seitz. Hypernerf: A higher-dimensional representation for
topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228, 2021b.
Christian Reiser, Rick Szeliski, Dor Verbin, Pratul Srinivasan, Ben Mildenhall, Andreas Geiger, Jon
Barron, and Peter Hedman. Merf: Memory-efficient radiance fields for real-time view synthesis
in unbounded scenes. ACM Transactions on Graphics (TOG), 42(4):1–12, 2023.
11
Preprint
Konstantinos Rematas, Andrew Liu, Pratul P Srinivasan, Jonathan T Barron, Andrea Tagliasacchi,
Thomas Funkhouser, and Vittorio Ferrari. Urban radiance fields. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 12932–12942, 2022.
Nur Muhammad Mahi Shafiullah, Chris Paxton, Lerrel Pinto, Soumith Chintala, and Arthur Szlam.
Clip-fields: Weakly supervised semantic fields for robotic memory. arXiv preprint arXiv: Arxiv-
2210.05663, 2022.
Prafull Sharma, Ayush Tewari, Yilun Du, Sergey Zakharov, Rares Andrei Ambrus, Adrien Gaidon,
William T Freeman, Fredo Durand, Joshua B Tenenbaum, and Vincent Sitzmann. Neural ground-
plans: Persistent neural scene representations from a single image. In The Eleventh International
Conference on Learning Representations, 2022.
Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui,
James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for au-
tonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pp. 2446–2454, 2020.
Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P Srini-
vasan, Jonathan T Barron, and Henrik Kretzschmar. Block-nerf: Scalable large scene neural
view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 8248–8258, 2022.
Xiaoyu Tian, Tao Jiang, Longfei Yun, Yue Wang, Yilun Wang, and Hang Zhao. Occ3d: A large-scale
3d occupancy prediction benchmark for autonomous driving. arXiv preprint arXiv:2304.14365,
2023.
Nikolaos Tsagkas, Oisin Mac Aodha, and Chris Xiaoxuan Lu.
Vl-fields: Towards language-
grounded neural implicit spatial representations. arXiv preprint arXiv:2305.12427, 2023.
Haithem Turki, Jason Y Zhang, Francesco Ferroni, and Deva Ramanan.
Suds: Scalable urban
dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 12375–12385, 2023.
Liao Wang, Jiakai Zhang, Xinhang Liu, Fuqiang Zhao, Yanshun Zhang, Yingliang Zhang, Minye
Wu, Jingyi Yu, and Lan Xu. Fourier plenoctrees for dynamic radiance field rendering in real-time.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.
13524–13534, 2022.
Qianqian Wang, Yen-Yu Chang, Ruojin Cai, Zhengqi Li, Bharath Hariharan, Aleksander Holynski,
and Noah Snavely. Tracking everything everywhere all at once. arXiv preprint arXiv:2306.05422,
2023a.
Zian Wang, Tianchang Shen, Jun Gao, Shengyu Huang, Jacob Munkberg, Jon Hasselgren, Zan
Gojcic, Wenzheng Chen, and Sanja Fidler. Neural fields meet explicit geometric representations
for inverse rendering of urban scenes. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 8370–8380, 2023b.
Tianhao Wu, Fangcheng Zhong, Andrea Tagliasacchi, Forrester Cole, and Cengiz Oztireli. Dˆ 2nerf:
Self-supervised decoupling of dynamic and static objects from a monocular video. Advances in
Neural Information Processing Systems, 35:32653–32666, 2022.
Ze Yang, Yun Chen, Jingkang Wang, Sivabalan Manivasagam, Wei-Chiu Ma, Anqi Joyce Yang, and
Raquel Urtasun. Unisim: A neural closed-loop sensor simulator. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 1389–1399, 2023.
12
Preprint
A
IMPLEMENTATION DETAILS
In this section, we discuss the implementation details of EmerNeRF. Our code is publicly available,
and the pre-trained models will be released upon request. See emernerf.github.io for more
details.
A.1
EMERNERF IMPLEMENTATION DETAILS
A.1.1
DATA PROCESSING
Data source. Our sequences are sourced from the waymo open dataset scene flow1 ver-
sion, which augments raw sensor data with point cloud flow annotations. For camera images, we
employ three frontal cameras: FRONT LEFT, FRONT, and FRONT RIGHT, resizing them to a reso-
lution of 640×960 for both training and evaluation. Regarding LiDAR point clouds, we exclusively
use the first return data (ignoring the second return data). We sidestep the rolling shutter issue in
LiDAR sensors for simplicity and leave it for future exploration. Dynamic object masks are derived
from 2D ground truth camera bounding boxes, with velocities determined from the given metadata.
Only objects exceeding a velocity of 1 m/s are classified as dynamic, filtering out potential sen-
sor and annotation noise. For sky masks, we utilize the Mask2Former-architectured ViT-Adapter-L
model pre-trained on ADE20k. Note that, the dynamic object masks and point cloud flows are used
for evaluation only.
Foundation model feature extraction. We employ the officially released checkpoints of DINOv2
Oquab et al. (2023) and DINOv1 (Caron et al., 2021), in conjunction with the feature extractor
implementation from Amir et al. (2021). For DINOv1, we utilize the ViT-B/16, resizing images to
640×960 and modifying the model’s stride to 8 to further increase the resolution of extracted feature
maps. For DINOv2, we use the ViT-B/14 variant, adjusting image dimensions to 644×966 and using
a stride of 7. Given the vast size of the resultant feature maps, we employ PCA decomposition to
reduce the feature dimension from 768 to 64 and normalize these features to the [0, 1] range.
A.1.2
EMERNERF
Representations. We build all our scene representations based on iNGP (M¨uller et al., 2022) from
tiny-cuda-nn (M¨uller, 2021), and use nerfacc toolkit (Li et al., 2023a) for acceleration. Fol-
lowing Barron et al. (2023), our static hash encoder adopts a resolution spanning 24 to 213 over
10 levels, with a fixed feature length of 4 for all hash entries. Features at each level are capped at
220 in size. With these settings, our model comprises approximately 30M parameters — a saving
of 18M compared to the StreetSurf’s SDF representation (Guo et al., 2023). For our dynamic hash
encoder, we maintain a similar architecture, but with a maximum hash feature map size of 218. Our
flow encoder is identical to the dynamic encoder. To address camera exposure variations in the
wild, 16-dimensional appearance embeddings are applied per image for scene reconstruction and
per camera for novel view synthesis. While our current results are promising, we believe that larger
hash encoders and MLPs could further enhance performance. See our code for more details.
Positional embedding (PE) patterns. We use a learnable feature map, denoted as U, with dimen-
sions 80×120×32 (H ×W ×C) to accommodate the positional embedding patterns, as discussed
in the main text. To decode the PE pattern for an individual pixel located at (u, v), we first sample
a feature vector from U using F.grid sample. Subsequently, a linear layer decodes this feature
vector to produce the final PE features.
Scene range. To define the axis-aligned bounding box (AABB) of the scene, we utilize LiDAR
points. In practice, we uniformly subsample the LiDAR points by a factor of 4 and find the scene
boundaries by computing the 2% and 98% percentiles of the world coordinates within the LiDAR
point cloud. However, the LiDAR sensor typically covers only a 75-meter radius around the vehicle.
Consequently, an unrestricted contraction mechanism is useful to ensure better performance. Fol-
lowing the scene contraction method detailed in Reiser et al. (2023), we use a piecewise-projective
contraction function to project the points falling outside the determined AABB.
1console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow
13
Preprint
Multi-level sampling. In line with findings in Mildenhall et al. (2021); Barron et al. (2021), we
observe that leveraging extra proposal networks enhances both rendering quality and geometry esti-
mation. Our framework integrates a two-step proposal sampling process, using two distinct iNGP-
based proposal models. In the initial step, 128 samples are drawn using the first proposal model
which consists of 8 levels, each having a 1-dim feature. The resolution for this model ranges from
24 to 29, and each level has a maximum hash map capacity of 220. For the subsequent sampling
phase, 64 samples are taken using the second proposal model. This model boasts a maximum res-
olution of 211, but retains the other parameters of the first model. To counteract the “z-aliasing”
issue—particularly prominent in driving sequences with thin structures like traffic signs and light
poles, we further incorporate the anti-aliasing proposal loss introduced by Barron et al. (2023) dur-
ing proposal network optimization. A more thorough discussion on this is available in Barron et al.
(2023). Lastly, we do not employ spatial-temporal proposal networks, i.e., we don’t parameterize
the proposal networks with a temporal dimension. Our current implementation already can capture
temporal variations from the final scene fields, and we leave integrating a temporal dimension in
proposal models for future exploration. For the final rendering, 64 points are sampled from the
scene fields.
Pixel importance sampling. Given the huge image sizes, we prioritize hard examples for efficient
training. Every 2k steps, we render RGB images at a resolution reduced by a factor of 32 and
compute the color errors against the ground truths. For each training batch, 25% of the training
rays are sampled proportionally based on these color discrepancies, while the remaining 75% are
uniformly sampled. This strategy is similar to Wang et al. (2023a) and Guo et al. (2023).
A.1.3
OPTIMIZATION
All components in EmerNeRF are trained jointly in an end-to-end manner.
Loss functions. As we discussed in §3.4, our total loss function is
L = Lrgb + Lsky + Lshadow + Lσd(pixel) + Lcycle + Lfeat
|
{z
}
for pixel rays
+ Ldepth + Lσd(LiDAR)
|
{z
}
for LiDAR rays
(A1)
With r representing a ray and Nr its total number, the individual loss components are defined as:
1. RGB loss (Lrgb): Measures the difference between the predicted color ( ˆC(r)) and the ground
truth color (C(r)) for each ray.
Lrgb = 1
Nr
X
r
|| ˆC(r) − C(r)||2
2
(A2)
2. Sky loss (Lsky): Measures the discrepancy between the predicted opacity of rendered rays and the
actual sky masks. Specifically, sky regions should exhibit transparency. The binary cross entropy
(BCE) loss is employed to evaluate this difference. In the equation, ˆO(r) is the accumulated
opacity of ray r as in Equation (5). M(r) is the ground truth mask with 1 for the sky region and
0 otherwise.
Lsky = 0.001 · 1
Nr
X
r
BCE

ˆO(r), 1 − M(r)

(A3)
3. Shadow loss (Lshadow): Penalizes the accumulated squared shadow ratio, following Wu et al.
(2022).
Lshadow = 0.01 · 1
Nr
X
r
 K
X
i=1
Tiαiρ2
i
!
(A4)
4. Dynamic regularization (Lσd(pixel) and Lσd(LiDAR)): Penalizes the mean dynamic density of
all points across all rays. This encourages the dynamic branch to generate density only when
necessary.
Lσ⌈ = 0.01 · 1
Nr
X
r
1
K
K
X
i=1
σd(r, i)
(A5)
14
Preprint
5. Cycle consistency regularization (Lcycle): Self-regularizes the scene flow prediction. This loss
encourages the congruence between the forward scene flow at time t and its corresponding back-
ward scene flow at time t + 1.
Lcycle = 0.01
2 E
h
[sg(vf(x, t)) + v′
b (x + vf(x, t), t + 1)]2 
sg(vb(x, t)) + v′
f(x + vb(x, t), t − 1)
2i
(A6)
where vf(x, t) denotes forward scene flow at time t, v′
b(x+vf(x, t), t+1) is predicted backward
scene flow at the forward-warped position at time t + 1, sg means stop-gradient operation, and E
represents the expectation, i.e., averaging over all sample points.
6. Feature loss (Lfeat): Measures the difference between the predicted semantic feature ( ˆF(r)) and
the ground truth semantic feature (F(r)) for each ray.
Lfeat = 0.5 · 1
Nr
|| ˆF(r) − F(r)||2
2
(A7)
7. Depth Loss (Ldepth): Combines the expected depth loss and the line-of-sight loss, as described in
Rematas et al. (2022). The expected depth loss ensures the depth predicted through the volumetric
rendering process aligns with the LiDAR measurement’s depth. The line-of-sight loss includes
two components: a free-space regularization term that ensures zero density for points before the
LiDAR termination points and a near-surface loss promoting density concentration around the
termination surface. With a slight notation abuse, we have:
Lexp depth = Er
h
|| ˆZ(r) − Z(r)||2
2
i
(A8)
Lline-of-sight = Er
"Z Z(r)−ϵ
tn
w(t)2dt
#
+ Er
"Z Z(r)+ϵ
Z(r)−ϵ
(w(t) − Kϵ (t − Z(r)))2
#
(A9)
Ldepth = Lexp depth + 0.1 · Lline-of-sight
(A10)
where ˆZ(r) represents rendered depth values and Z(r) stands for the ground truth LiDAR range
values. Here, the variable t indicates an offset from the origin towards the ray’s direction, dif-
ferentiating it from the temporal variable t discussed earlier. w(t) specifies the blending weights
of a point along the ray. Kϵ(x) = N(0, (ϵ/3)2) represents a kernel integrating to one, where
N is a truncated Gaussian. The parameter ϵ determines the strictness of the line-of-sight loss.
Following the suggestions in Rematas et al. (2022), we linearly decay ϵ from 6.0 to 2.5 during
the whole training process.
Training. We train our models for 25k iterations using a batch size of 8196. In static scenarios,
we deactivate the dynamic and flow branches. Training durations on a single A100 GPU are as
follows: for static scenes, feature-free training requires 33 minutes, while the feature-embedded
approach takes 40 minutes. Dynamic scene training, which incorporates the flow field and feature
aggregation, extends the durations to 2 hours for feature-free and 2.25 hours for feature-embedded
representations. To mitigate excessive regularization when the geometry prediction is not reliable,
we enable line-of-sight loss after the initial 2k iterations and subsequently halve its coefficient every
5k iterations.
A.2
BASLINE IMPLEMENTATIONS
For HyperNeRF (Park et al., 2021b) and D2NeRF (Wu et al., 2022), we modify their official JAX
implementations to fit our NOTR dataset. Both models are trained for 100k iterations with a batch
size of 4096. Training and evaluation for each model take approximately 4 hours on 4 A100 GPUs
per scene. To ensure comparability, we augment both models with a sky head and provide them
with the same depth and sky supervision as in our model. However, since neither HyperNeRF nor
D2NeRF inherently supports separate sampling of pixel rays and LiDAR rays, we project LiDAR
point clouds onto the image plane and apply an L2 loss between predicted depth and rendered
depth. We compute a scale factor from the AABBs derived from LiDAR data to ensure scenes are
encapsulated within their predefined near-far range. For StreetSuRF (Guo et al., 2023), we adopt
their official implementation but deactivate the monocular “normal” supervision for alignment with
our setup. Additionally, to ensure both StreetSuRF and EmerNeRF use the same data, we modify
their code to accommodate our preprocessed LiDAR rays.
15
Preprint
(1) Static
(2) Dynamic
(3) Ego-static
(4) High-speed
(5) Rainy
(6) Dusk/Dawn
(7) Nighttime
(8) Mismatch exposure
(9) Gloomy
Figure B.1: Samples from the NOTR Benchmark. This comprehensive benchmark contains (1) 32
static scenes, (2) 32 dynamic scenes, and 56 additional scenes across seven categories: (3) ego-static,
(4) high-speed, (5) rainy, (6) dusk/dawn, (7) nighttime, (8) mismatched exposure, and (9) gloomy
conditions. We include LiDAR visualization in each second row and sky masks in each third row.
B
NERF ON-THE-ROAD (NOTR) DATASET
As neural fields gain more attention in autonomous driving, there is an evident need for a compre-
hensive dataset that captures diverse on-road driving scenarios for NeRF evaluations. To this end,
we introduce NeRF On-The-Road (NOTR) dataset, a benchmark derived from the Waymo Open
Dataset (Sun et al., 2020). NOTR features 120 unique driving sequences, split into 32 static scenes,
32 dynamic scenes, and 56 scenes across seven challenging conditions: ego-static, high-speed, ex-
posure mismatch, dusk/dawn, gloomy, rainy, and nighttime. Examples are shown in Figure B.1.
Beyond images and point clouds, NOTR provides additional resources pivotal for driving percep-
tion tasks: bounding boxes for dynamic objects, ground-truth 3D scene flow, and 3D semantic
occupancy. We hope this dataset can promote NeRF research in driving scenarios, extending the
applications of NeRFs from mere view synthesis to motion understanding, e.g., 3D flows, and scene
comprehension, e.g., semantics.
Regarding scene classifications, our static scenes adhere to the split presented in StreetSuRF (Guo
et al., 2023), which contains clean scenes with no moving objects. The dynamic scenes, which
are frequently observed in driving logs, are chosen based on lighting conditions to differentiate
them from those in the “diverse” category. The Diverse-56 samples may also contain dynamic
objects, but they are split primarily based on the ego vehicle’s state (e.g., ego-static, high-speed,
camera exposure mismatch), weather condition (e.g., rainy, gloomy), and lighting difference (e.g.,
nighttime, dusk/dawn). We provide the sequence IDs of these scenes in our codebase.
16
Preprint
Figure C.1: Qualitative scene reconstruction comparisons.
C
ADDITIONAL RESULTS
C.1
QUALITATIVE RESULTS
Qualitative comparison.
Figures C.1 and C.2 show qualitative comparisons between our
EmerNeRF and previous methods under the scene reconstruction setting, while Figure C.3 high-
lights the enhanced static-dynamic decomposition of our method compared to D2NeRF (Wu et al.,
2022). Moreover, Figure C.4 illustrates our method’s superiority in novel view synthesis tasks
against HyperNeRF (Park et al., 2021b) and D2NeRF (Wu et al., 2022). Our method consistently
delivers more realistic and detailed renders. Notably, HyperNeRF does not decompose static and
dynamic components; it provides only composite renders, while our method not only renders high-
fidelity temporal views but also precisely separates static and dynamic elements. Furthermore, our
method introduces the novel capability of generating dynamic scene flows.
C.2
ABLATION STUDIES
Table C.1 provides ablation studies to understand the impact of other components on scene re-
construction, novel view synthesis, and scene flow estimation. For these ablation experiments, all
models are trained for 8k iterations, a shorter duration compared to the 25k iterations in the primary
experiments. From our observations: (a) Using a full 4D iNGP without the static field results in
the worst results, a consequence of the lack of multi-view supervision. (b-e) Introducing hybrid
representations consistently improves the results. (c) Omitting the temporal aggregation step or (d)
freezing temporal feature gradients (stop the gradients of gt−1
d
and gt+1
d
in Fig. 2) negates the emer-
gence of flow estimation ability, as evidenced in the final column. Combining all these settings
yields the best results.
17
Preprint
Figure C.2: Qualitative scene reconstruction comparisons.
Table C.1: Ablation study.
Setting
Scene Reconstruction
Novel View Synthesis
Scene Flow estimation
Full Image
Dynamic-Only
Full Image
Dynamic-Only
Flow
PSNR↑
PSNR↑
PSNR↑
PSNR↑
Acc5(%) ↑
(a) 4D-Only iNGP
26.55
22.30
26.02
21.03
-
(b) no flow
26.92
23.82
26.33
23.81
-
(c) no temporal aggregation
26.95
23.90
26.60
23.98
4.53%
(d) freeze temporally displaced features before aggregation
26.93
24.02
26.78
23.81
3.87%
(e) ours default
27.21
24.41
26.93
24.07
89.74%
18
Preprint
Figure C.3: Scene decomposition comparisons. Note that we utilize a green background to blend
dynamic objects, whereas D2NeRF’s results are presented with a white background.
C.3
LIMITATIONS
Consistent with other methods, EmerNeRF does not optimize camera poses and is prone to rolling
shutter effects of camera and LiDAR sensors. Future work to address this issue can investigate joint
optimization of pixel-wise camera poses, and compensation for LiDAR rolling shutter alongside
scene representations. Moreover, the balance between geometry and rendering quality remains a
trade-off and needs further study. Lastly, EmerNeRF occasionally struggles with estimating the
motion of slow-moving objects when the ego-vehicle is moving fast—a challenge exacerbated by
the limited observations. We leave these for future research.
C.4
VISUALIZATIONS
19
Preprint
(a) GT Image
(b) Ours- Novel View Synthesis
(c) Ours- Novel Dynamic RGB Decomposition
(d) Ours- Novel Scene Flow Synthesis
(e) HyperNeRF
(f) D2NeRF
(g) D2NeRF Novel Dynamic RGB Decomposition
Figure C.4: Qualitative novel temporal view comparison.
20
Preprint
(a) GT RGB Images
(b) GT DINOv2
(c) Reconstructed DINOv2
(d) Decomposed PE-free DINOv2
(e) Decomposed DINOv2 PE patterns
(f) GT DINOv1
(g) Reconstructed DINOv1
(h) Decomposed PE-free DINOv1
(i) Decomposed DINOv1 PE patterns
Figure C.5: Different positional embedding patterns in DINOv1 (Caron et al., 2021) and DINOv2
models(Oquab et al., 2023)
21
Preprint
(a) GT RGB
(b) Rendered RGB
(c) Decomposed Dynamic Depth
(d) Emerged Scene Flow
(e) GT DINOv2 Features
(h) Stacking dynamic RGB on Decomposed PE-Free Static DINOV2 Features
(g) Decomposed PE Patterns
(f) Reconstructed DINOv2 Features
Figure C.6: Scene reconstruction visualizations of EmerNeRF. We show (a) GT RGB images,
(b) reconstructed RGB images, (c) decomposed dynamic depth, (d) emerged scene flows, (e) GT
DINOv2 features, (f) reconstructed DINOv2 features, and (g) decomposed PE patterns. (h) We also
stack colors of dynamic objects onto decomposed PE-free static DINOv2 features.
22
Preprint
(a) GT RGB
(b) Rendered RGB
(c) Decomposed Dynamic Depth
(d) Emerged Scene Flow
(e) GT DINOv2 Features
(h) Stacking dynamic RGB on Decomposed PE-Free Static DINOV2 Features
(g) Decomposed PE Patterns
(f) Reconstructed DINOv2 Features
Figure C.7: Scene reconstruction visualizations of EmerNeRF under different lighting conditions.
We show (a) GT RGB images, (b) reconstructed RGB images, (c) decomposed dynamic depth, (d)
emerged scene flows, (e) GT DINOv2 features, (f) reconstructed DINOv2 features, and (g) decom-
posed PE patterns. (h) We also stack colors of dynamic objects onto decomposed PE-free static
DINOv2 features. EmerNeRF works well under dark environments (left) and discerns challenging
scene flows in complex environments (right). Colors indicate scene flows’ norms and directions.
23
Preprint
(a) GT RGB
(b) Rendered RGB
(c) Decomposed Dynamic Depth
(d) Emerged Scene Flow
(e) GT DINOv2 Features
(h) Stacking dynamic RGB on Decomposed PE-Free Static DINOV2 Features
(g) Decomposed PE Patterns
(f) Reconstructed DINOv2 Features
Figure C.8: Scene reconstruction visualizations of EmerNeRF under differet lighting and weather
conditions. We show (a) GT RGB images, (b) reconstructed RGB images, (c) decomposed dynamic
depth, (d) emerged scene flows, (e) GT DINOv2 features, (f) reconstructed DINOv2 features, and
(g) decomposed PE patterns. (h) We also stack colors of dynamic objects colors onto decomposed
PE-free static DINOv2 features. EmerNeRF works well under gloomy environments (left) and
discerns fine-grained speed information (right).
24
"""
    ]
    
    prompts: List[str] = [
        """
        Summarize this abstract.
We present EmerNeRF, a simple yet powerful approach for learning spatial-temporal representations of dynamic driving scenes. Grounded in neural fields, EmerNeRF simultaneously captures scene geometry, appearance, motion, and semantics via self-bootstrapping. EmerNeRF hinges upon two core components: First, it stratifies scenes into static and dynamic fields. This decomposition emerges purely from self-supervision, enabling our model to learn from general, in-the-wild data sources. Second, EmerNeRF parameterizes an induced flow field from the dynamic field and uses this flow field to further aggregate multi-frame features, amplifying the rendering precision of dynamic objects. Coupling these three fields (static, dynamic, and flow) enables EmerNeRF to represent highly-dynamic scenes self-sufficiently, without relying on ground truth object annotations or pre-trained models for dynamic object segmentation or optical flow estimation. Our method achieves state-of-the-art performance in sensor simulation, significantly outperforming previous methods when reconstructing static (+2.93 PSNR) and dynamic (+3.70 PSNR) scenes. In addition, to bolster EmerNeRF's semantic generalization, we lift 2D visual foundation model features into 4D space-time and address a general positional bias in modern Transformers, significantly boosting 3D perception performance (e.g., 37.50% relative improvement in occupancy prediction accuracy on average). Finally, we construct a diverse and challenging 120-sequence dataset to benchmark neural fields under extreme and highly-dynamic settings.
"""
    ]

    prompts2: List[str] = [
        "Tell me about Elon Musk"
    ]

    prompts_old: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
