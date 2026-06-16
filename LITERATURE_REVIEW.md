# SOTA Literature Review: CNN Convolutions vs. Exact String Matching Algorithms

**Date**: 2026-03-28
**Scope**: 2015-2026 (focus on 2019-2026)

---

## Table of Contents

1. [Q1: Are CNN Learned Weights Analogous to Failure/Prefix Tables?](#q1)
2. [Q2: Can We Preprocess the INPUT to Create Content-Aware Stride?](#q2)
3. [Cross-Cutting Surveys](#surveys)
4. [Synthesis and Open Problems](#synthesis)

---

<a id="q1"></a>
## Q1: Are CNN Learned Weights Analogous to Failure/Prefix Tables (KMP Failure Function, Z-Array)?

### 1.1 Direct Answer: The Gap in the Literature

**Critical finding: No paper directly establishes a formal analogy between CNN learned filters and KMP failure functions / Z-arrays.** This is a genuine gap. The literature approaches this question from multiple adjacent angles but never makes the direct connection. Below are the closest lines of work.

---

### 1.2 CNN Filters as Template Matchers / Matched Filters

#### Key Concept
The most basic connection is well-established: 1D convolution IS cross-correlation (template matching). A CNN filter of size k sliding over an input with stride 1 computes the same operation as scanning a "pattern" of length k over a "text." The output activation map encodes where the pattern matches.

#### Foundational Papers

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| **Zeiler & Fergus, "Visualizing and Understanding Convolutional Networks"** | 2014 | ECCV | Deconvolution-based visualization showing CNN filters learn hierarchical templates (edges -> textures -> parts -> objects). Showed first-layer filters are Gabor-like edge detectors. ~15,000+ citations. |
| **Gavrikov & Keuper, "CNN Filter DB: An Empirical Investigation of Trained Convolutional Filters"** | 2022 | CVPR (Oral) | Collected 1.4 billion 3x3 filters from hundreds of trained CNNs. Found strong distributional regularities across architectures, datasets, and tasks. Filters converge to a surprisingly small set of canonical patterns. |

#### What This Tells Us About Q1
CNN filters learn template patterns, but the analogy to KMP's failure function is NOT about what the filter detects -- it is about what happens at MISMATCH positions. KMP's failure function encodes "upon mismatch at position j, shift the pattern so that position failure[j] aligns." Standard CNN convolution has NO such mechanism: it simply slides by stride=1 regardless of match/mismatch. **The failure function is about SKIP logic, not detection logic.**

---

### 1.3 CNN Filters as Position Weight Matrices (PWMs) in Genomics

This is the strongest existing analogy to Q1, because in genomics, 1D CNN filters are explicitly shown to learn the same patterns as classical PWM motif scanners.

#### Key Papers

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| **Alipanahi et al., "Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning" (DeepBind)** | 2015 | Nature Biotechnology | First to show CNN filters learn PWM-equivalent motifs for protein-DNA binding. Single conv layer + pooling. ~3,000+ citations. |
| **Koo & Eddy, "Representation Learning of Genomic Sequence Motifs with Convolutional Neural Networks"** | 2019 | PLOS Computational Biology | Rigorous analysis showing CNN first-layer filters converge to PWMs. Developed methods to extract PWMs from learned filters by collecting maximally-activating subsequences. |
| **Novakovsky et al., "ExplaiNN: Interpretable and Transparent Neural Networks for Genomics"** | 2023 | Genome Biology | Architecture where each unit is an independent single-filter CNN, making each filter directly interpretable as a PWM. Showed these are equivalent to de novo motif discovery tools. Performance comparable to black-box CNNs. |
| **Avsec et al. (DeepMind), "Effective Gene Expression Prediction from Sequence by Integrating Long-Range Interactions" (Enformer)** | 2021 | Nature Methods | Transformer + CNN hybrid for gene expression. Conv layers still learn motif-like patterns. |
| **Variable Convolutional Layer (vConv)** | 2019-2023 | bioRxiv/various | Novel convolution layer that dynamically learns kernel length from data. Directly addresses variable-length motif detection. "In-place replacement" for canonical conv layers. |

#### What This Tells Us About Q1
PWM scanning IS pattern matching: slide a weight matrix over a sequence and compute scores. CNN filters learn equivalent PWMs. But PWMs (and CNN filters) are the PATTERN itself, not the failure function. The failure function is the META-information about the pattern (its self-overlap structure). **No paper in genomics examines whether multi-layer CNNs implicitly learn failure-function-like skip logic.**

---

### 1.4 Neural Networks and Formal Language Theory

This line of work asks: what class of languages can different neural architectures recognize?

#### Key Papers

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| **Merrill, "Sequential Neural Networks as Automata"** | 2019 | ACL Workshop (Deep Learning and Formal Languages) | **Proved that shallow CNNs can recognize at most strictly local languages (subregular hierarchy).** This is BELOW regular languages, meaning CNNs cannot even simulate a finite automaton. Key theoretical limitation. |
| **Weiss et al., "On the Practical Computational Power of Finite Precision RNNs for Language Recognition"** | 2018 | ACL | Simple RNNs recognize regular languages (upper bound). LSTMs can handle counter languages (beyond regular). GRUs are no more powerful than feedforward nets. |
| **Ackerman & Cybenko, "A Survey of Neural Networks and Formal Languages"** | 2020 | arXiv | Comprehensive survey. Notes even deep CNNs likely cannot exceed the power of shallow ones for language recognition. |
| **Klein & Barron, "Comparing Cognition Across Major Transitions Using the Hierarchy of Formal Automata"** | 2024 | WIREs Cognitive Science | Reviews the above results: "CNNs cannot recognize regular grammars; their power is at best enough to capture strictly local languages." |
| **Merrill, "Formal Language Theory Meets Modern NLP"** | 2021 | arXiv survey | Extended survey connecting transformers, RNNs, and CNNs to the Chomsky hierarchy. |

#### What This Tells Us About Q1
**This is the most theoretically damaging result for Q1.** If CNNs can only recognize strictly local languages (a proper subset of regular languages), they provably CANNOT implement the full KMP algorithm, which requires finite-state memory to track the failure function. KMP's failure function encodes the self-overlap structure of the pattern, which requires at minimum regular-language-level computation. A single CNN layer doing convolution is fundamentally a sliding window -- it has no memory of previous match attempts.

**However**: Multi-layer CNNs with pooling, skip connections, and nonlinearities MAY transcend this limitation. The formal results apply to restricted architectures. This is an open question.

---

### 1.5 Neural Algorithmic Reasoning (Learning to Execute Algorithms)

This line of work directly trains neural networks to execute classical algorithms, including string matching.

#### Key Papers

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| **Velickovic & Blundell, "Neural Algorithmic Reasoning"** | 2021 | Patterns (Cell Press) | Position paper arguing NNs should learn to mimic algorithms. Proposed "algorithmic alignment" principle: the architecture should structurally match the algorithm. ~250+ citations. |
| **Velickovic et al., "The CLRS Algorithmic Reasoning Benchmark"** | 2022 | ICML | **Benchmark includes naive string matcher AND KMP string matcher.** GNN-based processors trained to execute 30 algorithms from CLRS textbook. String matching is one of the hardest tasks. KMP has a "multiplier" of 64x to compensate for difficulty. |
| **Ibarz et al., "A Generalist Neural Algorithmic Learner"** | 2022 | LoG | Single GNN model learning all 30 CLRS algorithms. Performance on string matching algorithms remains below other algorithm categories. |
| **Rodionov et al., "Neural Algorithmic Reasoning Without Intermediate Supervision"** | 2023 | NeurIPS | Removes need for step-by-step supervision. String matching still challenging. |
| **"SALSA-CLRS: A Sparse and Scalable Benchmark"** | 2024 | NeurIPS | Extended CLRS to 100x larger graphs with sparse execution. |
| **"Learning to Execute Graph Algorithms Exactly with GNNs"** | 2025 | arXiv | Trains shared local MLPs to exactly execute local instructions. |
| **"Discrete Neural Algorithmic Reasoning"** | 2024 | arXiv | Uses discrete latent representations for better algorithmic reasoning. |
| **"Parallel Algorithms Align with Neural Execution"** | 2023 | LoG | Shows parallel algorithms are easier for GNNs to learn than sequential ones. **KMP is inherently sequential, which partly explains why it is hard for GNNs.** |

#### What This Tells Us About Q1
- GNNs CAN be trained to execute KMP, but it remains one of the harder algorithms in the benchmark.
- The failure function is learned as part of the intermediate computation, but it is encoded in the GNN's message-passing state, NOT in convolution filters.
- **The architecture that learns KMP is a GNN (message-passing network), NOT a CNN.** This aligns with the formal language theory results: CNNs lack the computational power for KMP.
- The "algorithmic alignment" principle suggests: to learn KMP, you need an architecture whose computation graph matches KMP's structure (sequential state updates + table lookups).

#### Key Researchers
- **Petar Velickovic** (Google DeepMind) -- leads neural algorithmic reasoning
- **Andrea Banino, Charles Blundell** (Google DeepMind) -- CLRS benchmark
- **William Merrill** (Allen AI / NYU) -- formal language theory + neural networks
- **Gail Weiss** (Technion) -- computational power of neural architectures

---

### 1.6 Summary for Q1

| Aspect | Status | Key Finding |
|--------|--------|-------------|
| CNN filters = templates/PWMs | Well-established | Filters learn pattern detectors equivalent to template matching |
| CNN filters = failure function | **NO EXISTING WORK** | No paper draws this analogy |
| CNN can implement KMP | **Provably limited** | Shallow CNNs limited to strictly local languages (below regular). Cannot implement failure function logic. |
| NN can learn KMP | Yes, with GNNs | CLRS benchmark shows GNNs can learn KMP, but it remains hard |
| Multi-layer CNN with skip connections = failure function | **OPEN QUESTION** | Deep residual CNNs MIGHT learn skip-like behavior, but no formal analysis exists |

**SOTA Assessment**: The direct Q1 analogy (CNN weights = failure tables) is **not supported by existing theory or experiments**. CNN filters ARE pattern templates (analogous to the pattern P itself), but they do NOT encode the failure function (which is meta-information about P's self-overlap). The failure function requires sequential state, which CNNs lack. This makes Q1 a **novel research question** with potential for a strong negative result (proving CNN weights cannot encode failure functions) or a nuanced positive result (showing that multi-layer CNN activations, not weights, implicitly compute something failure-function-like).

---

<a id="q2"></a>
## Q2: Can We Preprocess the INPUT to Create Content-Aware Stride?

### 2.1 Direct Answer: Partially Explored, Major Gap Remains

Several lines of work address input-dependent computation in CNNs, but **no paper frames this as "building a Z-table from the input to modify stride."** The closest approaches are:

---

### 2.2 Deformable Convolutions (Input-Dependent Sampling)

This is the most mature line of work addressing input-dependent spatial processing.

#### Key Papers

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| **Dai et al., "Deformable Convolutional Networks" (DCNv1)** | 2017 | ICCV | Introduced learnable offsets for convolution sampling positions. Each output position learns where to sample from the input. ~5,000+ citations. |
| **Zhu et al., "Deformable ConvNets v2"** | 2019 | CVPR | Added modulation (amplitude weights) to offsets. Better training. ~2,500+ citations. |
| **DCNv3 (InternImage)** | 2023 | CVPR | Combined sparse attention + convolution. Dynamic range + input-dependent weights. Sliding window with learnable sampling within each window. |
| **Xiong et al., "Efficient Deformable ConvNets: DCNv4"** | 2024 | CVPR | Removed softmax normalization for better expressiveness. 3x faster than DCNv3. SOTA on ImageNet, COCO, ADE20K. |

#### Relevance to Q2
Deformable convolutions allow input-dependent SAMPLING POSITIONS but not input-dependent STRIDE. The kernel still slides with a fixed stride; what changes is WHERE within the kernel's receptive field each weight samples from. This is closer to "adaptive receptive field" than "adaptive stride/skip." **The stride between consecutive kernel applications remains fixed.**

---

### 2.3 DiffStride: Learnable Stride (The Closest to Q2)

#### Key Paper

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| **Riad et al., "Learning Strides in Convolutional Neural Networks" (DiffStride)** | 2022 | ICLR | **First differentiable stride layer.** Casts spatial downsampling as frequency-domain cropping. Stride becomes a continuous, learnable parameter optimized by backpropagation. Drop-in replacement for strided convolutions. |
| **"Hybrid of DiffStride and Spectral Pooling"** | 2024 | arXiv | Combines DiffStride with spectral pooling. |

#### Relevance to Q2
DiffStride makes stride LEARNABLE but NOT INPUT-DEPENDENT. The stride is optimized during training and fixed at inference. Every input gets the same stride. **Q2 asks for stride that varies PER INPUT (and per position within an input), which DiffStride does not provide.**

---

### 2.4 Spatial Transformer Networks (Input Preprocessing)

#### Key Paper

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| **Jaderberg et al., "Spatial Transformer Networks"** | 2015 | NeurIPS | Learnable module that applies input-dependent spatial transformations (affine, projective, thin-plate spline). The transformation parameters are PREDICTED FROM THE INPUT by a localization network. ~7,000+ citations. |

#### Relevance to Q2
This IS input preprocessing: a network examines the input and produces a transformation that warps the input before downstream processing. However, the transformation is GLOBAL (one transformation per image) and continuous (not discrete skip/stride). **It does not produce a per-position skip table.** Extensions to local transformations exist but are architecturally complex.

---

### 2.5 Content-Aware Feature Reassembly

#### Key Papers

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| **Wang et al., "CARAFE: Content-Aware ReAssembly of FEatures"** | 2019 | ICCV | Upsampling operator that generates adaptive kernels from content. Instance-specific, content-aware. |
| **Wang et al., "CARAFE++: Unified Content-Aware ReAssembly"** | 2021 | TPAMI | Extended to both upsampling AND downsampling. Content-aware downsampling with adaptive kernels. |
| **Talebi & Milanfar (Google), "Learning to Resize Images for Computer Vision Tasks"** | 2021 | ICCV | Learned CNN-based resizer jointly trained with downstream model. Produces "machine-friendly" resized images. Input-adaptive preprocessing. |

#### Relevance to Q2
CARAFE++ does content-aware downsampling (which IS a form of content-aware stride), but the downsampling factor is fixed -- only the kernel weights are adaptive. "Learning to Resize" is closer: it preprocesses the input in a content-adaptive way, but the resize factor is fixed, not per-position.

---

### 2.6 Dynamic Token Pruning / Skipping in Vision Transformers

This is the most active area addressing "content-aware skipping of irrelevant regions."

#### Key Papers

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| **Rao et al., "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification"** | 2021 | NeurIPS | Predicts which tokens to prune at each layer. Binary decisions per token. |
| **Bolya et al., "Token Merging (ToMe)"** | 2023 | ICLR | Merges similar tokens via bipartite matching. No training required. ~500+ citations. |
| **Tang et al., "Dynamic Token Pruning in Plain Vision Transformers for Semantic Segmentation"** | 2023 | ICCV | Per-position pruning decisions for dense prediction. |
| **Cao et al., "MADTP: Multimodal Alignment-Guided Dynamic Token Pruning"** | 2024 | CVPR | Dynamic pruning for vision-language transformers. |
| **IdleViT: "No Token Left Behind"** | 2023 | AI Conference | Tokens not pruned but "idled" -- skip computation but maintain position. |
| **SPRINT (Sparse Pretraining with Reduced Input Tokens)** | 2025 | Photogrammetric Record | Two-stage token reduction for Mamba-Transformer: State-Preserving Token Condensation + attention-guided pruning. |

#### Relevance to Q2
Token pruning IS content-aware skipping: the network examines the input and decides which spatial positions to process. This is the closest existing analog to Q2's "Z-table that modifies stride." However:
- It operates on TOKENS (patches), not at the raw input level
- It is typically applied WITHIN the network, not as INPUT preprocessing
- The "table" is computed on-the-fly, not as a preprocessing step

**The gap**: No paper frames this as INPUT preprocessing that produces a skip table BEFORE convolution, analogous to how Z-array preprocessing happens before the search phase in string matching.

---

### 2.7 Early Exit and Adaptive Computation

#### Key Papers

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| **Graves, "Adaptive Computation Time for Recurrent Neural Networks"** | 2016 | arXiv | Foundational paper: RNN learns how many computation steps to take per input. ~1,500+ citations. |
| **Huang et al., "Multi-Scale Dense Networks (MSDNet)"** | 2018 | ICLR | Multi-scale early exit for CNNs. Easy images exit early. |
| **Teerapittayanon et al., "BranchyNet"** | 2016 | ICPR | Side branches for early exit in CNNs. |
| **He et al., "Two-Stage Early Exiting From Globality Towards Reliability"** | 2025 | CAAI Trans. Intelligence Technology | SOTA early exit: 2.17x speedup on GLUE with minimal performance loss. |
| **"LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding"** | 2024 | ACL | Early exit for LLMs. |

#### Relevance to Q2
Early exit is DEPTH-adaptive (skip later layers), not SPATIAL-adaptive (skip input regions). It addresses "how much to process" but not "where to look." Not directly applicable to Q2's spatial skip idea.

---

### 2.8 Skim Reading / Text Skipping in NLP

#### Key Papers

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| **Yu et al., "Learning to Skim Text"** | 2017 | ACL | LSTM-Jump: RL-trained model that decides how many tokens to skip. First "speed reading" model. |
| **Seo et al., "Neural Speed Reading via Skim-RNN"** | 2018 | ICLR | Skimming (small RNN) vs reading (big RNN) per token. Does not skip tokens entirely but allocates less computation. |
| **Campos et al., "Skip-RNN"** | 2018 | ICLR | Binary skip decisions for each timestep. |
| **"Block-Skim"** | 2021 | EMNLP | Identifies which context blocks can be safely discarded in Transformer QA. 3x speedup on BERT-base. |
| **"BADGE: Speeding Up BERT Inference"** | 2023 | ACL | Block-wise bypasses + divergence-based early exiting for BERT. |

#### Relevance to Q2
Skim-RNN and LSTM-Jump are the closest NLP analogs to Q2: they preprocess/examine the input and decide where to skip. But they are SEQUENTIAL (RNN-based), not convolutional. **The idea of preprocessing text to build a skip-table that then guides a CNN's stride has not been explored.**

---

### 2.9 Sparse / Submanifold Convolutions

#### Key Papers

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| **Graham & van der Maaten, "Submanifold Sparse Convolutional Networks"** | 2017/2018 | CVPR | Convolution only at occupied sites (originally for 3D point clouds/voxels). Active sites remain unchanged across layers. |
| **Choy et al., "MinkowskiEngine: 4D Spatio-Temporal ConvNets"** | 2019 | CVPR | Generalized sparse convolution on arbitrary coordinate sets. |

#### Relevance to Q2
Submanifold sparse convolution IS content-aware: convolution only happens where the input is non-empty. This is analogous to skipping regions of all-zeros. But the "sparsity pattern" comes from the data representation (3D occupancy), not from a learned preprocessing step. **For dense data (images, text, DNA), there is no natural sparsity to exploit without a learned sparsification step.**

---

### 2.10 Adaptive Convolution in Genomics

#### Key Papers

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| **Variable Convolutional Layer (vConv/vCNN)** | 2019-2023 | bioRxiv | Dynamically learns kernel LENGTH from data. Adapts to variable-length motifs. Outperforms fixed-kernel CNNs on DNA-protein binding tasks. |
| **"A Mechanistically Interpretable Neural Network for Regulatory Genomics"** | 2024 | OpenReview | Motifs and syntax directly encoded in learned weights. |
| **"Revisiting Convolution Architecture in the Realm of DNA Foundation Models"** | 2025 | arXiv | Re-examines convolution vs. attention for genomic sequences at scale. |

#### Relevance to Q2
vConv adapts kernel SIZE but not stride. The stride remains fixed. For DNA, the Q2 idea would be: scan the sequence first, identify low-complexity or repetitive regions, then skip over them when running the main CNN. **This specific approach has not been proposed.**

---

### 2.11 Survey: Dynamic Neural Networks

#### Key Surveys

| Paper | Year | Venue | Key Contribution |
|-------|------|-------|-----------------|
| **Han et al., "Dynamic Neural Networks: A Survey"** | 2021 | IEEE TPAMI | Comprehensive taxonomy: sample-wise, spatial-wise, temporal-wise dynamic models. ~800+ citations. |
| **"Conditional Computation in Neural Networks: Principles and Research Trends"** | 2024 | arXiv | Updated survey covering conditional computation methods. |
| **"Early-Exit Deep Neural Network: A Comprehensive Survey"** | 2024 | ACM Computing Surveys | Focused survey on early exit methods across domains. |

---

### 2.12 Summary for Q2

| Approach | Input-Dependent? | Spatial? | Stride/Skip? | Per-Position? | Status |
|----------|-------------------|----------|--------------|---------------|--------|
| Deformable Conv (v1-v4) | Yes | Yes | No (sampling offsets only) | Yes | Mature, SOTA |
| DiffStride | No (learned, not input-dependent) | Yes | Yes (learnable) | No (global) | Published ICLR 2022 |
| Spatial Transformer | Yes | Yes | No (global warp) | No | Mature (2015) |
| CARAFE++ | Yes | Yes | Fixed factor | Yes (adaptive kernel) | Published 2021 |
| Learning to Resize | Yes | Yes | Fixed factor | Partially | Published ICCV 2021 |
| Token Pruning (DynamicViT, ToMe) | Yes | Yes | Binary skip | Yes | Very active 2021-2025 |
| Skim-RNN / LSTM-Jump | Yes | Sequential | Yes (skip tokens) | Yes | NLP only, 2017-2018 |
| Sparse Conv | Data-dependent | Yes | Implicit skip | Yes | 3D point clouds only |
| vConv (genomics) | No | 1D | No (kernel size only) | No | Genomics only |

**SOTA Assessment**: The Q2 idea -- "preprocess the input to build a per-position table that dynamically modifies the CNN's stride" -- is a **novel research direction** that combines elements from:
1. Spatial Transformer Networks (input-dependent preprocessing)
2. DiffStride (learnable stride)
3. Token pruning (content-aware skipping)
4. Skim-RNN (skip decisions for sequential data)

**No existing paper unifies these into Q2's specific formulation.** The closest is dynamic token pruning in vision transformers, but this operates at the token level within the network, not as a preprocessing step that generates a skip table for convolution.

---

<a id="surveys"></a>
## 3. Cross-Cutting Surveys

| Survey | Year | Venue | Relevance |
|--------|------|-------|-----------|
| Han et al., "Dynamic Neural Networks: A Survey" | 2021 | IEEE TPAMI | Taxonomy of all dynamic NN approaches. Best overview for Q2 context. |
| Merrill, "Formal Language Theory Meets Modern NLP" | 2021 | arXiv | Connects NNs to Chomsky hierarchy. Key for Q1 theoretical bounds. |
| Ackerman & Cybenko, "Survey of Neural Networks and Formal Languages" | 2020 | arXiv | Comprehensive formal results for Q1. |
| "A Survey on Deep Learning in DNA/RNA Motif Mining" | 2021 | Briefings in Bioinformatics | CNN filters as motif detectors. Key for Q1 genomics angle. |
| "Conditional Computation in Neural Networks" | 2024 | arXiv | Updated review of conditional computation for Q2. |
| "Early-Exit Deep Neural Network: A Comprehensive Survey" | 2024 | ACM Computing Surveys | All early exit methods. Peripheral to Q2. |
| Velickovic, "Neural Algorithmic Reasoning" | 2021 | Patterns | Framework for learning algorithms with NNs. Key for Q1. |

---

<a id="synthesis"></a>
## 4. Synthesis and Open Problems

### 4.1 For Q1: CNN Weights as Failure Functions

**Current SOTA Understanding:**
- CNN filters learn template patterns (well-established since Zeiler & Fergus 2014, confirmed at scale by CNN Filter DB 2022)
- In genomics, CNN filters = PWMs (established by DeepBind 2015, formalized by Koo & Eddy 2019, made transparent by ExplaiNN 2023)
- CNNs are provably limited to strictly local languages (Merrill 2019), which means they CANNOT implement the KMP failure function in a single layer
- GNNs CAN learn to execute KMP (CLRS benchmark 2022), but this requires message-passing (state), not convolution

**Open Problems:**
1. **Can deep residual CNNs (with skip connections) implicitly learn failure-function-like skip behavior?** Skip connections add limited state; does this suffice?
2. **Do multi-layer CNN activation patterns (not weights) encode information equivalent to failure tables?** The weights are the pattern; the activations at mismatch positions might encode shift information.
3. **Is there a formal mapping between the convolutional computation graph and the KMP state machine?** Could one layer implement the comparison and another implement the shift logic?
4. **For 1D tasks (text, DNA), does a CNN with attention (as in Enformer) gain sufficient computational power to implement failure-function logic?**

### 4.2 For Q2: Input-Dependent Stride via Preprocessing

**Current SOTA Understanding:**
- Deformable convolutions (v1-v4) provide input-dependent sampling but fixed stride
- DiffStride (ICLR 2022) provides learnable but input-independent stride
- Token pruning in ViTs provides content-aware spatial skipping but not as CNN stride modification
- Skim-RNN provides content-aware token skipping but only for sequential (RNN) models

**Open Problems (all novel):**
1. **Design an input preprocessing module that outputs a per-position stride map for downstream 1D convolution.** Closest existing: Spatial Transformer + DiffStride fusion.
2. **For DNA sequences**: preprocess to identify low-complexity regions (poly-A, tandem repeats), generate a skip table, apply variable-stride convolution. **No existing work.**
3. **For images**: preprocess to identify texturally uniform regions (sky, walls), generate a spatial skip map, apply variable-stride 2D convolution. **Closest: dynamic token pruning, but applied pre-convolution rather than mid-network.**
4. **For text**: preprocess to identify stop words, filler phrases, build a skip table for 1D CNN text classification. **Closest: Skim-RNN, but for CNNs rather than RNNs.**
5. **Theoretical question**: Can input-dependent stride provably reduce the computation complexity of CNN pattern matching from O(n*m) to O(n + m), analogous to how KMP improves over naive string matching?
6. **Differentiability challenge**: How to make a discrete per-position stride differentiable for end-to-end training? DiffStride's frequency-domain approach works for global stride but extending to per-position is non-trivial.

### 4.3 Key Researchers by Topic

| Topic | Key Researchers | Affiliation |
|-------|----------------|-------------|
| Neural Algorithmic Reasoning | Petar Velickovic, Andrea Banino, Charles Blundell | Google DeepMind |
| Formal Languages + NNs | William Merrill, Gail Weiss | Allen AI / NYU, Technion |
| CNN Filter Analysis | Paul Gavrikov, Janis Keuper | Fraunhofer IOSB / Offenburg |
| DiffStride | Rachid Riad, Olivier Teboul, David Grangier, Neil Zeghidour | Google Research |
| Deformable Conv (DCNv1-v4) | Jifeng Dai, Xizhou Zhu, Yuwen Xiong | MSRA / SenseTime / OpenGVLab |
| Spatial Transformers | Max Jaderberg, Karen Simonyan, Andrew Zisserman | DeepMind / Oxford |
| CNN Genomics (DeepBind, ExplaiNN) | Babak Alipanahi, Brendan Frey, Greg Novakovsky | U Toronto, Wasserman Lab (UBC) |
| Token Pruning | Daniel Bolya (ToMe), Yongming Rao (DynamicViT) | Meta, Tsinghua |
| Skim-RNN | Minjoon Seo, Adams Wei Yu | UW / Google Brain |
| Dynamic NNs Survey | Yizeng Han, Gao Huang, Zheng Zhu | Tsinghua |
| CARAFE | Jiaqi Wang, Kai Chen | Chinese University of HK |
| Learning to Resize | Hossein Talebi, Peyman Milanfar | Google Research |

---

<a id="q2-refined"></a>
## Q2 (Refined): Preprocessare il Filtro Basandosi sull'Input per Stride Non Costante

### Riformulazione Precisa

La domanda non è "imparare lo stride" (DiffStride) né "cambiare i punti di campionamento" (Deformable Conv). La domanda è:

> Dato un filtro CNN e un input, si può **analizzare cheaply la relazione filtro-input** per decidere **dove vale la pena calcolare** la convoluzione completa, saltando le posizioni dove l'output sarà irrilevante?

Questo è esattamente il pattern KMP:
- KMP preprocessa il **pattern** (failure function)
- Durante la scansione del **testo**, usa la failure function per **saltare posizioni**
- Analogamente: un pre-scan cheap del filtro sull'input produce una **skip schedule** che guida dove calcolare la convoluzione completa

### 5.1 TIER 1: Match Diretto — Pre-scan Cheap → Skip Dove Non Serve

Questi paper implementano **esattamente** il meccanismo descritto: un'operazione economica che analizza filtro+input, produce una mappa di skip, e la convoluzione completa viene calcolata solo dove serve.

| Paper | Year | Venue | Mechanism | Results |
|-------|------|-------|-----------|---------|
| **SeerNet: Predicting CNN Feature-Map Sparsity Through Low-Bit Quantization** (Cao, Ma, Xiao, Zhang, Liu, Zhang, Nie, Yang) | 2019 | CVPR | Convoluzione **quantizzata a 2-4 bit** come pre-scan → maschera binaria di sparsità → conv completa solo dove l'output sarà non-zero dopo ReLU. Il pre-scan quantizzato è la "failure function"; la maschera è la "skip schedule." | Skip ~60-80% delle posizioni |
| **Channel Gating Neural Networks** (Hua, Zhou, De Sa, Zhang, Suh) | 2019 | NeurIPS | "Base path" cheap (pochi canali) → gating **per-posizione spaziale** → "conditional path" solo dove il gate è attivo. Il gate decide per ogni posizione se vale la pena calcolare i canali rimanenti. | 2.7-8.0x FLOP reduction, negligible accuracy loss (CIFAR-10) |
| **Dynamic Convolutions: Exploiting Spatial Sparsity for Faster Inference** (Verelst, Tuytelaars) | 2020 | CVPR | **Gating branch** (1x1 conv + sigmoid) → maschera spaziale binaria: 1="calcola conv qui", 0="salta". Training end-to-end con Gumbel-Softmax. Implementazione CUDA gather-scatter per speedup reale. | Wall-clock speedup reale su MobileNetV2, ShuffleNetV2 |
| **SBNet: Sparse Blocks Network for Fast Inference** (Ren, Pokrovsky, Yang, Urtasun — Uber ATG) | 2018 | CVPR | Pre-scan a **bassa risoluzione** (o da dominio) → block mask → conv solo su blocchi attivi. La versione "cheap network mask" è direttamente il paradigma coarse-to-fine. | 2-3x wall-clock speedup, no accuracy loss |
| **Dynamic Dual Gating Neural Networks** (Li, Li, He, Cheng) | 2021 | ICCV (Oral) | Estende Channel Gating con **spatial gating esplicito** + channel gating. Due moduli lightweight predicono (a) quali posizioni spaziali e (b) quali canali servono. Solo l'intersezione viene calcolata. | Migliora Channel Gating su ImageNet |
| **Batch-Shaping for Learning Conditional Channel Gated Networks** (Bejnordi, Blankevoort et al.) | 2020 | ICLR | Feature map individuali attivate/disattivate **condizionatamente all'input**. Un "batch-shaping regularizer" forza i gate a essere veramente data-dependent. | — |
| **Focused Convolutions** | 2023 | arXiv | Soglia sulle attivazioni → identifica regioni da ignorare → sostituisce conv standard con "focused conv" che salta quelle regioni. **No retraining** necessario su CNN pre-addestrate. | 25% latenza, 22% energia |

#### Paper di riferimento chiave:
- [SeerNet (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cao_SeerNet_Predicting_Convolutional_Neural_Network_Feature-Map_Sparsity_Through_Low-Bit_Quantization_CVPR_2019_paper.pdf)
- [Channel Gating (NeurIPS 2019)](https://zhouyuan1119.github.io/papers/cg-neurips2019.pdf) | [arXiv:1805.12549](https://arxiv.org/abs/1805.12549)
- [Dynamic Convolutions (CVPR 2020)](https://arxiv.org/abs/1912.03203) | [GitHub](https://github.com/thomasverelst/dynconv)
- [SBNet (CVPR 2018)](https://arxiv.org/abs/1801.02108) | [GitHub](https://github.com/uber-research/sbnet)
- [Dynamic Dual Gating (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Dynamic_Dual_Gating_Neural_Networks_ICCV_2021_paper.pdf) | [GitHub](https://github.com/lfr-0531/DGNet)
- [Batch-Shaping (ICLR 2020)](https://arxiv.org/abs/1907.06627)
- [Focused Convolutions (2023)](https://arxiv.org/html/2310.07782)

---

### 5.2 TIER 2: Hashing — Trattare la Convoluzione come Ricerca (MIPS)

Ogni posizione della convoluzione è un **dot product** tra filtro e patch locale. Questi paper usano hashing per trovare le posizioni ad alta attivazione senza calcolare tutti i dot product.

| Paper | Year | Venue | Mechanism | Relevance |
|-------|------|-------|-----------|-----------|
| **SLIDE: In Defense of Smart Algorithms over Hardware Acceleration** (Chen, Medini, Farber, Tai, Shrivastava) | 2020 | MLSys | **Locality-Sensitive Hashing (LSH)** per identificare quali neuroni avranno le attivazioni più alte senza calcolare tutti i dot product. CPU competitive con GPU. | In principio applicabile alla conv spaziale: hasha il filtro + ogni patch locale, calcola il dot product solo dove i bucket collidono. Attualmente solo FC layers. |
| **MONGOOSE: A Learnable LSH Framework** (Chen, Liu, Peng, Xu, Li, Dao, Song, Shrivastava, Re) | 2021 | ICLR | Hash function **apprese** (non random) con schedule adattivo. | 5-20x più veloce di SLIDE, fino a 4x più memory efficient. |
| **SMYRF: Efficient Attention using Asymmetric Clustering** (Daras, Kitaev, Odena, Dimakis) | 2020 | NeurIPS | LSH asimmetrico per clusterizzare query e key, calcolando attention solo dentro i cluster. Da O(N²) a O(N log N). | Self-attention è matematicamente una forma di convoluzione (all-pairs inner product); SMYRF mostra che l'hashing può identificare quali coppie contano. |

#### Paper di riferimento:
- [SLIDE (MLSys 2020)](https://arxiv.org/abs/1903.03129)
- [MONGOOSE (ICLR 2021)](https://openreview.net/forum?id=wWK7yXkULyh)
- [SMYRF (NeurIPS 2020)](https://arxiv.org/abs/2010.05315) | [GitHub](https://github.com/giannisdaras/smyrf)

---

### 5.3 TIER 3: Skip Spaziale Adattivo e Interpolazione

| Paper | Year | Venue | Mechanism | Results |
|-------|------|-------|-----------|---------|
| **PerforatedCNNs** (Figurnov, Ibraimova, Vetrov, Kohli) | 2016 | NIPS | Salta posizioni spaziali, interpola i buchi con nearest-neighbor. Il framework si estende naturalmente a maschere input-dependent. | 2-4x speedup su AlexNet e VGG-16 |
| **Spatially Adaptive Computation Time** (Figurnov, Collins, Zhu, Zhang, Huang, Vetrov, Salakhutdinov) | 2017 | CVPR | Ogni posizione spaziale decide **indipendentemente** quanti blocchi residuali eseguire (early halting). L'halting score è calcolato dalle feature — skip input-dependent. | Adaptive depth per-position |
| **Spatially Adaptive Inference with Stochastic Feature Sampling** (Xie et al., Microsoft Research) | 2020 | ECCV (Oral) | Feature map come campo di probabilità → campionamento stocastico sparse delle posizioni → conv solo lì → interpolazione del resto. Distribuzione di campionamento **input-dependent**. | — |
| **Skip-Convolutions for Efficient Video Processing** (Habibian, Abati, Cohen, Bejnordi — Qualcomm) | 2021 | CVPR | Per video: conv(frame) = conv(frame_precedente) + residuo. Il residuo è spazialmente sparse (solo dove il video è cambiato) → skip input-dependent. | — |

#### Paper di riferimento:
- [PerforatedCNNs (NIPS 2016)](https://arxiv.org/abs/1504.08362)
- [Spatially Adaptive Computation Time (CVPR 2017)](https://arxiv.org/abs/1612.02297) | [GitHub](https://github.com/mfigurnov/sact)
- [Spatially Adaptive Inference (ECCV 2020)](https://arxiv.org/abs/2003.08866) | [GitHub](https://github.com/zdaxie/SpatiallyAdaptiveInference-Detection)
- [Skip-Convolutions (CVPR 2021)](https://arxiv.org/abs/2104.11487) | [GitHub](https://github.com/Qualcomm-AI-research/Skip-Conv)

---

### 5.4 TIER 4: Hardware-Level Activation Sparsity

| Paper | Year | Venue | Mechanism |
|-------|------|-------|-----------|
| **SCNN: Accelerator for Compressed-sparse CNNs** (Parashar, Rhu et al.) | 2017 | ISCA | Acceleratore hardware che sfrutta zeri nelle attivazioni (da ReLU) e nei pesi (da pruning) per saltare operazioni MAC. La sparsità è inerentemente input-dependent. |
| **Spatial Pruned Sparse Convolution** | 2023 | OpenReview | Pota posizioni spaziali in conv sparse 3D predicendo quali voxel contribuiscono al risultato finale. |

#### Paper di riferimento:
- [SCNN (ISCA 2017)](https://www.cs.utexas.edu/~skeckler/pubs/ISCA_2017_SCNN.pdf)
- [Spatial Pruned Sparse Conv](https://openreview.net/pdf?id=QqWqFLbllZh)

---

### 5.5 Mapping Formale: KMP ↔ Pre-scan CNN

| KMP String Matching | Equivalente CNN (dai paper sopra) |
|---|---|
| Pattern P (fisso) | Filtro/kernel CNN (fisso a inference) |
| Testo T (input) | Feature map di input |
| **Failure function** (preprocessata da P) | **Gating function / pre-scan quantizzata / hash table** (da filtro + input) |
| Shift variabile dopo mismatch | Skip delle posizioni dove il gate predice attivazione zero/bassa |
| Evita confronti ridondanti | Evita di calcolare conv dove l'output sarà nullo |
| Complessità O(n+m) vs O(nm) | Speedup 2-8x vs convoluzione densa |

**Differenza chiave (e vantaggio):** In KMP la failure function dipende **solo dal pattern**. Negli approcci CNN, la skip schedule dipende da **filtro + input insieme** — il che è **più potente** di KMP perché lo skip pattern si adatta a ogni input specifico anziché essere fissato per un dato pattern.

**Il gap nella letteratura:** Nessun paper formalizza questa connessione. L'analogia concettuale tra la failure function come "pre-scan del pattern che guida lo skip" e il gating/pre-scan come "analisi cheap del filtro sull'input che guida lo skip" non è mai stata esplicitata. Questo potrebbe essere un **contributo concettuale/teorico originale**.

---

### 5.6 Ricercatori Chiave per Q2 (Refined)

| Topic | Researchers | Affiliation |
|-------|-------------|-------------|
| Channel/Spatial Gating | Weizhe Hua, Yuan Zhou, Christopher De Sa | Cornell |
| Dynamic Convolutions (spatial sparsity) | Thomas Verelst, Tinne Tuytelaars | KU Leuven |
| SBNet (sparse blocks) | Mengye Ren, Raquel Urtasun | Uber ATG / U Toronto |
| SeerNet (quantized prediction) | Shijie Cao et al. | — |
| SLIDE / Hashing for NN | **Beidi Chen**, Anshumali Shrivastava | Rice / CMU / Meta |
| MONGOOSE | Beidi Chen, Christopher Re | Stanford / CMU |
| PerforatedCNNs / SACT | Mikhail Figurnov, Dmitry Vetrov | Google DeepMind / HSE Moscow |
| Skip-Conv (video) | Amirhossein Habibian | Qualcomm AI Research |
| Dynamic Dual Gating | Fanrong Li | — |
| Focused Convolutions | — | — |

---

### 4.4 Bottom Line

**Q1 is largely a novel theoretical question with a likely NEGATIVE answer for vanilla CNNs** (due to Merrill 2019's subregularity result) but an OPEN answer for deep residual/attention-augmented architectures. The closest positive results come from the GNN algorithmic reasoning literature (CLRS benchmark), where KMP can be learned but requires message-passing, not convolution.

**Q2 (original formulation — input-dependent learned stride) sits at the intersection of 5+ active research areas** (deformable conv, learnable stride, token pruning, spatial transformers, sparse conv) but has not been unified into the specific formulation proposed.

**Q2 (refined formulation — preprocessing the filter-input relationship for non-constant stride) HAS existing implementations** (SeerNet, Channel Gating, Dynamic Convolutions, SBNet) that achieve 2-8x speedup. However, **no paper formalizes these as the CNN analog of KMP's failure-function-guided skipping**. The conceptual bridge between algorithmic pattern matching theory and CNN spatial gating remains undrawn — this is a publishable contribution.

The most promising direction for a novel contribution would be:
1. **Formalize** the mapping KMP failure function ↔ CNN spatial gating (theoretical paper)
2. **Design** a unified "Algorithmic Gating" module explicitly inspired by KMP preprocessing, applicable across domains (architecture paper)
3. **Apply** to 1D domains (DNA, text) where the KMP analogy is most direct and existing spatial gating work (mostly 2D vision) is underexplored

---

## Sources

### Q1 Sources
- [CNN Filter DB (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Gavrikov_CNN_Filter_DB_An_Empirical_Investigation_of_Trained_Convolutional_Filters_CVPR_2022_paper.html)
- [Zeiler & Fergus, Visualizing and Understanding CNNs (ECCV 2014)](https://arxiv.org/abs/1311.2901)
- [Merrill, Sequential Neural Networks as Automata (2019)](https://aclanthology.org/W19-3901/)
- [Merrill, Formal Language Theory Meets Modern NLP (2021)](https://arxiv.org/abs/2102.10094)
- [CLRS Algorithmic Reasoning Benchmark (ICML 2022)](https://arxiv.org/abs/2205.15659)
- [Neural Algorithmic Reasoning (2021)](https://arxiv.org/abs/2105.02761)
- [DeepBind (Nature Biotechnology 2015)](https://www.nature.com/articles/nbt.3300)
- [Koo & Eddy, Representation Learning of Genomic Motifs (2019)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007560)
- [ExplaiNN (Genome Biology 2023)](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-02985-y)
- [Klein & Barron (WIREs Cognitive Science 2024)](https://doi.org/10.1002/wcs.1680)
- [Neural Algorithmic Reasoning Without Intermediate Supervision (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/a2370db7c99791ad5d9f3ef48ad6d464-Paper-Conference.pdf)
- [SALSA-CLRS (2024)](https://openreview.net/pdf/43f9353c8974c311d724aa6ef7cc8a5e1f04aea5.pdf)
- [CLRS-Text Benchmark (2024)](https://arxiv.org/pdf/2406.04229)
- [Discrete Neural Algorithmic Reasoning (2024)](https://arxiv.org/html/2402.11628)
- [Survey on Deep Learning in DNA/RNA Motif Mining (2021)](https://academic.oup.com/bib/article/22/4/bbaa229/5916939)
- [vConv Variable Convolutional Layer](https://www.biorxiv.org/content/10.1101/508242v5)

### Q2 Sources
- [DiffStride (ICLR 2022)](https://arxiv.org/abs/2202.01653)
- [DiffStride GitHub](https://github.com/google-research/diffstride)
- [Deformable ConvNets v4 (CVPR 2024)](https://arxiv.org/abs/2401.06197)
- [DCNv4 GitHub](https://github.com/OpenGVLab/DCNv4)
- [Spatial Transformer Networks (NeurIPS 2015)](https://arxiv.org/abs/1506.02025)
- [CARAFE (ICCV 2019)](https://arxiv.org/abs/1905.02188)
- [CARAFE++ (2021)](https://arxiv.org/abs/2012.04733)
- [Learning to Resize Images (ICCV 2021)](https://arxiv.org/abs/2103.09950)
- [Skim-RNN (ICLR 2018)](https://arxiv.org/abs/1711.02085)
- [DynamicViT (NeurIPS 2021)](https://arxiv.org/abs/2106.02034)
- [Token Merging (ToMe, ICLR 2023)](https://arxiv.org/abs/2210.09461)
- [Dynamic Token Pruning for Semantic Segmentation (ICCV 2023)](https://arxiv.org/abs/2308.01045)
- [MADTP (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Cao_MADTP_Multimodal_Alignment-Guided_Dynamic_Token_Pruning_for_Accelerating_Vision-Language_Transformer_CVPR_2024_paper.pdf)
- [Submanifold Sparse ConvNets (CVPR 2018)](https://arxiv.org/abs/1706.01307)
- [MinkowskiEngine (CVPR 2019)](https://nvidia.github.io/MinkowskiEngine/)
- [Dynamic Neural Networks: A Survey (IEEE TPAMI 2021)](https://arxiv.org/abs/2102.04906)
- [Conditional Computation Survey (2024)](https://arxiv.org/html/2403.07965v1)
- [Early-Exit DNN Survey (ACM Computing Surveys 2024)](https://dl.acm.org/doi/10.1145/3698767)
- [Adaptive Computation Time (2016)](https://arxiv.org/abs/1603.08983)
- [BranchyNet (ICPR 2016)](https://arxiv.org/abs/1709.01686)
- [Two-Stage Early Exiting (2025)](https://doi.org/10.1049/cit2.70010)
- [LayerSkip (ACL 2024)](https://arxiv.org/abs/2404.16710)
- [Revisiting Convolution for DNA Foundation Models (2025)](https://arxiv.org/html/2502.18538v1)
- [Mechanistically Interpretable NN for Regulatory Genomics (2024)](https://arxiv.org/html/2410.06211v1)

### Q2 Refined Sources (Pre-scan / Gating / Hashing)
- [SeerNet (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cao_SeerNet_Predicting_Convolutional_Neural_Network_Feature-Map_Sparsity_Through_Low-Bit_Quantization_CVPR_2019_paper.pdf)
- [Channel Gating Neural Networks (NeurIPS 2019)](https://zhouyuan1119.github.io/papers/cg-neurips2019.pdf) | [arXiv:1805.12549](https://arxiv.org/abs/1805.12549)
- [Dynamic Convolutions — Spatial Sparsity (CVPR 2020)](https://arxiv.org/abs/1912.03203) | [GitHub](https://github.com/thomasverelst/dynconv)
- [SBNet (CVPR 2018)](https://arxiv.org/abs/1801.02108) | [GitHub](https://github.com/uber-research/sbnet)
- [Dynamic Dual Gating (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Dynamic_Dual_Gating_Neural_Networks_ICCV_2021_paper.pdf) | [GitHub](https://github.com/lfr-0531/DGNet)
- [Batch-Shaping (ICLR 2020)](https://arxiv.org/abs/1907.06627)
- [Focused Convolutions (2023)](https://arxiv.org/html/2310.07782)
- [SLIDE (MLSys 2020)](https://arxiv.org/abs/1903.03129)
- [MONGOOSE (ICLR 2021)](https://openreview.net/forum?id=wWK7yXkULyh)
- [SMYRF (NeurIPS 2020)](https://arxiv.org/abs/2010.05315) | [GitHub](https://github.com/giannisdaras/smyrf)
- [PerforatedCNNs (NIPS 2016)](https://arxiv.org/abs/1504.08362)
- [Spatially Adaptive Computation Time (CVPR 2017)](https://arxiv.org/abs/1612.02297) | [GitHub](https://github.com/mfigurnov/sact)
- [Spatially Adaptive Inference (ECCV 2020)](https://arxiv.org/abs/2003.08866) | [GitHub](https://github.com/zdaxie/SpatiallyAdaptiveInference-Detection)
- [Skip-Convolutions (CVPR 2021)](https://arxiv.org/abs/2104.11487) | [GitHub](https://github.com/Qualcomm-AI-research/Skip-Conv)
- [SCNN Accelerator (ISCA 2017)](https://www.cs.utexas.edu/~skeckler/pubs/ISCA_2017_SCNN.pdf)
