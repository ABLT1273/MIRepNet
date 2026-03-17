<!-- 518 IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 73, NO. 2, FEBRUARY 2026 -->

<!-- EMBS -->
![](https://web-api.textin.com/ocr_image/external/dc1fedb2a042b759.jpg)

# Intracortical Brain-Machine Interfaces With

# High-Performance Neural Decoding Through

# Efficient Transfer Meta-Learning

Xingjian Chen®, Student Member, IEEE, Zhongzheng Fu, Student Member, IEEE,

Peng Zhang®,Member, IEEE, Xinxing Chen ®, Member, IEEE, and Jian Huang ®, Senior Member, IEEE

**Abstract-Implantable brain-machine interfaces (iBMIs)** have emerged as a **groundbreaking neural technology for** restoring **motor function and enabling direct neural com-**munication **pathways. Despite their therapeutic potential** in **neurological rehabilitation, the critical challenge of neu-ral decoder calibration persists, particularly in the con-text of transfer learning. Traditional calibration approaches** **assume the availability of extensive neural recordings,** **which is often impractical in clinical settings due to pa-tient fatigue annd neural signal variability. Furthermore,** **the inherent constraints of implanted neural processors-including limited computational capacity and power con-sumption requirements-demand streamlined processing** **algorithms. To address these clinical and technical chal-lenges, we developed DMM-WcycleGAN (Dimensionality** **Reduction Model-Agnostic Meta-Learning based Wasser-stein Cycle Generative Adversarial Networks), a novel** **neural decoding framework that integrates meta-learning** **principles with optimal transfer learning strategies. This in-novative approach enables efficient decoder calibration us-ing minimal neural data while implementing dimensionality** **reduction techniques to optimize computational efficiency** **in implanted devices. In vivo experiments with non-human** **primates demonstrated DMM-WcycleGAN's superior per-formance in mitigating neural signal distribution shifts** **between historical and current recordings, achieving a 3%**

Received 23 January 2025; revised 31 May 2025; accepted 27 June 2025. Date of publication 8 July 2025;date of current version 15 January 2026. This work was supported in part by the Major Program (JD) of Hubei Province under Grant 2023BAA005. (Corresponding authors: Xinxing Chen;Jian Huang.)

This work involved human subjects or animals in its research. Ap-proval of all ethical and experimental procedures and protocols was granted by the Institutional Animal Care and Use Committee under Ap-plication No. 2011096, and performed in line with the Wuhan University Center for Animal Experiment.

Xingjian Chen and Zhongzheng Fu are with the Key Laboratory of Image Processing and Intelligent Control and the Hubei Key Laboratory of Brain-inspired Intelligent Systems, School of Artificial Intelligence and Automation,Huazhong University of Science and Technology,China.

Peng Zhang is with the Department of Biomedical Engineering, Col-lege of Life Science and Technology, Huazhong University of Science and Technology,China.

Xinxing Chen and Jian Huang are with the Key Laboratory of Im-age PProcessing and Intelligent Control and the Hubei Key Laboratory of Brain-inspired Intelligent Systems, School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, Wuhan 430074, China (e-mail: cxx@hust.edu.cn; huangjan@mail. hust.edu.cn).

This article has supplementary downloadable material available at https://doi.org/10.1109/TBME.2025.3586870,provided by the authors.

Digital Object Identifier 10.1109/TBME.2025.3586870

**enhancement in neural decoding accuracy using only ten** **calibration trials while reducing the calibration duration by** **over 70%, thus significantly improving the clinical viability** **of iBMI systems.**

**Index Terms-CycleGAN, intracortical borain-machine in-terface, meta-learning, transfer learning.**

## I. INTRODUCTION

**IBMI** represents a significant breakthrough in biomedical engineering, enabling high-precision neural communication between the human brain and external devices [1], [2].Through cortical microelectrode array implantation, these neural inter-faces can capture high-fidelity neuronal firing activities, offering superior signal resolution and stability compared to non-invasive alternatives. This technology has demonstrated remarkable clin-ical value in functional restoration for patients with severe motor disabilities,enabling direct neural control of computing de-vices [3], neural prostheses [4],[5],and other rehabilitative assis-tive devices. While recent advances in neural engineering have emphasized its transformative potential in neurorehabilitation and brain-machine interaction, significant technical challenges remain in clinical translation [6],[7],particularly in neural signal processing and system stability.

Human fine motor control involves complex neural circuit networks, including critical brain regions such as the primary motor cortex (M1), primary somatosensory cortex (S1),and posterior parietal cortex (PPC) [8]. Intracortical electrodes can directly record neural population activities from these regions, providing essential tools for investigating motor control mechanisms and decoding motor intentions. However, the clinical application of iBMI systems faces multiple biomedical engineering challenges. Primarily, there is temporal degradation of decoding performance, mainly attributed to electrode-tissue interface reactions and mechanical micromotion [9], necessi-tating frequent recalibration processes [10]. Clinical practice constraints, such as patient fatigue and limited therapeutic time windows, severely restrict calibration data acquisition. Additionally, the inherent computational and storage limitations of implantable devices further emphasize the urgent need for efficient algorithms to maintain stable decoding performance in practical applications. Machine learning and deep learning approaches offer new perspectives and possibilities for addressing these biomedical engineering challenges.

<!-- 0018-9294 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission. See httpos://www.ieee.org/publications/rights/index.html -->

<!-- for more information. -->

<!-- Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on March 04,2026 at 03:48:01 UTC from IEEE Xplore. Restrictions apply. -->

<!-- CHEN et al.: INTRACORTICAL BRAIN-MACHINE INTERFACES WITH HIGH-PERFORMANCE NEURAL DECODING 519 -->

For iBMI systems, researchers have recently developed various decoding algorithms, including Hilbert-Huang Trans-form (HHT) [11] and Common Spatial Patterns (CSP) [12], [13].However, these algorithms primarily focus on initial de-coder construction, failing to effectively address the neural signal characteristic drift caused by dynamic changes in the electrode-tissue interface [14], [15]. Domain Adaptation Net-works (DAN) achieve feature alignment by minimizing dis-tribution differences in high-dimensional feature spaces [16], while transformer-based approaches like BERT and its improved version Audio ALBERT optimize model performance through innovative parameter decomposition and cross-layer parameter sharing mechanisms [17]. Generative Adversarial Networks (GANs) have shown particular value in iBMI adaptive cali-bration, garnering widespread attention in brain-machine inter-face applications [18]. Through adversarial training between generators and discriminators [19],GANs not only generate high-quality simulated neural data but also achieve precise fea-ture mapping between source and target domains,effectively reducing domain discrepancies. Notably,CycleGAN,which incorporates cycle consistency loss [20],[21],enables stable domain-to-domain transformation without paired data, making it particularly suitable for handling dynamically changing neural signal distributions and effectively mitigating decoding perfor-mance degradation.

To address the challenge of limited calibration data acqui-sition in clinical environments, researchers have proposed var-ious few-shot learning approaches. Although existing transfer learning methods demonstrate excellent performance on large-scale datasets [22], [23],[24],their performance significantly degrades in data-constrained scenarios due to insufficient fine-tuning and training data [25]. Meta-learning, based on the con-cept of "learning to learn”, provides an innovative solution to this challenge. Compared to traditional pre-trainingmethods,meta-learning demonstrates significant advantages in rapid task adap-tation through multi-task experience accumulation [26]. Specif-ically, Model-Agnostic Meta-Learning (MAML) [27] achieves efficient task adaptation through learning optimal parameter initialization points, making it particularly suitable for data-constrained iBMI clinical applications. However, modern mul-tichannel neural electrode arrays present new complexities in addressing computational resource constraints. While these sys-tems enable synchronous acquisition of high-resolution neural signals from multiple cortical regions, enhancing decoding accu-racy [28], they introduce significant computational burdens that become particularly pronounced in resource-constrained im-plantable systems. In real-time decoding applications of iBMI, model response latency directly impacts the clinical viability of the devices [29], with excessive computational complexity emerging as one of the critical bottlenecks limiting the clinical translation of iBMI [30],[31].

To address these critical biomedical engineering challenges in iBMI applications, we proposed a novel DMM-WCycleGAN(Dimensionality Reduction Model-Agnostic Meta-Learning based Wasserstein Cycle Generative Adversarial Networks) framework, which innovatively integrates **MAML** principles with CycleGAN-based transfer learning for neural signal decoding. The framework synergistically combines the rapid adaptation capabilities of meta-learning with the domain transformation abilities of CycleGAN, while efficiently utilizing both historical and online neural recordings to achieve effective model transfer under limited data conditions. To meet the resource constraints of implantable devices, we decompose CycleGAN's monolithic optimization into sequential steps and apply dimensionality reduction techniques. In addition, we redesign the MetaCycleGAN loss to eliminate computational redundancies and streamline neural signal processing. In motor intention decoding experiments with rhesus monkeys, DMM-WCycleGAN reduced calibration time by 70% while maintaining high decoding accuracy with minimal patient data. These advancements directly address the dual challenges of limited clinical data and restricted computational resources, significantly enhancing the clinical viability of iBMI systems.

The main contributions of this study are:

1) A MAML-based neural decoder framework was engi-neered to address recalibration challenges in clinical iBMII systems, enabling rapid decoder adaptation with minimal neural recordings and significantly reducing pa-tient calibration burden.

2) A computationally efficient transfer learning framework with opotimized MetaCycleGAN loss was developed, specifically designed for resource-constrained implantable neural devices while maintaining robust decoding performance.

3) Validation using intracortical recordings from rhesus macaques demonstrated superior calibration performance compared to existing methods, achieving higher accuracy with reduced calibration time and data requirements.

The rest of this paper is organized as follows: Section II presents the experimental methodology, encompassing experi-mental setup, data preprocessing protocols, and evaluation met-rics, along with a detailed description of the algorithm architec-ture. Section III presents the experimental results, while Sec-tion IV discusses the findings. Finally, Section V concludes the paper.

## II. METHOD

All experimental and surgical procedures were conducted in accordance with protocols approved by the Institutional Ani-mal Care and Use Committee at the Wuhan University Center for Animal Experiment (approval date: September 6, 2011, no.2011096).

### A. Overall Design of System

The proposed iBMIs transfer learning architecture, illustrated in Fig. 1. A male rhesus monkey was trained to perform spatial reaching and grasping tasks with its right hand. Data was col-lected through microelectrode array implantation in its brain. The collected data was band-pass filtered and spike signals were generated using the threshold crossing method. A sophisticated dual-stage framework encompassing offline training and online recognition phases was proposed, wherein the offline training

<!-- Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on March 04,2026 at 03:48:01 UTC from IEEE Xplore. Restrictions apply. -->

<!-- 520 -->

<!-- IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 73, NO. 2, FEBRUARY 2026 -->

<!-- Off-line Meta Part Raw Data Sampling Test Dataset InitialModel InitializationParameter Feature Meta Band Pass Extraction Pre-train Dataset Sampling Inner Update Inner Update Inner Update Filter Data 250hz-6000hz Inner Loop Update direction Outer Update Synchronization Threshold Crossing Common Model 0 m m Source a mI m Domain Standardization Train Dataset - Fake Target b Align Data Dataset Build Universal Domain Discriminator $D_{u}$ On-line Transfer Learning Part -->
![](https://web-api.textin.com/ocr_image/external/0ca16ae2faa83afd.jpg)

Fig. 1. The framework of the proposed iBMI and the proposed DMM-WcycleGAN. The off-line meta-learning stage leverages MAML, where the inner loop determines the gradient direction for parameter updates, which is then used in the outer loop optimization to learn a task-adaptive initialization. The on-line transfer phase build a universal discriminator $D_{u}$ to enable efficient domain alignment under limited calibration data.

stage systematically processes and integrates multi-day histor-ical data to construct a comprehensive generalized dataset that facilitates the development of a general model comprising both generator and discriminator components. The subsequent online recognition phase leverages a minimal set of newly acquired data to create a strategic fine-tuning dataset,which enables precise adaptation of the generalized model, ultimately producing an expanded training dataset that serves to train a Convolutional Neural Networks (CNN) discriminator $D_{u}$  optimized for ac-curate classification of current-day data patterns through the model's enhanced transfer learning capabilities.

### B. Rhesus Monkey With Cerebral Cortex Electrode Implantation for Grasping Action Intention Estimation

Microelectrode array implantation was utilized to investi-gate brain-machine interface functionality in rhesus macaques (Macaca mulatta). Specifically, the neural signals recorded were voltage fluctuations (actionpotentials) of neurons, cap-tured via metallic wires or conductive fluid-filled microglass pipettes positioned adjacent to neurons.To achieve simultaneous multi-neuronal recording, we implemented multi-channel mi-croelectrode array technology. Under the chronic experimental paradigm, long-term cortical implantation of microelectrode arrays was necessary, primarily utilizing two types: 32-channel Utah Electrode Array (UEA) and 16-channel Floating Micro-electrode Array (FMA). Two 32-channel Utah arrays were im-planted in the M1 and S1 regions respectively, while a single 16-channel floating array was implanted in the PPC, totaling 80 recording electrodes across the three arrays, as illustrated in Fig. 2(a). To optimize post-operative recovery and minimize infection risk in the experimental subject, we innovatively em-ployed a titanium alloy pedestal with superior biocompatibility for electrode interface fixation, departing from traditional dental cement mounting methods.

As shown in Fig. 3(a), the proposed iBMI was utilized to decoding neural signals of the rhesus monkey to estimate its

<!-- IPS S1 Monkey B -->
![](https://web-api.textin.com/ocr_image/external/03d32ce55ff9d0d3.jpg)


![](https://web-api.textin.com/ocr_image/external/98d48aa951f0cd44.jpg)

(a) (b)

Fig. 2. (a) Cortical electrode in M1, S1, PPC specific location. (b) Rhesus monkeys are trained in a monkey chair.

grasping action intentions. A trained adult male rhesus macaque performed spatially precise reach-to-grasp tasks with the right upper limb while the left was constrained. The experimental apparatus consisted of a strategically positioned central touch pad at the base and three structurally identical target objects on the front panel, each equipped with adjacent LED indicators. This design enabled systematic investigation of grasping behav-iors across different spatial locations while maintaining target object morphological consistency.Our research focused on the neural encoding mechanisms of primate upper limb grasping actions, involving three key brain regions: M1, S1, PPC,with their anatomical locations depicted in Fig. 3(b).

The experiment employed a strictly controlled temporal pro-tocol (Fig. 3(c)). Each trial was initiated by a “Center On”LED signal, requiring the subject to establish contact with the central sensor ("Center Click"). Following a mandatory 500 ms maintenance period, successful maintenance triggered the "Target On” event. Subsequently, the subject executed a precise motor sequence comprising "Center Release" and target acquisition. Target contact ( $"TargetHit"$  had to be maintained for 200 ms before release, with successful completion rewarded with fluid reinforcement. This experimental paradigm estab-lished four distinct behavioral phases: waiting,reaction, reach-ing, and grasping. As shown in Fig. 2(b), the subject was secured

<!-- Authorized licensed use limitedd to:Beijing Jiaotong University. Downloaded on March 04,2026 at 03:48:01 UTC from IEEE Xplore. Restrictions apply. -->

<!-- CHEN et al.: INTRACORTICAL BRAIN-MACHINE INTERFACES WITH HIGH-PERFORMANCE NEURAL DECODING -->

<!-- 521 -->

<!-- Electrodes locations $PPC$ M1 S1 (b) Trial Start Target Light On Center Light On Center Light Off Target Light Off Reward Trial end Waiting Reaching stage stage Reaction Grasping stage stage Center Hit Center Release Target Hit Target Release (c) Bin1 Bin2 Bin3 Bin4 Bin5 Bin6 Bin7 Bin8 Channel 1 IIIII II IIII IIIIII II I III IIII ...... Channel n IIII IIIIII II I IIIIII III IIIII (a) Eigenvector 1 5 2 4 6 2 3 4 ...... Eigenvector n 4 6 4 6 2 1 3 4 -->
![](https://web-api.textin.com/ocr_image/external/542bc77454c19374.jpg)

(d)

Fig.3. The experimental setup of the proposed iBMI system: (a) Experimental protocol for neural signal acquisition during primate spatial reaching and grasping behaviors uutilizing the Omniplex recording platform. (b) Specific locations of cortical electrode placement. (c) Temporal organization of the behavioral paradigm. (d) Structural visualization of the warp activity matrix.

in a custom-designed primate chair, and behavioral performance was optimized through a systematic fluid restriction protocol combined with progressive task difficulty adjustment. Through standardized operant conditioning training, a success rate of 85% was achieved. The neural signals and the behaviors of the rhesus monkey are recorded with the presented experimental setup of the iBMI system for investigating the performance of the following presented neural decoding method.

### C. DMM-WcycleGAN

For neural spike signal decoding in the iBMI system,we propose a novel DMM-WcycleGAN framework that addresses several challenges in neural decoder calibration for iBMI sys-tems. We have constructed a novel adaptive system that achieves efficient neural signal decoding in data-constrained clinical sce-narios by synergistically integrating historical neural recordings with real-time neural data. In contrast to traditional calibra-tion approaches thatfocus on single-decoder optimization, this framework introduces a lightweight computational architec-ture specifically designed for resource-constrained implantable neural devices.

We propose an end-to-end multi-stage optimization frame-work that decomposes conventional neural signal processing into a series of progressive micro-optimization steps, signif-icantly reducing the computational demands on implantable hardwvare. In addition, we introduce a resource-aware online adaptation strategy based on partial parameter freezing and dynamic channel pruning during fine-tuning, which further alleviates memory and computation burdens without compro-mising decoding performance. By incorporating these reduc-tion techniques, we achieve precise alignment of neural pat-terns acrossdifferent recording sessions while maintaining de-coding accuracy. This methodology not only overcomes the computational constraints inherent in implantable neural inter-faces but also provides an efficient solution for neural decoder adaptation in clinical settings where calibration data is limited. Through this framnework, the iBMI system can achieve efficient neural signal decoding in data-constrained clinical scenarios by synergistically integrating historical neural recordings with real-time neural data.

During the experiment, meta-initialization utilized historical data, while fine-tuning was performed by combining 630 sam-ples from the previous day and 10 randomly selected samples from the current day. The experiment involved six calibration trials, conducted across two phases with three calibration ses-sions in each phase, simulating real-world calibration scenarios with limited data to assess the model's adaptability. The final outcome was the dlevelopment of a universal discriminator, capable of classifying actions on the current day, by inputting the 640-dimensional spike signals from the iBMI system to produce classification results. Further details are provided in Section **III-A.**

The DMM-WcycleGAN framework is mainly divided1 into three parts: The meta initialization, the model fine-tuning stage and the the online discriminator training. The meta-initialization stage corresponds to the Off-line Meta Part in Fig.1,and the moodel fine-tuning stage and online discriminator train-ing stage correspond to the On-line Transfer Learning Part in Fig.1.

**1)** **Meta-Initialization**: The proposed meta-initialization mechanism implements a sophisticated approach to parameter initialization by leveraging historical data to construct shared initial parameters that effectively capture cross-temporal task characteristics. Theframework employs an advanced bi-level training architecture to optimize the initial model across het-erogeneous task distributions. The architectural framework en-compasses dual generators $\left(G_{1},\right.$ $\left.G_{2}\right)$ and discriminators $\left(D_{1},\right.$ 

<!-- Authorized licensed use limited to:Beijing Jiaotong University. Downloaded on March 04,2026 at 03:48:01 UTC from IEEE Xplore. Restrictions apoply. -->

<!-- 522 -->

<!-- IEEE TRANSACTIONS ON BIOMEDICAL **ENGINEERING,** VOL. 73, NO. 2, FEBRUARY 2026 -->

$\left.D_{2}\right)$ ,which are strategically designed to identify optimal ini-tialization parameters $\theta _{G}$  and $\theta _{D}$ ,facilitating expedited model convergence on novel tasks through minimized gradient descent iterations.

In the context of data processing, the meta-initialization dataset within DMM-WcycleGAN consists of a comprehensive collection of historical data $H$ , which includes all data collected prior to the current day's experiment. The framework employs a systematic sampling method, where the sampled subsets are time-decomposed into multiple optimization tasks $T_{i}$ . For ex-ample, for the S2D2 task, its $T_{i}$ is composed of pairs such as S1D1-S1D2,S1D2-S1D3,S1D3-S1D4,and S1D4-S2D1.Each task $T_{i}$ is split into source domain $X_{i}^{\text {meta}}$  and target domain $Y_{i}^{\text {meta}}$ ,where $X_{i}^{\text {ma}}$ $\in \mathbb {R}^{M\times N}$  and $Y_{i}^{\text {ma}}\in \mathbb {R}^{M\times N}$ .Here,M denotes the sampling size, and N represents the dimensionality of the spike signal from the iBMI system. Additionally, $X_{i}^{\text {meta}}$ and $Y_{i}^{\text {met}}$  are derived from historical datasets,with the source domain coming from the previous day's data and the target domain coming from the following day's data. The datasets are then partitioned into training sets $\left(X_{i}^{\text {trai}}\right.$ $\left.Y_{i}^{\text {trai}}\right)$ and test sets $\left(X_{i}^{\text {test}},Y_{i}^{\text {test}}\right)$ following an 8:2 ratio. The aggregation of multiple tasks forms a comprehensive task set $T$ .

The mneta-initialization protocol incorporates a dual-loop op-timization paradigm. Within the inner loop, the framework executes gradient updates for generators $G_{1}$ and $G_{2}$ utilizing existing parametric configurations on the training corpus. A notable optimization strategy involves the exclusive focus on generator parameter optimization within the inner loop,while constraining discriminator updates to the outer loop, resulting in significant computational efficiency gains. For any given task instance $T_{i}$ ,the generator's initialization parameters are utilized to execute gradient descent operations across both the source domain training set $X_{i}^{\text {rain}}$ and target domain training set $Y_{i}^{\text {train}}$  .The optimization is governed by the following function:

$$\theta _{\mathcal {G}_{i}}^{\prime }=\theta _{G}-α\nabla _{\theta _{G}\mathcal {T}_{i}}\left(\theta _{G},X_{i}^{\text {train}},Y_{i}^{\text {train}}\right)\tag{1}$$

in this formulation, $\theta _{G}$ denotes the shared initialization param-eters of the coupled generators $G_{1}$ and $G_{2}$ ,while $\theta _{\mathcal {G}_{i}}^{\prime }$ represents the optimized parametric configuration for task $\mathcal {T}_{i}$ following inner-loop adaptation. The inner-loop learning rate is param-eterized by $α$ . The loss term $\mathcal {L}\left(\theta _{G},X^{\text {train}},\right.$ $\left.Y^{\text {train}}\right)$ encapsulates the generator's objective function evaluated over the training distributionof task $\mathcal {T}_{i}$ , achieving dynamic equilibrium through concurrent optimization of the dual generators $G_{1}$ and $G_{2}$ $\mathcal {L}_{\mathcal {T}_{i}}$ will be further explained in Section II-C2 In the outer loop,the model uses the data obtained from multiple innerloops to update the generator and discriminator, and its optimization goal is to obtain a new pair of generators and minimize the sum of the loss of all task test sets:

$$\min _{\theta _{G}}\sum _{\mathcal {T}_{i}}\mathcal {L}_{\mathcal {T}_{i}}\left(\theta _{\mathcal {G}_{i}}^{\prime },X_{i}^{\text {test}},Y_{i}^{\text {test}}\right)\tag{2}$$

Substituting the definition of $\theta _{\mathcal {G}_{i}}^{\prime }$ into the above objective:

$$\min _{\theta _{G}}\sum _{\mathcal {T}_{i}}\mathcal {L}_{\mathcal {T}_{i}}\left(\theta _{G}-α\nabla _{\theta _{G}}\mathcal {L}_{\mathcal {T}_{i}}\left(\theta _{G},X^{\text {train}},Y^{\text {train}}\right),X_{i}^{\text {test}},Y_{i}^{\text {test}}\right)$$

(3)

<!-- $L_{idt}=\left\|Y-G_{1}(Y)\right\|_{1}$ Y $G_{1}$ $Y^{\prime }$ ' X $D_{1}$ $L_{adv}$ $X^{\prime }$ $\left.L_{cyc}=\left\|Y-Y^{\prime \prime }\right)\right\|_{1}$ $G_{2}$ Y' $Y^{\prime }$ $L_{adv}$ $D_{2}$ Y Y $L_{idt}=\left\|X-G_{2}(X)\right\|$ 1 -->
![](https://web-api.textin.com/ocr_image/external/e4788a65fb8654ca.jpg)

Fig.4. The structure of Wasserstein CycleGAN.

These losses are summed and used to update the initial pa-rameter $\theta _{G}:$ 

$$\theta _{G}\leftarrow \theta _{G}-\beta _{G}\sum _{\mathcal {T}_{i}}\nabla _{\theta _{G}}\mathcal {L}_{\mathcal {T}_{i}}\left(\theta _{G}\right.\quad \left.-α\nabla _{\theta _{G}}\mathcal {L}_{\mathcal {T}_{i}}\left(\theta _{G},X_{i}^{\text {train}},Y_{i}^{\text {train}}\right),X_{i}^{\text {test}},Y_{i}^{\text {test}}\right)\tag{4}$$

in this formulation, $\beta$  represents the outer loop learningrate parameter. The computational framework necessitates simulta-neous optimization of dual generators, while the calculation of second-order derivatives for gradients with respect to $\theta _{G1}$  and $\theta _{G2}$  imposes substantial computational complexity and resource utilization constraints. Moreover, owing to the inherent local smoothness characteristics, parametric perturbations within in-dividual generators demonstrate minimal variance [27], espe-cially under conditions of reduced learning rates, facilitating the following approximation:

$$\nabla _{\theta _{G}}\theta _{\mathcal {G}_{i}}^{\prime }\approx 1.\tag{5}$$

Thus, the outer loop update rule is simplified to:

$$\theta _{G}\leftarrow \theta _{G}-\beta \sum _{\mathcal {T}_{i}}\nabla _{\theta _{G}}\mathcal {L}_{\mathcal {T}_{i}}\left(\theta _{\mathcal {G}_{i}}^{\prime },X_{i}^{\text {train}},Y_{i}^{\text {train}}\right)\tag{6}$$

We update the discriminator $D_{1}$  $D_{2}$ with the pseudo-data generated by the new generator and the original data. Its update loss function is as follows:

$$\theta _{D}=\theta _{D}-\beta _{D}\nabla _{\theta _{D}}\mathcal {L}_{D}\left(\theta _{G},\theta _{D},X_{i}^{\text {test}},Y_{i}^{\text {test}}\right)\tag{7}$$

$\mathcal {L}_{D}$ is defined as follows:

$$\mathcal {L}_{D}\left(\theta _{G},\theta _{D},X,Y\right)=-\mathbb {E}_{X\;p_{\text {data}}(X)}[\log D(X)]\quad -\mathbb {E}_{Y\;p_{G}(Y)}[\log (1-D(G(Y)))]$$

2) Model Fine-Tuning: In the model fine-tuning phase,our framework leverages complete spike signal from the iBMI sys-tem from the previous day $X\in \mathbb {R}^{ExN}$  in conjunction with extremely limited spike signal from1 the subsequent day) $Y\in$ $\mathbb {R}^{I\times N}$ ,where N denotes feature dimensionality, while E and I represent the sampling cardinalities for source and target domains respectively. The fine-tuning protocol employs an enhanced bidirectional mapping architecture, as illustrated in Fig.4. This architectural framework comprises dual generators: generator $G_{1}$ specifically optimizes feature transfer from source

<!-- Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on March 04,2026 at 03:48:01 UTC from IEEE Xplore. Restrictions apply. -->

<!-- CHEN et al.: INTRACORTICAL BRAIN-MACHINE INTERFACES WITH HIGH-PERFORMANCE NEURAL DECODING -->

<!-- 523 -->

domain X to target domain Y,while generator $G_{2}$  executes inverse feature mapping, facilitating precise transformation from target domain Y to source domain $X$ , thereby establishing a comprehensive bidirectional feature alignment mechanism. Discriminators $D_{1}$ and $D_{2}$ are employed to validate the authen-ticity of generator-synthesized data,establishing an adversarial learning paradigm.

To further reduce the computational overhead during online calibration, we incorporate a lightweight fine-tuning scheme in this stage. Specifically, the lower convolutional layers of $G_{1}$ and $G_{2}$ ,which primarily capture invariant neural characteristics, are frozen to avoid redundant parameter updates. Only the upper lay-ers, responsible for task-specific domain adaptation, are actively updated. In addition, we apply a progressive channel pruning mechanism to the trainable layers, where low-activation filters are dynamically identified and removed based on accumulated activation statistics.

To enhance the robustness of feature transfer, this frame-work incorporates a multi-objective joint optimization strat-egy. Through the unification of loss functions from both meta-initialization and fine-tuning phases, the model demonstrates significantly improved generalization performance in few-shot scenarios. The joint loss function is defined as:

$$L=λ_{a}\mathcal {L}_{adv}+λ_{c}\mathcal {L}_{\text {cyclic}}+λ_{i}\mathcal {L}_{\text {identity}},\tag{9}$$

the parameters $λ_{a}$ $λ_{c}$ and $λ_{i}$ are weights related to adversarial loss, cycle consistency loss and identity loss,respectively.These values are used as hyperparameters during network training. Additionally, the loss function in the meta-initialization phase is the same as that in the fine-tuning phase:

$$\mathcal {L}_{\mathcal {T}}=λ_{a}\mathcal {L}_{adv}+λ_{c}\mathcal {L}_{\text {cyclic}}+λ_{i}\mathcal {L}_{\text {identity}},\tag{10}$$

the core objective is to ensure that the model paramneters initial-ized through the obtained meta-initialization can quickly adapt to new tasks, while also enabling effective optimization during the subsequent fine-tuning phase.

The adversarial loss enables adaptive neural signal mapping across different recording sessions without requiring paired data. It ensures that the generated neural signals from historical recordings match the statistical properties of current neural pat-terns by training generators and discriminators in a competitive framework. Adversarial loss is defined below:

$$\mathcal {L}_{adv}=-\mathbb {E}_{Y\;p_{Y}}\left[D_{2}(Y)\right]+\mathbb {E}_{X\;p_{X}}\left[D_{2}\left(G_{1}(X)\right)\right]$$

$$+λ_{gp}\mathbb {E}_{G_{1}(X)}\left[\left(\left\|\nabla _{G_{1}(X)}D_{2}\left(G_{1}(X)\right)\right\|_{2}-1\right)^{2}\right]$$

$$-\mathbb {E}_{X\;p_{X}}\left[D_{1}(X)\right]+\mathbb {E}_{Y\;p_{Y}}\left[D_{1}\left(G_{2}(Y)\right)\right]$$

$$+λ_{gp}\mathbb {E}_{G_{2}(Y)}\left[\left(\left\|\nabla _{G_{2}(Y)}D_{1}\left(G_{2}(Y)\right)\right\|_{2}-1\right)^{2}\right]\tag{11}$$

where $λ_{gp}$ is the weight hyperparameter for the gradient penalty term, used to adjust the impact of the gradient penalty on the overall loss function.

The cycle consistency loss maintains the essential character-istics of neural activities during domain adaptation. It ensures that when neural signals are transformed between historical and current domains, the key neural firing patterns and temporal dynamics are preserved. Cycle consistency loss is defined below:

$$\mathcal {L}_{\text {cyclic}}\left(G_{1},G_{2}\right)=\mathbb {E}_{X\;p_{X}}\left[\left\|G_{2}\left(G_{1}(X)\right)-X\right\|_{2}\right]\quad +\mathbb {E}_{Y\;p_{Y}}\left[\left\|G_{1}\left(G_{2}(Y)\right)-Y\right\|_{2}\right]$$

The identity loss ensures stability in neural decoding when signal characteristics remain consistent between sessions. It helps preserve distinct neural patterns associated with specific motor intentions. Identify loss is defined below:

$$\mathcal {L}_{i}\left(G_{1},G_{2}\right)=\mathbb {E}_{X\;p_{X}}\left[\left\|G_{1}(X)-X\right\|_{1}\right]\quad +\mathbb {E}_{Y\;p_{Y}}\left[\left\|G_{2}(Y)-Y\right\|_{1}\right]\tag{13}$$

3) **Online** **Discriminator** Training: For real-time neural de-coding in the iBMI system, we train a universal online dis-criminator that processes the 640-dimensional neural features representing different grasping intentions. We first utilize the optimized generator G1 to map historical neural patterns into the current neural signal space:

$$Y_{\text {fake}}=G_{1}(X)\tag{14}$$

Leveraging the synthesized $Y_{\text {fake}},$ we construct an augmented dataset that facilitates the training of a universal discriminator $D_{u}$ through iterative optimization procedures.Leveraging the synthesized $Y_{\text {fak}}$ generated by the optimized generator $G_{1}$ ,we construct an augmented dataset combining both the transformed source domain samples $Y_{\text {fak}}$ and the target domain samples Y.This augmented dataset is subsequently utilized to train a universal discriminator $D_{u}$  implemented as an online CNN model. The discriminator $D_{u}$ then learns to classify these neural patterns into three categories corresponding to different target positions in the experimental grasping task, enabling the iBMI system to accurately decode the monkey's intended actions in real-time while maintaining stable performance through online adaptation. The training process employs iterative optimization procedures to enhance $D_{u}\text {'s}$ ability to generalize across diverse domain distributions while maintaining high discriminative ca-pacity.

The CNN model $D_{u}$ is designed with a hierarchical structure comprising convolutional layers, pooling layers for and fully connected layers. The total loss function $\mathcal {L}_{\mathrm {CNN}}$  for training the CNN model is formulated as follows:

$$\mathcal {L}_{\mathrm {CNN}}=\mathcal {L}_{\mathrm {CE}}+λ_{r}\mathcal {L}_{\mathrm {reg}},\tag{15}$$

where the total loss function $\mathcal {L}_{\mathrm {CNN}}$  is composed of two primary components: the cross-entropy loss $\mathcal {L}_{\mathrm {CE}}$ ,which measures the difference between the predicted and true class distributions, and the regularization loss $\mathcal {L}_{\mathrm {reg}}$ ,which prevents overfitting by penalizing excessive model complexity [32]. The comprehen-sive algorithmic framework is delineated in Algorithm 1.

## III. PERFORMANCE EVALUATION OF THE PROPOSED IBMI

### SYSTEM

#### A. Data Preparation

In neural decoding, we implemented a three-class classifica-tionframework for target position determination. Neural and behavioral data were recorded using a 128-channel Omniplex

<!-- Authorized licensed use limited to:Beijing Jiaotong University. Downloaded on March 04,2026 at 03:48:01 UTC from IEEE Xplore. Restrictions apply. -->

<!-- 524 -->

<!-- IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 73, NO. 2, FEBRUARY 2026 -->

### Algorithm 1: DMM-WcycleGAN Algorithm.

**Input**: Historical data with $N$ features over $M$ days and

run $E$  times a day: $\left\{H_{m}\right\}_{m=1}^{}$ ,where $H_{m}\in \mathbb {R}^{E\times };$ 

Previous day's data $X\in \mathbb {R}^{E\times N}$ ; Small current sample set

$Y\in \mathbb {R}^{I\times N},$ Where I is the number of calibration

experiments; Hyperparameters: $λ_{a}=1,λ_{c}=10,λ_{i}=5,$ 

$λ_{gp}=3,$  $α=3,$  $\beta =3,$ ,Maxmetaepoch =20

$$\text {Max_{inner}_{step}}=4,\text {Max_{T}}\quad \text {rans_{epoch}}=200,$$

$$\text {Max_{CNN}_{epoch}}=100$$

**Output:** On-line CNN Discriminator $D_{u}$ 

1: Initialize parameters: $\theta _{G_{1}},\theta _{G_{2}},\theta _{D_{1}},\theta _{D_{2}},D_{u}$ 

2: **for** Metaepoch=1:Maxmetaepoch do

3:

Extract source and target domains from historical data: $X_{M}$ and $Y_{M}\in \mathbb {R}^{ExN}$ 

4:

5:

Set $\theta _{G_{1}}=\theta _{G_{1}}^{\prime },\theta _{G_{2}}=\theta _{G_{2}}^{\prime },\theta _{D_{1}}=\theta _{D_{1}}^{\prime },\theta _{D_{2}}=\theta _{D_{2}}^{\prime }$ 

**for** inner $\text {step}=1$ : Maxinnerstep **do**

6:

Generate fake data based on $\theta _{G_{1}}^{\prime },\theta _{G_{2}}^{\prime },X_{M},Y_{M}$ 

7:

Calculate generator loss $\mathcal {L}_{G}$ and gradient values

(using equations (1)

8:

Update generator parameters $\theta _{G_{1}}^{\prime },\theta _{G_{2}}^{\prime }$ 

9:

**end for**

10:

Calculate parameter change direction (using

equations (2)to (3))

11:

Update generator parameters $\theta _{G_{1}},\theta _{G_{2}}$ (using equation (4) to(6))

12:

Update discriminator parameters $\theta _{D_{1}},\theta _{D_{2}}$ (using equation (7) to(8))

13:**end for**

14: **for** Transepoch $=1$ :MaxTransepoch do

15:

Freeze lower convolutional blocks of $G_{1}$  and $G_{2}$ 

16:

Generate fake data based on $G_{1},G_{2},X,Y$ 

17:

Update trainable generator parameters $\theta _{G_{1}},\theta _{G_{2}}$ 

18:

Calculate discriminator loss L and gradient values

19:

Update discriminator paramneters $\theta _{D_{1}},\theta _{D_{2}}$ 

20: **end for**

21:Generate common dataset D based on $G_{1}(X)$ and Y

22: **for** CNNepoch=1:MaxCNNepoch do

24:

23: Calculate CNN network loss $\mathcal {L}_{C}$ and gradient values Update CNN model parameters $\theta _{D_{u}}$ 

25:**end for**

26**: return** $D_{u}$ 

system (Plexon,Inc.). Signal synchronization was achieved by analyzing neural activity within a 400 ms time window (±200 ms relative to the central sensor detachment). Neural activity acquisition utilized a 40 kHz sampling frequency with bandwidth filtration spanning 250 Hz to 6 kHz. Spike detection was performed using threshold crossing methodology,where action potentials were identified when the signal crossed a threshold set at -4.5 times the root mean square value of each channel's spike band [8], yielding discrete spike events for neural activity analysis. Subsequently, MIN-MAX normalization was applied to standardize the data. Neural signals were acquired from 80 channels and segmented into 8 temporal bins (50 ms each), generating a 640-dimensional feature space based on neuronal firing rates. The experimental protocol encompassed 12 datasets collected across three four-day periods. The initial period was dedicated to meta-learning pretraining, followed by subsequent periods for transfer learning optimization. Daily neural recordings yielded 640-dimensional data (represented as an $80\times 8$ channel-bin matrix), as illustrated in Fig. 3(d).

The experimental data were systematically partitioned into two components: Meta initialization and Fine-tuning phases. The Meta initialization dataset comprises all historical data available up to the current day ( $H$ ), ensuring continuous in-corporation of the most recent neural patterns. The Fine-tuning dataset is constructed by combining 630 samples ( $E$ ) from the day preceding calibration with 10 randomly selected samples (I) from the current day, where each sample maintains a feature dimensionality $N$ of 640. Specifically, the previous day's data serves as the source domain ( $X$ ),while the randomly selected current-day samples constitute the target domain (Y).This random sampling strategy effectively simulates the limited data collection process inherent in real-world calibration scenarios. Notably, the composition of both meta-learning and transfer learning datasets evolves dynamically with each calibration iteration, reflecting the expansion of the historical data pool and the selection of new target domain samples. The experi-mental protocol encompassed eight days (S1D1, S1D2,S1D3, S1D4, S2D1, S2D2, S2D3, S2D4) across two distinct phases, with three calibrationsessions per phase, yielding a total of six calibration experiments. This structured data organization methodology enables rigorous evaluation of algorithmic perfor-mance under temporal evolution while maintaining controlled conditions for assessing transfer learning capabilities. The lim-ited sampling of 10 iterations in the target domain ensures unbiased evaluation of model adaptability while simulating re-alistic scenarios where only limited current data is available for calibration.

### B. Performance Comparison Between DMM-WCycleGAN and Other Decoding Schemes

In the monkey grasping experiment, DMM-WcycleGAN demonstrated high and stable decoding performance.To validate the effectiveness of this approach, we implemented five com-mon calibration algorithms and four transfer learning methods as benchmarks. All algorithms used a 640-dimensional spike signal as input. The specific configurations of these models are provided in the supplementary materials for reference. All algorithms were implemented in Python, and all reported results represent the average of ten experimental trials.

Table I presents a performance comparison of different de-coding algorithms, including the proposed DMM-WcycleGAN, CSP-SVM, AC-SVM, CNP, CASS, DANN, HDNN, GAN, CycleGAN and NF-ML. Empirical analysis demonstrates DMM-WcycleGAN's superior recognition capabilities,achiev-ing mean accuracy exceeding 93.21% across all evaluated datasets. Fig.5 illustrates the comparative performance metrics and variability of each methodology, while Table II presents comprehensive evaluation metrics (accuracy, precision, recall, F1-score) for the DMM-WcycleGAN implementation.

<!-- Authorized licensed use limited to:Beijing Jiaotong University. Downloaded on March 04,2026 at 03:48:01 UTC from IEEE Xplore. Restrictions apply. -->

<!-- CHEN et al.: INTRACORTICAL BRAIN-MACHINE INTERFACES WITH HIGH-PERFORMANCE NEURAL DECODING -->

525

TABLEI

THE AVERAGE RECOGNITION ACCURACY OF DIFFERENT METHODS (%)


| Method | AC-SVM | CSP-SVM | CNP | CASS | DANN |
| --- | --- | --- | --- | --- | --- |
| Avg. Accuracy | 89.05±4.48 | 74.48±8.15 | 84.99±2.62 | 85.40±2.72 | 88.94±3.78 |
| Method | HDNN | NF-ML | GAN | CycleGAN | DMM-WcycleGAN |
| Avg.Accuracy | 89.25±2.88 | 89.92±6.75 | 84.46±3.61 | 90.25±3.83 | 93.21±4.99 |


<!-- Accuracy of Different Methods 100 Reference Accuracy AC-SVM 80 CSP-SVM CNP CASS DANN （迟）Aoe Jno2y 60 HDNN NF-ML 40 GAN CycleGAN DMM-WcycleGAN 20 0 S1D2 S1D3 S1D4 Date -->
![](https://web-api.textin.com/ocr_image/external/760e2dc54247928f.jpg)

Fig. 5. We evaluated seven different decoder calibration approaches across multiple temporal sessions. For each dataset, accuracy mea-surements were derived by averaging the results from 10 independent calculations, where training samples were randomly drawn from the available data. A red dashed line is included in each plot to represent the baseline accuracy, serving as a reference point to evaluate the effectiveness of the calibration approaches. Session identifiers follow a systematic naming convention: the format 'SxDy' indicates Session x, Day y (e.g., S1D2 represents data collected on the second day of the first session).

TABLE II

THE AVERAGE ACCURACY, PRECISION, RECALL, AND F1-SCORE OF THE

DMM-WCYCLEGAN METHOD IN EXPERIMENTS (%)


| Date | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- |
| S1D2 | 96.59 | 96.84 | 96.59 | 96.61 |
| S1D3 | 97.73 | 97.82 | 97.73 | 97.73 |
| S1D4 | 94.94 | 94.96 | 94.96 | 94.96 |
| S2D2 | 94.32 | 94.34 | 94.32 | 94.30 |
| S2D3 | 91.72 | 92.33 | 91.72 | 91.73 |
| S2D4 | 83.93 | 87.25 | 83.93 | 84.31 |


The comparative analysis across algorithms revealed varied performance levels. CSP-SVM exhibited limited effectiveness (69.75%) due to poor generalization, especially under non-stationary neural signals. HDNN achieved strong performance (89.25%) by leveraging its hybrid CNN-Transformer structure. However, both DANN and CASS struggled to adapt in few-shot settings (88.94% and 85.40%, respectively), reflecting a mis-match between self-supervised learning mechanisms and sparse calibration data. In contrast, AC-SVM and DANN performed more competitively (89.04% and 88.61%) by leveraging adap-tive strategies to compensate for domain shifts.Traditional GAN achieved moderate performance (79.52%) by synthesizing data, but lacked the structural constraints necessary to preserve essen-tial signal features. CycleGAN, enhanced by cycle-consistency loss, surpassed GAN (90.2%), yet its improvements remained limited when faced with significant domain shifts or minimal calibration samples. CycleGAN, used in ablation studies, pro-duced a comparable result of 90.25%.

Notably, DMM-WcycleGAN achieved the highest accuracy (93.21%) and exhibited exceptional performance in iBMI sce-narios that are inherently characterized by substantial do-main distribution discrepancies and severely limited calibration data-challenges arising from temporal neural signal drift and practical data collection constraints [6], [7]. Its meta-learning-enhanced architecture enabled precise domain adaptation using only ten calibration trials per session, effectively minimizing inter-domain divergence and improving model generalization.

The exceptional performance of DMM-WcycleGAN can be primarily attributed to its sophisticated capability in minimizing the distributional discrepancy between historical and current data distributions, wherein this effectiveness was systematically evaluated through acomprehensive visualization analysis utiliz-ing dimensionally-reduced representations via Principal Com-ponent Analysis (PCA) of datasets comprising previous-day historical data and current-day test data. The visualization results from Dataset 1 in Session 1, as illustrated in Fig. 6, demon-strate a distinct initial separation between historical and current data distributions in their original state; however,following the application of DMM-WcycleGAN, a remarkable reduction in inter-distribution distance is observed, indicating successful domain adaptation and feature alignment. As further evidence, Fig. 7 presents the confusion matrices before and after applying DMM-WcycleGAN, clearly illustrating the improvements in inter-domain classification consistency and supporting the above findings. The comprehensive confusion matrices presented in Fig.8 reveal a consistent pattern across multiple experimental sessions wherein certain categories consistently demonstrate suboptimal classification performance, as exemplified in S1D2where category 2 exhibits notably lower classification accuracy compared to the other two categories.

### C. Quantitative Evaluation of Decoder Calibration Time Reduction in DMM-WCycleGAN

A systematic experimental framework was implemented to assess the calibration efficiency of DMM-WcycleGAN in rela-tion to established benchmark methodologies, including HDNN, DANN,and GAN architectures.These benchmark methods were selected based on their competitive accuracy demonstrated in previous experiments, and they represent typical approaches

<!-- Authorized licensed use limited to:Beijing Jiaotong University. Downloaded on March 04,2026 at 03:48:01 UTC from IEEE Xplore. Restrictions apply. -->

<!-- 526 -->

<!-- IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 73, NO.2, FEBRUARY 2026 -->

TABLE III

THE NUMBER OF CURRENT SAMPLES FOR DIFFERENT METHODS


| Method | S1D2 | S1D3 | S1D4 | S2D1 | S2D2 | S2D3 | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HDNN | 49 | 76 | 35 | 27 | 36 | 35 | 42 |
| DANN | 56 | 63 | 41 | 56 | 52 | 40 | 51 |
| cycleGAN | 43 | 52 | 46 | 41 | 43 | 44 | 45 |
| DMM-WcycleGAN | 10 | 10 | 10 | 10 | 10 | 10 | 10 |


<!-- 10.0 PCA of Dataset X and Dataset Y 7.5 Dataset X Dataset Y 5.0 ZauauoduoCPdliCuAd 2.5 0.0 -2.5 -5.0 -7.5 -10.0 -10.0 -7.5 -5.0 -2.5 0.0 2.5 5.0 7.5 10.0 Principal Component 1 -->
![](https://web-api.textin.com/ocr_image/external/678bbc2d5de5acd2.jpg)

(a)

<!-- 10.0 PCA of Dataset X and Dataset Y 7.5 Dataset X Dataset Y 5.0 ZuauoduoCPdlICuAd 2.5 0.0 -2.5 -5.0 -7.5 -10-10.0 -7.5 -5.0 -2.5 0.0 2.5 5.0 7.5 10.0 Principal Component 1 -->
![](https://web-api.textin.com/ocr_image/external/121bf2692adcd447.jpg)

(b)

Fig.6. Distribution analysis of S1D1 and S1D2, illustrating the spatial characteristics prior to and following DMM-WcycleGAN implementation. (a) Before applying the DMM-WcycleGAN method. (b) After applying the DMM-WcycleGAN method.

<!-- 80 82.35 9.80 7.84 60 공pdel onA TAotaeZko6ae SAota1e 8.78 77.56 13.66 40 4.52 9.05 86.43 20 Category1 Category2 Category3 Predicted labels -->
![](https://web-api.textin.com/ocr_image/external/69eafb4a046698c0.jpg)

<!-- 97.06 1.96 0.98 80 60 발del anL TAo6ae zkobae CAo6a18 2.93 89.76 7.32 40 0.00 3.86 96.14 20 Category1 Category2 Category3 -0 Predicted labels -->
![](https://web-api.textin.com/ocr_image/external/067aa493aca29436.jpg)

(a) (b)

Fig. 7. The confusion matrix before and after using the DMM-WcycleGAN method. (a) The confusion matrix of the discriminator trained directly using the data of the previous day. (b) The confusion matrix using the DMM-WcycleGAN method. (a) Before applying the method. (b) After applying the method.

from deep learning (HDNN) and transfer learning paradigms (DANN and GAN). This representative selection ensures a com-prehensive and fair evaluation of calibration efficiency across different algorithmic families. A lower calibration data require-ment indicates a shorter calibration time. Therefore, we assess the calibration efficiency of each method by comparing the num-ber of samples required to achieve the same decoding accuracy. Since each calibration trial consumes a fixed amount of time in practice, the number of required samples can be directly used as a proxy for calibration time. This evaluation approach is similar to the quantitative analysis of calibration time employed by Zhang et al. [8] in their research. The experimental results are presented in Table III.The data reveal that DMM-WcycleGAN exhibits superior sample efficiency compared to traditional methods. The performance of HDNN, DANN, and CycleGAN was relatively limited, as all three required more than 40 calibration samples to achieve the same decoding accuracy that DMM-WcycleGAN at-tained using only 10 samples. These results vividly illustrate the significant differences in sample efficiency among the methods.

Further quantitative analysis unveils the remarkable advan-tage of DMM-WcycleGAN in reducing the calibration sample requirement. Compared to HDNN, DANN, and CycleGAN,the calibration sample requirement of DMM-WcycleGAN is re-duced by 76.19%(10/42),80.39%(10/51),and 77.78%(10/45), respectively. This result strongly proves the outstanding data utilization efficiency of DMM-WcycleGAN. The substantial reduction in calibration sample requirement directly leads to a significant decrease in calibration time, highlighting the unique advantage of DMM-WcycleGAN in calibration efficiency.This advantage is particularly prominent in application scenarios where data acquisition is time-consuming or restricted. The efficient calibration process of DMM-WcycleGAN makes it the best choice for practical needs, especially in real-world scenarios that require rapid and low-cost calibration. DMM-WcycleGAN is undoubtedly a feasible solution in such cases.

### D. Quantitative Evaluation of Reduced Computational Burden in DMM-WCycleGAN

To further evaluate the comnputational efficiency and de-ployment practicality of the proposed lightweight DMM-WcycleGAN framework, we adopted a spike-level evaluation methodology inspired by the per-spike computational cost framework [33],[34]. This framework considers both inference efficiency and resource footprint as key indicators of real-world applicability. As shown in Table VI, we report a comprehensive comparison of decoding accuracy and resource consumption across several representative algorithms. Notably, in order to ensure fair comparison regarding decoder construction, only the online fine-tuning component of DMM-WcycleGAN is included in the computation budget.

The results clearly demonstrate the efficiency advantage of our approach. The proposed DMM-WcycleGAN consumes only 0.57 MB of memory, which is significantly smaller than that of HDNN (1.41 MB) and CycleGAN (0.69 MB). Its per-inference computational load is also markedly reduced, requiring only 18.39 FLOPs, compared to 30.97 FLOPs and 26.48 FLOPs for

<!-- Authorized licensed use limited to:Beijing Jiaotong University. Downloaded on March 04,2026 at 03:48:01 UTC from IEEE Xplore. Restrictions apply. -->

<!-- CHEN et al.: INTRACORTICAL BRAIN-MACHINE INTERFACES WITH HIGH-PERFORMANCE NEURAL DECODING -->

527

<!-- 97.06 0.00 2.94 80 60 Sadel anL TAo6ateO ZAo6a1eo ckofa1e 0.98 92.68 6.34 40 0.00 0.48 99.52 20 Category1 Category2 Category3 -0 Predicted labels -->
![](https://web-api.textin.com/ocr_image/external/023940ff97d82c69.jpg)

(a)

<!-- 97.06 1.96 0.98 80 60 Sadel anL TAo6aed Zo6a1e co5a1eo 2.93 89.76 7.32 40 0.00 3.86 96.14 20 Category1 Category2 Category3 0 Predicted labels -->
![](https://web-api.textin.com/ocr_image/external/d87968f2a4fc1c1b.jpg)

(d)

<!-- 99.51 0.49 0.00 80 60 Sadel an4 TAo6aeO ZAobaeo ckofa1e 4.41 95.10 0.49 40 1.44 0.00 98.56 20 Category1 Category2 Category3 -0 Predicted labels -->
![](https://web-api.textin.com/ocr_image/external/b8377d8a6ea10db9.jpg)

(b)

<!-- 99.01 0.99 0.00 80 60 SadelanL TAofaed Zho6a1eC EAo6a1e 10.73 88.78 0.49 40 5.29 7.21 87.50 20 Category1 Category2 Category3 0 Predicted labels -->
![](https://web-api.textin.com/ocr_image/external/72e2db30a18ecbe0.jpg)

(e)

<!-- 94.61 2.45 2.94 80 60 Sadel an4 TAo6a1eS ZAo6a1e chofae 4.41 94.12 1.47 40 1.44 2.40 96.15 20 Category1 Category2 Category3 Predicted labels -->
![](https://web-api.textin.com/ocr_image/external/9ccdcecef3fc3310.jpg)

(c)

<!-- 78.43 21.57 0.00 80 60 SiadelanL TAioba1ed Zko6a1eC EAo5a1eO 4.90 95.10 0.00 40 1.92 19.71 78.37 20 Category1 Category2 Category3 0 Predicted labels -->
![](https://web-api.textin.com/ocr_image/external/aaa13f7c96e2d561.jpg)

(f)

Fig.8. Confusion matrices of the DMM-WcycleGAN in different days. Category1, Category2 and Category3 represent three different types of monkey grasping classifications. (a) S1D2. (b) S1D3. (c) S1D4. (d) S2D2. (e) S2D3. (f) S2D4.

TABLE IV

THE AVERAGE RECOGNITION ACCURACY OF LIGHTWEIGHT AND NON-LIGHTWEIGHT METHODS (%)


| Method | S1D2 | S1D3 | S1D4 | S2D1 | S2D2 | S2D3 | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DMM-WcycleGAN | 96.59 | 97.73 | 94.94 | 94.32 | 91.72 | 83.93 | 93.21 |
| MAML-WcycleGAN | 96.25 | 96.69 | 94.13 | 94.51 | 90.97 | 86.36 | 93.15 |


TABLE V

THE AVERAGE SYSTEM EXECUTION TIME FOR LIGHTWEIGHT AND NON-LIGHTWEIGHT METHODS (MIN)


| Method | S1D2 | S1D3 | S1D4 | S2D1 | S2D2 | S2D3 | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DMM-WcycleGAN | 25.3 | 26.7 | 24.6 | 26.2 | 23.6 | 24.7 | 25.2 |
| MAML-WcycleGAN | 39.7 | 40.5 | 39.4 | 40.1 | 38.3 | 41.5 | 39.9 |


TABLE VI

THE AVERAGE COMPUTATIONAL POWER OVERHEAD OF DIFFERENT REPRESENTATIVE METHODS


| Method | Footprint(MB) | FLOPs per Inference | Execution Frequency (Hz) | MACs per Spike | Accuracy(%) |
| --- | --- | --- | --- | --- | --- |
| HDNN | 1.41 | 30.97 | 10 | 2.18 | 89.76 |
| DANN | 1.03 | 12.57 | 10 | 1.05 | 88.61 |
| CycleGAN | 0.69 | 26.48 | 10 | 2.65 | 90.25 |
| DMM-WcycleGAN | 0.57 | 18.39 | 10 | 1.84 | 93.21 |


HDNN and CycleGAN,respectively. Furthermore, in terms of multiply-accumulate operations per spike (Per-Spike MACs), DMM-WcycleGAN achieves a notably lower value of 1.84 M, outperforming CycleGAN's 2.65 M. These findings underscore the enhanced computational efficiency of our model in low-resource neural decoding scenarios.

Building upon the experimental results, a comprehen-sive comparative analysis was conducted between the proposed lightweight DMM-WcycleGAN and the baseline method MAML-WcycleGAN, which serves as the ablation ver-sion without dimensionality reduction and optimization strate-gies. Both methods were evaluated using the same computer system equipped with an NVIDIA RTX 4090 GPU to ensure fairness and reproducibility. The experimental outcomes,de-tailed in Tables IV and V, reveal significant distinctions in both recognition performance and computational efficiency. Specifi-cally, DMM-WcycleGAN demonstrates a 36.8% reduction1in computation time compared to MAML-WcycleGAN,while

<!-- Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on March 04,2026 at 03:48:01 UTC from IEEE Xplore. Restrictionsapply. -->

<!-- 528 -->

<!-- IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 73, NO. 2, FEBRUARY 2026 -->

TABLE VII

COMPARISION WITH RELATED RESEARCH WORKS


| Author | Year | Target Domain<br>Data Quantity | Pre-training | Classification Algorithm | Class Types | Accuracy(%) |
| --- | --- | --- | --- | --- | --- | --- |
| Zhang et al.[8] | 2020 | 10 | Yes | SUTL | 3 | 89.4 |
| Zhang et al. [35] | 2020 | 10 | No | AG-RL | 3 | 86.5 |
| Li et al.[36] | 2025 | 15 | No | AL-DANN | 3 | 89.8 |
| This Method | 2024 | 10 | Yes | DMM-WcycleGAN | 3 | 93.2 |


maintaining nearly identical recognition accuracy (93.21% vs. 93.15%),validating the effectiveness of the proposed dimen-sionality reduction strategy for resource optimization. These improvements translate directly into lower processing overhead and energy consumption, particularly beneficial in handling large-scale datasets. The lightweight architecture of DMM-WcycleGAN proves to be superior in computational efficiency without compromising accuracy, underscoring the value of the dimensionality reduction strategy in enhancing transfer learning frameworks for deep learning architecture optimization, espe-cially in resource-constrained environments.

## IV. DISCUSSION

We present a high-performance iBMI system through our innovative DMM-WcycleGAN architecture. This advanced framework achieves exceptional decoding capabilities by syner-gistically integrating meta-learning principles with an optimized CycleGAN architecture. The system demonstrates superior adapotability through its sophisticated dual-modality mapping mechanism, which enables rapid neural decoder optimization even with limited neural data inputs. Our framework's distinctive architecture not only minimizes computational overhead but also significantly enhances decoder performance through its optimized initialization model, making it particularly valuable for clinical applications where neural data acquisition is con-strained. A detailed comparative analysis against recent neural decoding studies conducted on the same dataset, as summarized in Table VII, highlights the superior performance of our frame-work across diverse neural recording configurations and ex-perimental paradigms. The implementation of dimensionality-reduced meta-learning significantly reduces computational com-plexity,as evidenced by the ablation study in Section III-D, establishing this approach as a viabole solution for clinical neural interface applications.

Despite the demonstrated efficacy in limited-data clinical sce-narios, as discussed in Section III-B, the confusion matrices in Fig.8 reveal two distinct neural decoding challenges that warrant further investigation. First, the S2D4 dataset demonstrates sig-nificantly lower accuracy compared to other datasets, reflecting a notable temporal degradation in neural decoding accuracy. This phenomenon aligns with the fundamental challenges of maintaining stable neural recordings in iBMI implants. Current cortical electrode arrays, including Utah arrays and FMA ar-rays, while designed to mnaintain stable neural recordings at the cortical surface, face inevitable biomechanical challenges over extended recording periods. Chronic neural recording sessions are susceptible to electrode impedance drift, tissue scarring

<!-- PCA of Dataset X and Dataset Y 25 Dataset X (category 1 and 3) Dataset Y (category 1 and 3) 20 Dataset X (category 2) Dataset Y (category 2) 15 ZueuoduoCIedlCuIAd 10 5 0 -5 -10 -10 -5 0 5 10 15 20 25 Principal Component 1 -->
![](https://web-api.textin.com/ocr_image/external/78a29d6ac353d12f.jpg)

Fig. 9. Distribution analysis of S2D3 and S2D4. Dataset X originates from S2D3, while Dataset Y originates from S2D4.

around implantation sites, and gradual signal attenuation due to foreign body responses. Microscopic electrode displacements due to tissue responses can induce subtle yet significant alter-ations in neural signal characteristics, potentially compromising recording quality and collectively contributing to reduced signal-to-noise ratios in later sessions. Second, although Category 2generally exhibits notably lower classification accuracy com-pared to other categories, an anomalous increase in its perfor-mance was observed in S2D4. This can be attributed to inherent dataset variability and stochastic sampling. As shown in Fig. 9, PCA analysis of the data distribution between S2D3 and S2D4reveals that the Category 2 clusters are much closer together in these two domains, while the clusters of other categories remain relatively more separated. This close clustering of Category 2reduces the domain transfer difficulty for this specific category, thus likely leading to the unexpected improvement in accuracy in S2D4. Notably, this feature of close clustering for Category 2is not present in other experimental datasets, further supporting that the observed improvement is a random fluctuation rather than a systematic trend.

This neural interface instability appears to bbe a primary factor in the observed decline in decoder performance,representing a significant challenge that affects overall system performance over time as deteriorating signal fidelity reduces the discrimi-native power of neural features. The current random sampling approach in daily neural data collection may also lead to im-balanced neural pattern representation. Although our approach partially addresses this through adaptive integration of historical neural recordings, the long-term decoding stability remains challenging. These findings suggest promising directions for improving neural recording protocols and decoder design: im-plementing a temporal weighting mechanism for historical data wherein more recent historical data would be assigned greater significance in the meta-training process, potentially attenuating the impact of gradual electrode displacement, developing adap-tive algorithms that compensate for temporal signal degradation, implementing quality control metrics to mitigate deteriorat-ing recording conditions, and optimizing neural data sampling strategies through stratified or balanced sampling methods. In addition, we plan to extend our method to broader and more standardized datasets, such as the Neural Latents Benchmark (NLB),which offers a diverse range of tasks and brain areas for evaluating latent variable models [37].

<!-- Authorized licensed use limited to: Beijing Jiaotong University. Downloaded on March 04,2026 at 03:48:01 UTC from IEEE Xplore. Restrictions apply. -->

<!-- CHEN et al.: INTRACORTICAL BRAIN-MACHINE INTERFACES WITH HIGH-PERFORMANCE NEURAL DECODING -->

<!-- 529 -->

## V. CONCLUSION

We present a high-performance iBMI system that achieves exceptional decoding capabilities through an innovative inte-gration of meta-learning and transfer learninng techniques. Our advanced framework revolutionizes iBMI decoder optimization by leveraging pre-existing neural datasets to construct a so-phisticated architectural foundation,enabling rapid and precise domain adaptation with minimal new data requirements. The system's novel meta-learning implementation delivers remark-able improvements in both computational efficiency and decod-ing accuracy, substantially outperforming traditional mmethod-ologies. Through its optimized calibration protocol and superior performance characteristics, our framework represents a signif-icant advancement in clinical iBMI technology, demonstrating unprecedented potential for therapeutic applications.

## REFERENCES

[1] M. A. Nicolelis and M. A. Lebedev, “Principles of neural ensemble physiology underlying the operation of brain-machine interfaces,"Nature Rev.Neurosci.,vol. 10,no.7,pp.530-540,2009.

[2] K. G. Oweiss and I.S. Badreldin, "Neuroplasticity subserving the opera-tion of brain-machine interfaces," Neurobiol.Dis.,vol. 83,pp.161-171, 2015.

[3] M. Naddaf,"Brain-reading devices allow paralysed people to talk using their thoughts," Nature, vol. 620,no.7976,pp.930-931,2023.

[4] M. J. Vansteensel et al., "Fully implanted brain-computer interface in a locked-in patient with ALS," New England J. Med., vol. 375,no.21, pp.2060-2066,2016.

[5] L.R. Hochberg et al., "Neuronal ensemble control of prosthetic devices by a human with tetraplegia," Nature, vol. 442,no.7099,pp.164-171,2006.

[6] V.Gilja et al., “Challenges and opportunities for next-generation intracor-tically based neural prostheses," IEEE Trans. Biomed. Eng.,vol. 58,no.7, pp.1891-1899,Jul.2011.

[7] S. N. Abdulkader, A. Atia, and M.-S. M. Mostafa, “Brain computer interfacing: Applications and challenges," Egyptian Informat. J.,vol. 16, no.2,pp.213-230,2015.

[8] P.Zhang et al.,"Feature-selection-based transfer learning for intracortical brain-machine interface decoding," IEEE Trans. Neural Syst. Rehabil. Eng.,vol. 29,pp.60-73,2020.

[9] U. Salahuddin and P.-X. Gao,“Signal generation, acquisition, and pro-cessing in brain machine interfaces: A unified review," Front.Neurosci., vol.15,2021,Art.no.728178.

[10] J.A.Perge et al., “Intra-day signal instabilities affect decoding perfor-mance in an intracortical neural interface system," J. Neural Eng., vol.10, no.3,2013,Art. no.036004.

[11] P.Agarwal and S.Kumar, "EEG-based imagined words classification using Hilbert transform and deep networks," Multimedia Tools Appl., vol. 83, no.1,pp.2725-2748,2024.

[12] W.Liang et al., "Variance characteristic preserving common spatial pat-tern for motor imagery BCI," Front. Hum. Neurosci.,vol.17,2023, Art.no.1243750.

[13] K.D.Ghanbar et al., "Correlation-based common spatial pattern (CCSP): A novel extension of CSP for classification of motor imagery signal," PLoS One,vol.16,no.3,2021,Art. no.e0248511.

[14] H.He and D.Wu,"Transfer learning for brain-computer interfaces: A Euclidean space data alignment approach," IEEE Trans. Biomed. Eng., vol.67, no.2, pp. 399-410,Feb. 2020.

[15] Z. Wang et al., "Stimulus-stimulus transfer based on time-frequency-joint representation in SSVEP-based BCIs," IEEE Trans.Biomed. Eng.,vol.70, no.2,pp.603-615,Feb.2023.

[16] D.Valencia, J. Thies, and A. Alimohammad, "Frameworks for efficient brain-computer interfacing," IEEE Trans. Biomed.Circuits Syst.,vol.13, no.6,pp.1714-1722,Dec.2019.

[17] P.-H.Chi et al.,“Audio ALBERT: A lite BERT for self-supervised learning of audio representation," in Proc. 2021 IEEE Spoken Lang. Technol. Workshop,2021,pp.344-350.

[18] F.Fahimi et al., "Generative adversarial networks-based data augmentation for brain-computer interface," IEEE Trans. Neural Netw. Learn. Syst., vol.32,no.9,pp.4039-4051,Sep.2021.

[19] I. Goodfellow et al., "Generative adversarial nets," in Proc. Adv.Neural Inf.Process.Syst., 2014,vol.27,pp.2672-2680.

[20] J.-Y. Zhu et al., “Unpaired image-to-image translation using cycle-consistent adversarial networks," in Proc. IEEE Int. Conf. Comput. Vis., 2017,pp.2223-2232.

[21] M.R. Mohebbian et al., "Fetal ECG extraction from maternal ECG using attention-based CycleGAN," IEEE J. Biomed.Health Informat.,vol.26, no.2,pp.515-526,Feb.2022.

[22] X. Li et al., "PTMA: Pre-trained model adaptation for transfer learning,”in Proc. Int. Conf. Knowl. Sci., Eng. Manage., 2024, pp. 176-188.

[23] V.Peterson et al., "Transfer learning based on optimal transport for motor imagery brain-computer interfaces," IEEE Trans. Biomed.Eng.,vol.69, no.2,pp.807-817,Feb.2022.

[24] Z. Zhang et al., "Joint optimization of CycleGAN and CNN classifier for detection and localization of retinal pathologies on color fundus pho-tographs," IEEE J. Biomed. Health Informat., vol. 26,no.1,pp.115-126, Jan.2022.

[25] B. Liu et al., "TransTailor: Pruning the pre-trained model for improved transfer learning," in Proc. AAAI Conf. Artif. Intell., 2021, vol.35,no. 10, pp.8627-8634.

[26] Q. Sun et al., "Meta-transfer learning through hard tasks," IEEE Trans. Pattern Anal. Mach. Intell., vol.44, no.3,pp.1443-1456,Mar.2022.

[27] C.Finn, P. Abbeel, and S. Levine, "Model-agnostic meta-learning for fast adaptation of deep networks," in Proc. Int. Conf. Mach. Learn., 2017, pp.1126-1135.

[28] V. Jayaram et al., "Transfer learning in brain-computer interfaces,” IEEE Comput.Intell.Mag., vol. 11, no. 1, pp. 20-31,Feb.2016.

[29] B.Wittevrongel et al., "Practical real-time MEG-based neural interfac-ing with optically pumped magnetometers," BMC Biol., vol. 19,no.1, pp.158-158,2021.

[30] K. Nguyen et al., “TinyMPC: Model-predictive control on resource-constrained microcontrollers," in Proc. 2024 IEEE Int. Conf. Robot. Au-tomat.,2024,pp.1-7.

[31] Y.Wang et al.,"Computation-efficient deep learning for computer vision: A survey," in Proc. Cybern. Intell., 2024,pp.1-24.

[32] J.Zhou et al., "Are all losses created equal: A neural collapse perspective,”in Proc. Adv. Neural Inf. Process. Syst., 2022, vol. 35,pp.31697-31710.

[33] J.Yik et al.,“The neurobench framework for benchmarking neuromorphic computing algorithms and systems," Nature. Commun.,vol. 16,2023, Art.no.1545.

[34] T.Wu et al., "Deep compressive autoencoder for action potential compres-sion in large-scale neural recording," J. Neural Eng., vol. 15,no.6,2018, Art.no.066019.

[35] P. Zhang et al., "Reinforcement learning based fast self-recalibrating decoder for intracortical brain-machine interface,"Sensors,vol.20,no.19, 2020,Art.no.5528.

[36] X.Li et al.,“Deep transfer learning-based decoder calibration for intra-cortical brain-machine interfaces," Comput. Biol. Med.,vol. 192,2025, Art.no.110231.

[37] F. Pei et al.,“Neural Latents benchmark'21: Evaluating latent variable models of neural population activity," 2021,arXiv:2109.04463.

<!-- Authorized licensed use **limited** to: Beijing Jiaotong University. Downloaded on March 04,2026 at 03:48:01 UTC from IEEE Xplore. Restrictions apply. -->

