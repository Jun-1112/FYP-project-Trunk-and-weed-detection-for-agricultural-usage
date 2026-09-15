# FYP-project-Trunk-and-weed-detection-for-agricultural-usage

This project focuses on comparing object detection and video segmentation for precision weed control in oil palm plantations.

A final year project investigating whether pixel-level segmentation offers a practical advantage over bounding-box detection when identifying palm tree trunks and weeds from ground-level plantation video with the goal of informing the design of automated, precision herbicide application systems.


# Problem
Conventional weed control in oil palm plantations relies on blanket herbicide spraying, which wastes chemicals, degrades soil, and risks damaging the crop. A precision spraying system needs to reliably distinguish **weeds** (spray targets) from **palm tree trunks** (avoid targets) in real time from a moving vehicle.

This raises a design question that had no clear answer in existing literature: for this specific task, is bounding-box detection sufficient, or does the added computational cost of segmentation earn its keep?

Four gaps this project addresses:
| 1 | No model specifically trained to distinguish trunks from weeds under real plantation conditions (canopy shadow, occlusion, cluttered ground cover) |
| 2 | No comparative benchmark of detection vs. segmentation for this task |
| 3 | Unknown robustness under real handheld-video conditions (motion blur, lighting variation) |
| 4 | Unquantified risk of relying on externally-sourced training data |

# Approach
Three models were trained and evaluated:

|     Model           | Architecture      |     Class       | Annotation Type |
|   Trunk Detection   | YOLOv8n           | Palm Tree Trunk | Bounding box    |
| Trunk Segmentation  | YOLOv8n-seg       | Palm Tree Trunk | Polygon mask    |
| Weed Detection      | YOLOv8n           | 15 weed species | Bounding box    |

To ensure consistency and the results obtained are from comparing between the performance of detection and segmentation, the same YOLO architecture family was chosen to train for all models where the object detection is trained using YOLOv8n and the segmentation model is trained using YOLOv8n-seg, so any observed difference is attributable to the detection-vs-segmentation task itself rather than unrelated architectural differences. Their ground truth is derived from the same underlying polygon annotations where bounding boxes are computed from each polygon's bounding rectangle so both models can see the same annotated instances, differing only in representation.


#Data
- Video footage: GoPro handheld, 1920×1080 (30 FPS)
- Raw duration: 12,450 frames
- External weed dataset: MH-Weed16 (Shinde & Attar, 2024)


#Tools
- Google Colab, NVIDIA Tesla T4 GPU
- Ultralytics YOLOv8, PyTorch


# Data Pipeline

The most involved part of this project was producing usable ground truth from raw plantation footage.

```
Raw GoPro video (12,450 frames)
        │
        ├─ Frame subsampling (15-frame interval, CVAT) ──> 831 frames
        │
        ├─ Manual polygon annotation of 50 seed frames
        │
        ├─ Train throwaway YOLOv8n-seg "mini-model" on those 50 frames
        │
        ├─ Mini-model auto-generates draft masks for remaining 781 frames
        │
        ├─ Manual review + correction of all draft masks in CVAT
        │
        └─> Consolidated 831-frame trunk dataset (box + mask ground truth)
```

This is a model-assisted, human-in-the-loop annotation workflow. The mini seg model was disposable built solely to accelerate labelling and annotating. It is discarded afterward and was never part of the final product. Every annotation used for training passed through human verification and refinement after trained.

CVAT is an annotation tool that was self-hosted locally via Docker Compose, which orchestrates its multi-service stack (web server, database, message queue, background workers) in isolated containers.


# Annotation scope: actionable range, by design

1. Annotation was deliberately scoped to tree trunks within the vehicle's actionable range which is the near-field trunks that are clearly and distinctly resolvable in frame rather than every trunk visible down the plantation row, including distant background trees.

2. This is a design decision, not incomplete labelling. Under the intended deployment context stated above, a moving ground vehicle making real-time spray decisions:

3. The trunks relevant to a spray decision at any given moment are the ones the vehicle is about to pass. Distant trees several rows back are not actionable as no spray decision applies to them yet.

4. As the vehicle advances, those distant trees become the near-field trunks the system acts on. Every tree is eventually annotated at the range where it matters; it simply isn't labelled while still far away.

5. Labelling distant trunks would mean annotating targets that are only a few pixels wide, heavily occluded by intervening fronds and nearer trees, and at a scale where boundary precision is unreliable. Including them would inject noisy ground truth into the training set rather than improving it.

6. Evaluating at the actionable range therefore reflects the system's real operating condition, not a favourable subset chosen to inflate scores. The trade-off is stated openly: this project does not claim performance on distant, small-scale trunk detection, and the corresponding weakness is documented in Limitations below.


# Weed dataset

1. Weeds are visible in the project's own footage, but not at a scale or separation that supports reliable annotation. The camera captures the plantation scene at working distance, not close-up: weeds appear as low-lying, densely interlocking ground cover whose boundaries are visually continuous with surrounding vegetation, fallen fronds, and the base of the trunks themselves.

2. The problem is not that annotation is difficult — it is that annotating under these conditions would actively damage the model. Any polygon drawn around a weed patch would inevitably enclose adjacent fronds, leaf litter, and trunk base within the same boundary. Training on that ground truth teaches the model that fronds and leaves are weeds, producing a detector that fires on palm foliage and trunk bases in which precisely the objects the system exists to avoid. The bad ground truth here is worse than no ground truth.

3. This is why trunk annotation succeeds where weed annotation does not, despite using the same footage. A palm trunk is a large, vertical, visually distinct structure with a clear silhouette against the background; its boundary can be traced unambiguously at working distance. A weed patch at the same distance has no such separable boundary.

4. The footage additionally contains a wide range of weed species, which would require consistent species-level discrimination that neither the image quality nor the annotator's domain knowledge could reliably support.

5. An external, publicly available bounding-box-annotated weed dataset (Intel RealSense Depth) was used instead. This introduces a domain gap where the dataset originates from a different region with different weed species in which it was flagged as a risk before training and then explicitly measured (see RQ4).


# Results

Training environment: Google Colab, NVIDIA Tesla T4 GPU, Ultralytics YOLOv8.

# Standard metrics

| Model           | Method       | Precision | Recall    | mAP50     |
| Weed            | Bounding Box | 69.2%     | 58.7%     | 62.8%     |
| Trunk (Detect)  | Bounding Box | 90.7%     | 93.2%     | 96.2%     |
| Trunk (Segment) | Polygon Mask | 94.4%     | 87.0%     | 95.3%     |

Neither trunk model dominates. Detection has higher recall (misses fewer trunks); segmentation has higher precision (fewer false positives). This precision/recall trade-off is why a localisation-specific comparison was needed.

RQ1 — Localisation precision
| Metric                              |   Value   |
| Detection Box IoU (vs. GT box)      | 0.8901    |
| Segmentation Mask IoU (vs. GT mask) | 0.8048    |
| Segmentation Box IoU (vs. GT box)   | 0.8585    |
| Extraneous Area Ratio               |   21.8%   |

Detection scores higher on raw box IoU but that metric structurally favours boxes, since mask IoU demands pixel-level agreement across an irregular boundary while box IoU only needs rectangle overlap.

The extraneous area ratio is the metric that answers the actual question. It measures how much of a detection bounding box is *not* trunk, relative to the segmentation mask's tighter boundary:

extraneous_area_ratio = (detection_box_area − segmentation_mask_area) / detection_box_area

At 21.8%, roughly a fifth of a box-shaped spray zone would cover non-trunk area. Segmentation's contour-following boundary meaningfully reduces unnecessary herbicide exposure — the property that matters for this application, even where raw IoU doesn't favour it.

*This metric was designed for this project; it is not drawn from existing literature.*

# RQ2 — Computational cost

| Model | Mean Latency (ms) | P95 (ms) | Mean FPS | Actual FPS |
|---|---|---|---|---|
| Weed Detection | 14.24 | 24.74 | 70.21 | 41.52 |
| Trunk Detection | 15.15 | 26.95 | 66.01 | 40.21 |
| Trunk Segmentation | 19.78 | 35.01 | 50.56 | 35.33 |

Segmentation costs +30.6% mean latency and −23.4% mean FPS versus detection. It remains real-time viable: 35.33 FPS end-to-end, above the conventional 24–30 FPS threshold.

A secondary finding: the gap between "Mean FPS" (model compute only) and "Actual FPS" (full pipeline) stays at roughly a constant ~9ms across all three models, even as model compute time varies from 14.24ms to 19.78ms. Since that overhead doesn't scale with model complexity, it isn't coming from the model. The video I/O is the larger bottleneck, which is relevant to any future optimisation effort.

# RQ3 — Robustness

Frames were stratified into terciles by motion blur (variance of Laplacian) and brightness (mean grayscale intensity), then IoU compared across groups.

| Motion Blur Group   | Detection IoU | Segmentation IoU |
| High Blur           | 0.8893        | 0.8022           |
| Medium Blur         | 0.8944        | 0.8222           |
| Low Blur            | 0.8938        | 0.8189           |

| Lighting Group    | Detection IoU | Segmentation IoU |
| Low Brightness    | 0.8909        | 0.8210           |
| Medium Brightness | 0.8936        | 0.8176           |
| High Brightness   | 0.8931        | 0.8048           |

Detection stays essentially flat (<0.6 percentage point range). Segmentation varies slightly more (~2pp), consistent with pixel-level boundaries being more sensitive to edge clarity than box overlap.

Important scoping note: These groups are a relative tercile split within a single continuous daylight video. It is not a comparison across genuinely distinct lighting conditions. "High Brightness" means the brightest third of frames in this clip, not bright daylight in absolute terms.

# RQ4 — Domain gap

| Metric                | External Validation (in-domain) | Plantation Video (out-of-domain) |
| Frames/images         | 743                             | 461                              |
| Total detections      | 8,215                           | 1,408                            |
| Mean confidence       | 0.5507                          | 0.3978                           |
| Zero-detection frames | 0.4%                            | 10.0%                            |

No ground-truth weed labels exist for the plantation video, so this compares model *behaviour* rather than accuracy: **mean confidence dropped 27.8%**, and **frames with zero detections rose 25-fold**. These two indicators are relatively unconfounded by scene composition and together support the domain gap hypothesis.

The detections-per-frame drop (72.4%) is reported with more caution — the external dataset's curated close-ups naturally contain more weed instances per image than wide plantation shots, so that figure partly reflects scene composition rather than recognition failure alone.


# Limitations
It is stated plainly because they affect how these results should be read.

1. Weed footage was unusable for annotation. The project's own video lacks close-up weed shots, and contains many weed species. At the capture distance and resolution, weed boundaries are not visually separable from surrounding vegetation. So an external dataset was used, introducing the domain gap measured in RQ4.

2. IoU and robustness were evaluated on training-set frames. The originally-planned held-out validation annotations were lost when an ephemeral Colab session reset. These results are therefore an upper-bound estimate of localisation precision, not confirmed generalisation performance. The relative comparison between models should survive this bias as both were evaluated identically but the absolute values likely overstate real-world performance.

3. The domain gap uses indirect indicators. Confidence and detection-rate comparison, not a direct accuracy measurement, because no weed ground truth exists for the target video.

4. The weed model is a 15-class species detector. The external dataset labels weed species individually where it distinguish 15 visually similar plant types. This is substantially harder than binary weed-vs-not-weed detection, so part of the weed model's lower scores reflects task difficulty in addition to domain gap.

5. Detection degrades with distance. Consistent with the annotation scope described above, the models are trained and evaluated for near-field, actionable-range trunks. Distant trunks in the background of a frame are frequently not detected. This is a combination of known small-object limitations in single-stage detectors (objects shrink to a few pixels after backbone downsampling) with increasing occlusion from intervening fronds and nearer trees, and confidence scores falling below threshold. This is expected given the deployment scope, but it does mean the reported metrics should not be read as performance on full-depth trunk detection across an entire plantation row.

6. No hyperparameter tuning was performed. Training relied on Ultralytics defaults with built-in early stopping (patience) and automatic best-checkpoint retention. These are default safeguards, not deliberate optimisation.

7. No data augmentation was applied. Models saw only the natural variation present in the source footage.

8. Lighting conditions are narrow. All footage was captured in one continuous daylight session. Performance under overcast, dawn/dusk, or high-glare conditions is untested.

9. Latency was benchmarked on a cloud GPU. A Tesla T4 is not representative of the edge hardware a deployed ground vehicle would use. FPS figures are a relative comparison, not an absolute deployment prediction.

10. No live field deployment testing. Evaluation is entirely offline on pre-recorded video.

11. No physical platform was built or tested. The ground vehicle described as the deployment context is the intended application, not something implemented in this project. All footage is handheld, used as a proxy for a vehicle-mounted camera. Camera height, angle, stabilisation, and forward speed on an actual vehicle would all differ from handheld capture, and could affect performance in ways this evaluation does not measure. Design decisions are justified by the ground vehicle scenario where the real-time FPS threshold and near-field annotation scope rest on this stated assumption rather than on measured platform characteristics.


# Future work
1. Re-establish a genuine held-out validation set and re-run IoU/robustness evaluation to convert upper-bound estimates into confirmed results
2. Annotate real in-domain weed samples to enable direct domain-gap measurement and in-domain fine-tuning
3. Collapse the 15 weed species into a single class and retrain, to separate task difficulty from domain shift
4. Systematic hyperparameter optimisation
5. Collect footage across a wider range of lighting conditions
6. Apply data augmentation during training
7. Extend detection to distant/small-scale trunks via higher input resolution or tiled inference, if full-depth row detection becomes a requirement
8. Benchmark on target edge hardware and conduct live field testing


# References
Architecture and methods
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. *CVPR*, 779–788.
- He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. *ICCV*, 2961–2969.
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How Transferable Are Features in Deep Neural Networks? *NeurIPS*, 27, 3320–3328.
- Prechelt, L. (1998). Early Stopping — But When? In *Neural Networks: Tricks of the Trade*, LNCS 1524, 55–69. Springer.
- Settles, B. (2009). *Active Learning Literature Survey*. Tech. Report 1648, University of Wisconsin–Madison.

Evaluation metrics
- Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The PASCAL Visual Object Classes (VOC) Challenge. *IJCV*, 88(2), 303–338.
- Lin, T.-Y., et al. (2014). Microsoft COCO: Common Objects in Context. *ECCV*, 740–755.
- Pech-Pacheco, J. L., Cristóbal, G., Chamorro-Martínez, J., & Fernández-Valdivia, J. (2000). Diatom Autofocusing in Brightfield Microscopy: A Comparative Study. *ICPR-2000*, Vol. 3, 314–317.
- Torralba, A., & Efros, A. A. (2011). Unbiased Look at Dataset Bias. *CVPR*, 1521–1528.

Software
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO* (v8.0.0) [Software].
- Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*, 32, 8024–8035.
- Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*, 25(11), 120–125.
- Harris, C. R., et al. (2020). Array Programming with NumPy. *Nature*, 585(7825), 357–362.
- McKinney, W. (2010). Data Structures for Statistical Computing in Python. *SciPy 2010*, 56–61.
- CVAT.ai Corporation (2023). *Computer Vision Annotation Tool (CVAT)* [Software].
- Merkel, D. (2014). Docker: Lightweight Linux Containers for Consistent Development and Deployment. *Linux Journal*, 2014(239).


# Acknowledgements

Dataset: Intel RealSense Depth weed dataset (external, publicly available) for weed detection training. Plantation video footage captured on-site for trunk detection and segmentation.
Shinde, S., & Attar, V. (2024). MH-Weed16: An Indian Multiclass Annotated Weed Dataset for Computer Vision Tasks (Version 2) [Data set]. Mendeley Data. https://doi.org/10.17632/d3n3mgjjbv.2

Accompanying paper:
Shinde, S., & Attar, V. (2025). An Indian annotated weed dataset for computer vision tasks in precision farming. Data in Brief. https://doi.org/10.1016/j.dib.2025.111766
