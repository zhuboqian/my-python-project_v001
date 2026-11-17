# 工程代码用途与清理建议

本清单只给出归档建议；除明确的缓存或误生成空文件外，不物理删除历史实验代码。

## 历史训练实验

| 文件 | 用途 | 建议 |
|---|---|---|
| `yolo_related/finetune_all_3domain.py` | Fine-tune all 6 models (A-F) on expanded 3-domain dataset (1149 images, 30 epochs). | 建议归档，不直接删除 |
| `yolo_related/resume_controlled_3domain_b_to_f.py` | Resume the controlled three-domain A-F experiment from Model B. Model A is already complete. Model B resumes from its existing last.pt, then Models C-F train sequentially with the same settings as the single-domain v2 runs. Existing complet | 建议归档，不直接删除 |
| `yolo_related/resume_model_g_checked.py` | Resume G only when checkpoint contains optimizer and epoch state. | 建议归档，不直接删除 |
| `yolo_related/resume_train_3domain.py` | Resume 3-domain training from Model D (A/B/C already done). | 建议归档，不直接删除 |
| `yolo_related/train_all_3domain_controlled.py` | Train controlled A-F models on the current Combined three-domain split. All optimization and augmentation settings intentionally match the single-domain *_v2 runs. The only intended variables are architecture (A-F) and training data (Origin | 建议归档，不直接删除 |
| `yolo_related/train_all_3domain_controlled_v2.py` | Strict controlled three-domain A-F training, revision v2. All six models start fresh and use one explicit, identical training recipe. The v2 suffix separates these runs from the earlier warmup-inconsistent v1 runs. | 建议归档，不直接删除 |
| `yolo_related/train_all_models_3domain.py` | Train all architecture variants (A, B, C, E) on the SAME 3-domain dataset. Controlled comparison — all models share identical training data + augmentations. Only architecture differs. Order: A → B → C → E (F already done as target_detector_ | 建议归档，不直接删除 |
| `yolo_related/train_all_models_3domain_v2.py` | 3-domain 训练脚本（新方案 NEW scheme） 在 923 张合并数据集上训练所有 6 个模型 A-F。 新方案模型定义： A: Baseline (yolov8_orignial.yaml) B: +Preprocessing (yolov8_orignial.yaml + 强数据增强) C: +SPDConv (yolov8_SPDConv.yaml) D: +SPDConv+Preprocessing (yolov8_SPDConv.yaml + 强数据增强 | 建议归档，不直接删除 |
| `yolo_related/train_all_single.py` | Train A/B/C/E/F on single-domain (original 348 imgs). Original settings. | 建议归档，不直接删除 |
| `yolo_related/train_model_a.py` | 代码中无模块说明，需结合调用关系复核 | 建议归档，不直接删除 |
| `yolo_related/train_model_b.py` | 训练 Model B：YOLOv8n + 图像预处理（光照归一化增强训练） 与 Model A 使用相同网络结构，但训练时启用更强的数据增强， 使模型对光照变化和杂波干扰更具鲁棒性。 使用 yolov8_orignial.yaml | 建议归档，不直接删除 |
| `yolo_related/train_model_c.py` | 训练 Model C：YOLOv8n + SPDConv (仅替换Backbone前两层) 使用 yolov8_SPDConv.yaml | 建议归档，不直接删除 |
| `yolo_related/train_model_d.py` | 训练 Model D：YOLOv8n + SPDConv + 图像预处理 使用 yolov8_SPDConv.yaml，训练时启用更强的数据增强， 使 SPDConv 模型对光照变化和杂波干扰更具鲁棒性。 | 建议归档，不直接删除 |
| `yolo_related/train_model_d_c2fattn.py` | Train Model D: YOLOv8n + C2fAttn only (3-domain dataset, controlled comparison) | 建议归档，不直接删除 |
| `yolo_related/train_model_d_single.py` | Train controlled Model D v2 on the current single-domain Original split. | 建议归档，不直接删除 |
| `yolo_related/train_model_e.py` | 训练 Model E：YOLOv8n + SPDConv + P2 (四尺度检测头) 使用 yolov8_spdconv_p2.yaml 使用Model C的权重作为起点（迁移SPDConv层的学习成果） | 建议归档，不直接删除 |
| `yolo_related/train_model_f.py` | 训练 Model F：YOLOv8n + SPDConv + P2 + C2fAttn (完整改进模型) 使用 yolov8_designed.yaml | 建议归档，不直接删除 |
| `yolo_related/train_model_f_augmented.py` | Train Model F (augmented): YOLOv8n + SPDConv + P2 + C2fAttn With enhanced data augmentation for cross-domain robustness. Key improvements over baseline training: - Combined dataset: original + tension + degraded_physics (923 train / 161 val | 建议归档，不直接删除 |
| `yolo_related/train_model_f_dynamic_hard_v2.py` | Fine-tune current best Model F with hard dynamic samples and replay. | 建议归档，不直接删除 |
| `yolo_related/train_model_f_dynamic_v1.py` | Fine-tune the current Model F with dynamic-scene replay data. | 建议归档，不直接删除 |
| `yolo_related/train_model_f_enhanced_aug.py` | Train Model F on original dataset only (348 images) with enhanced augmentations. CONTROLLED comparison: same training data as A-F, enhanced aug only. | 建议归档，不直接删除 |
| `yolo_related/train_model_f_illumination_aug_v1.py` | Fine-tune Model F with training-only illumination randomization. | 建议归档，不直接删除 |
| `yolo_related/train_model_f_lowlight_real_0828_v1.py` | Fine-tune Model F using real low-light images and old-data replay. | 建议归档，不直接删除 |
| `yolo_related/train_model_f_original.py` | Train Model F on original dataset ONLY (348 images). Moderate augmentations that converge well on small datasets. Same training data as models A-F, for controlled comparison. | 建议归档，不直接删除 |
| `yolo_related/train_model_f_real_phone_v1.py` | Incrementally fine-tune Model F with legacy replay and real-phone data. | 建议归档，不直接删除 |
| `yolo_related/train_model_f_single.py` | Retrain Model F on single-domain (original 348 imgs). No MixUp, standard aug. | 建议归档，不直接删除 |
| `yolo_related/train_model_f_synthetic_lowlight_100_v1.py` | Fine-tune controlled three-domain Model F with 100 synthetic dark images. | 建议归档，不直接删除 |
| `yolo_related/train_model_g.py` | Model G: SPDConv + P2 + SimAM (无参数3D注意力) SimAM: 零参数零FLOPs，纯即插即用 | 建议归档，不直接删除 |
| `yolo_related/train_model_g_darm.py` | Train Model G: F architecture + DARM (original multi-statistic attention) | 建议归档，不直接删除 |
| `yolo_related/train_model_g_p2_c2fattn_controlled_v1.py` | Controlled three-domain training for Model G (P2 + C2fAttn, no SPDConv). | 建议归档，不直接删除 |
| `yolo_related/train_model_h_gdarm.py` | Train Model H: F architecture + GDARM (Gated DARM) | 建议归档，不直接删除 |
| `yolo_related/train_model_h_nwd.py` | Model H: Model F architecture + NWD Loss for small objects. Same 3-domain dataset + augmentations as A/B/C/E/F (3-domain). Isolates the effect of NWD loss function. NWD (Normalized Wasserstein Distance): - Models boxes as 2D Gaussians - For | 建议归档，不直接删除 |
| `yolo_related/train_model_h_single_then_3domain.py` | Train Model H in two preserved stages: Original-only, then three-domain. Stage 2 starts from Stage 1 best.pt but writes to a separate run directory. Partial runs resume from their own last.pt when optimizer state is available. | 建议归档，不直接删除 |
| `yolo_related/train_model_i_carafe.py` | Model I: Model F architecture + CARAFE upsampling in Neck. Same 3-domain dataset + augmentations. Isolates CARAFE effect. | 建议归档，不直接删除 |

## 实验评价与分析

| 文件 | 用途 | 建议 |
|---|---|---|
| `yolo_related/analyze_dynamic_candidate_confidence.py` | Analyse Model F confidence distribution on extracted dynamic left frames. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/analyze_overexposure_failures.py` | Per-image diagnosis for the 14-image real overexposure frozen test set. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/analyze_static_stereo_bias.py` | Exclude a documented outlier and quantify signed stereo depth bias. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/benchmark_cuda_remap_library.py` | Benchmark the selected CUDA_REMAP_LIBRARY and save one output for comparison. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/benchmark_end_to_end_roi_ncc_ad.py` | Fair stage-wise latency benchmark for current Models A/D + ROI-NCC. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/benchmark_fps.py` | Micro-benchmark: 逐阶段测量每帧处理时间 | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/benchmark_full_frame_sgbm_static.py` | Fair static-pair benchmark of full-frame SGBM versus ROI-SGBM. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/benchmark_six_backbones_gpu.py` | Uniform raw-forward GPU benchmark for six lightweight YOLO backbones. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/benchmark_speed_map.py` | Benchmark speed + mAP for all 6 3-domain models. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/benchmark_trt_end_to_end_roi_ncc_ad.py` | Pure TensorRT FP16 end-to-end benchmark for final Models A/D + ROI-NCC. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/benchmark_trt_gpu_preprocess.py` | 代码中无模块说明，需结合调用关系复核 | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/benchmark_vibration_trt_pipeline.py` | Real 1,800-frame vibration-video throughput: optimized rectify + D TRT + ROI-NCC. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/diagnose_bd_scale_behavior.py` | Per-image diagnosis of B/D scale-stratified behavior at 640 and 960. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/diagnose_new_b_vibration.py` | Read-only model diagnosis; transformations are diagnostic, not final tests. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_acfg_unified5_frozen.py` | Evaluate A/C/F/G on the unified five-domain frozen detection test set. The existing A/C/F results for the original three frozen domains are reused from the controlled 2026-09-08 evaluation. Model G is evaluated on all five domains, while A/ | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_adaptive_lowlight_enhancement.py` | Zero-training adaptive low-light enhancement ablation on frozen dark images. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_all.py` | 统一评估脚本：对所有模型 × 所有数据集运行检测并评估 | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_augmented_model.py` | Evaluate Model F Augmented on all 3 test datasets using PyTorch. Saves detection results in the expected format for compute_all_metrics.py. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_continuous_stereo_matching.py` | Evaluate frozen stereo matchers on +/-5-frame windows around truth frames. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_controlled_ablation_3domains.py` | Evaluate controlled-v2 Models A-F on the three frozen 46-image test domains. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_controlled_scale_bd.py` | Evaluate Models B and D on the paired controlled-scale diagnostic set. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_dark_matching_oracle_bbox.py` | Diagnostic dark-image matching with manually labelled left target boxes. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_dark_stereo_20260827.py` | Evaluate detection and stereo depth on the frozen 2026-08-27 dark subset. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_detection_dataset.py` | Evaluate a registered custom Ultralytics detector and persist metrics as JSON. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_equalized_3domain_acfh.py` | Evaluate equal-sized three-domain Models A/C/F/H on common frozen tests. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_final_bd_independent_v2.py` | Evaluate final Models B and D on leakage-free frozen tests. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_final_bd_real_illumination.py` | Evaluate final Models B/D on real dark and real overexposure datasets. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_final_bd_small_target.py` | Evaluate final B/D on real labeled distance images, stratified by target scale. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_illumination_aug_v1.py` | Compare old and illumination-augmented Model F on the frozen dark set. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_illumination_g.py` | Fixed-weight, fixed-parameter F/G illumination comparison; no test-set tuning. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_labeled_video_frames_abcd.py` | Diagnostic A-D evaluation on historically annotated video frames. These frames were used by earlier dynamic-scene adaptation work, so the output is an internal diagnostic rather than an independent generalization result. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_lowlight_real_0828_v1.py` | Evaluate the current Model F on the frozen real low-light test split. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_model_f_dynamic_test.py` | Compare pre/post dynamic fine-tuning Model F on the frozen dynamic test block. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_model_f_synthetic_lowlight_100_v1.py` | Evaluate baseline/new Model F on frozen real-dark and three-domain tests. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_model_h_single_vs_3domain.py` | Evaluate Model H single-domain and three-domain weights on frozen tests. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_multivideo_abcd_repeats.py` | Evaluate final A-D candidate-output continuity on independently recorded clips. The extracted clip folders were manually selected from three stereo recordings. They do not provide exhaustive frame-wise ground-truth boxes, so this program re | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_new_data0826_detection_scale.py` | Evaluate Models A/F on the frozen, manually labelled new_data0826 left images. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_new_distance_20261012_13.py` | Prepare and evaluate the 46 labeled stereo pairs captured on Oct 12-13. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_overexposure_v2_acfg.py` | Evaluate controlled A/C/F/G weights on the 34-image overexposure v2 set. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_saved_final_D_boxes_with_ncc.py` | Evaluate new ROI-NCC continuity using audited final-D PyTorch detections. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_single_abcd_frozen_degraded_20261017.py` | Evaluate final single-domain A-D weights on the two frozen degraded tests. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_single_abcd_new_data0826.py` | Evaluate final single-domain A-D weights on labelled new_data0826 left images. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_single_for_domain_comparison.py` | Re-evaluate existing single-domain weights using the controlled evaluation protocol. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_six_backbones_frozen_test.py` | Evaluate six lightweight YOLO backbones on one frozen test split. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_static_stereo_truth.py` | Evaluate frozen Model F stereo pipelines on the current static pairs. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_stereo_distance_truth.py` | Evaluate Model F + three stereo matchers on measured-distance frames. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_stratified.py` | 按距离分段评估 ================ 对每个模型在每个数据集上按目标距离进行分层评估: 近距: 0.0 - 1.0m 中距: 1.0 - 2.0m 远距: 2.0 - 5.0m (样本较少) 输出: - 终端表格 - JSON (stratified_metrics.json) - 分段柱状图 (mAP / MAE / 成功率) 用法: python evaluate_stratified.py | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_temporal_association.py` | Compare frame-independent detection with lightweight temporal association. Each extracted clip is treated as an independent sequence. Stereo depth uses the same ROI-NCC implementation in both branches, so the only changed component is the l | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_vibration_video_detection.py` | Evaluate Model A/F detection stability on vibration-table stereo videos. The target is known to be unique and approximately stationary in the image. A fixed spatial gate prevents unrelated low-confidence background detections from being cou | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/evaluate_vibration_video_stereo.py` | Evaluate stereo matching stability on the vibration-table video windows. This experiment reuses the frozen Model F left-eye detections produced by evaluate_vibration_video_detection.py. Matching searches the rectified right image directly a | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/plot_controlled_scale_bd.py` | Plot the controlled target-scale response of Models B and D. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/plot_final_abcd_vibration_1800.py` | Paper-ready continuous-frame comparison for final independent_v2 A-D. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/plot_final_abcd_vibration_status_old_style.py` | Plot only panel (a) for final A-D, matching the historical figure style. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/plot_multivideo_abcd_repeats.py` | Plot repeated-video A-D diagnostics with clearly separated metric scopes. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/plot_training_curves.py` | 从 results.csv 生成 YOLO 风格的训练曲线图 | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/plot_vibration_1800_all_models.py` | Plot the completed A/C/F/G/H vibration-table detection+ROI-SGBM study. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/plot_vibration_1800_preview.py` | Create an honest layout preview from currently available 1800-frame data. | 按论文结果依赖保留；旧版可归档 |
| `yolo_related/preview_new_b_vibration.py` | Export all predicted boxes for representative vibration frames (not GT). | 按论文结果依赖保留；旧版可归档 |

## 当前多目标评估

| 文件 | 用途 | 建议 |
|---|---|---|
| `yolo_related/evaluate_multi_target_detection.py` | Evaluate final Model D on the frozen real multi-target stereo capture. | 保留 |
| `yolo_related/sweep_multi_target_inference.py` | Sweep inference settings for final Model D on the frozen multi-target set. | 保留 |

## 当前核心检测定位

| 文件 | 用途 | 建议 |
|---|---|---|
| `yolo_related/compare_dynamic_stereo_matching.py` | Compare four stereo matching/depth pipelines on extracted dynamic frames. P1: Model F left/right detections + bounding-box center disparity P2: Model F left detection + ROI-SGBM P3: Model F left detection + ROI-NCC template matching P4: Mod | 保留 |
| `yolo_related/estimate_target_xyz_left_camera.py` | Estimate target XYZ in the rectified left-camera coordinate system. Pipeline: stereo rectification -> Model D detection on rectified left image -> ROI-NCC disparity -> Z = fx*B/d -> pinhole back-projection for X and Y. | 保留 |
| `yolo_related/model_f_pipeline.py` | Model F 完整优化管线 ===================== Model F = Model E (SPDConv + P2) + C2fAttn + 速度优化 速度优化策略 (针对 Jetson Orin, 目标 30 FPS): 1. TensorRT FP16 引擎导出 → 2-3x 加速 2. GPU 立体校正 (grid_sample) → 消除 CPU remap 瓶颈 3. 批量推理 (batch=2 左右目一次前向) → 1.3x 加速 4. 降低 | 保留 |
| `yolo_related/multi_target_tracker.py` | Multi-target YOLO detection with lightweight temporal association. The detector returns every target box. Tracks use a constant-velocity Kalman filter and Hungarian assignment. A short detector miss is retained as a *prediction*, never misl | 保留 |
| `yolo_related/my_modules.py` | 代码中无模块说明，需结合调用关系复核 | 保留 |
| `yolo_related/realtime_positioning.py` | 实时双目靶标定位管线 — 面向30 FPS优化 (Jetson Orin) 综合优化策略: 1. GPU立体校正 (grid_sample) — 消除CPU remap瓶颈 2. 批量推理 (batch=2, 左右目一次前向) — 复用GPU kernel launch 3. FP16半精度 — Jetson Orin FP16吞吐翻倍 4. 降低推理分辨率 (640×384) — 保持精度同时提速 5. TensorRT INT8引擎 — 进一步2-3x加速 6. 流水线化 | 保留 |
| `yolo_related/register_spdconv.py` | SPDConv注册模块：将SPDConv注入ultralytics，使其可在YAML中被识别 | 保留 |

## 最终训练与数据审计

| 文件 | 用途 | 建议 |
|---|---|---|
| `yolo_related/build_independent_3domain_split_v2.py` | Audit and build a leakage-free three-domain split without modifying legacy data. Default mode is audit-only. Pass --materialize to create symlinks in OUTPUT. Grouping deliberately ignores domain and camera side, so all derivatives of one ph | 保留 |
| `yolo_related/finalize_independent_dataset_suite_v2.py` | Materialize and audit the final leakage-free dataset suite. | 保留 |
| `yolo_related/train_final_abcd_independent_v2.py` | Controlled final A/B/C/D training on leakage-free three-domain v2 data. | 保留 |
| `yolo_related/train_final_abcd_single_independent_v2.py` | Controlled final A-D single-Original-domain training on independent_v2. | 保留 |

## 第三方YOLOv7代码

| 文件 | 用途 | 建议 |
|---|---|---|
| `yolo_related/yolov7_official/deploy/triton-inference-server/boundingbox.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/deploy/triton-inference-server/client.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/deploy/triton-inference-server/labels.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/deploy/triton-inference-server/processing.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/deploy/triton-inference-server/render.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/detect.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/export.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/hubconf.py` | PyTorch Hub models Usage: import torch model = torch.hub.load('repo', 'model') | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/models/__init__.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/models/common.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/models/experimental.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/models/yolo.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/test.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/train.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/train_aux.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/__init__.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/activations.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/add_nms.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/autoanchor.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/aws/__init__.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/aws/resume.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/checkpoint.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/datasets.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/general.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/google_utils.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/loss.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/metrics.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/plots.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/torch_utils.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/wandb_logging/__init__.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/wandb_logging/log_dataset.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |
| `yolo_related/yolov7_official/utils/wandb_logging/wandb_utils.py` | 代码中无模块说明，需结合调用关系复核 | 独立归档；基础型号复现仍需保留 |

## 辅助/待复核

| 文件 | 用途 | 建议 |
|---|---|---|
| `AerialRecoveryDegradation.py` | 空中回收场景数据集退化处理 - 去除湍流，适配新路径 无气动光学湍流，保留：运动模糊、光照变化、背景杂波、传感器噪声，均可独立开关。 | 人工复核后再决定 |
| `ORBdetect.py` | ORB (Oriented FAST and Rotated BRIEF) 目标识别算法 - 实用版 移除过于严格的过滤，确保能够产生检测结果。 包含：CLAHE预处理、多尺度检测、加权NMS融合、完整评估指标 已验证集路径适配：val/images 和 val/labels | 人工复核后再决定 |
| `annotate.py` | 轻量标注工具 — 仿 labelImg 操作逻辑，matplotlib 实现 | 人工复核后再决定 |
| `assessment.py` | 通用评估脚本：评估双目检测系统 只需修改 EXPERIMENT_CONFIG 配置即可评估不同实验 使用方法： 1. 复制此文件为 assess_A.py、assess_B.py 2. 修改每个文件顶部的 EXPERIMENT_CONFIG 3. 运行即可 | 人工复核后再决定 |
| `calib.json.py` | 代码中无模块说明，需结合调用关系复核 | 人工复核后再决定 |
| `compare.py` | 代码中无模块说明，需结合调用关系复核 | 人工复核后再决定 |
| `data_statistics.py` | 对比双目测距结果与激光测距仪真值 计算平均误差、最大误差、成功率 | 人工复核后再决定 |
| `dataset.py` | 空中回收双目数据集退化处理 - PyCharm直接运行版 直接在代码中修改配置变量，无需命令行参数。 | 人工复核后再决定 |
| `debug_assessment.py` | 带调试功能的评估脚本 | 人工复核后再决定 |
| `extract_truth_frame_windows.py` | Extract synchronized +/-N windows around measured stereo center frames. | 人工复核后再决定 |
| `findthedifference.py` | 代码中无模块说明，需结合调用关系复核 | 人工复核后再决定 |
| `fix_yaml.py` | 代码中无模块说明，需结合调用关系复核 | 人工复核后再决定 |
| `prepare/diagnose.py` | 代码中无模块说明，需结合调用关系复核 | 人工复核后再决定 |
| `prepare/prepare.py` | 代码中无模块说明，需结合调用关系复核 | 人工复核后再决定 |
| `prepare/rename.py` | 直接批量改名：distance_detect_*.txt → distance_*.txt | 人工复核后再决定 |
| `prepare/split.py` | 代码中无模块说明，需结合调用关系复核 | 人工复核后再决定 |
| `testIMU.py` | 代码中无模块说明，需结合调用关系复核 | 人工复核后再决定 |
| `yolo_related/apply_fixed_offset_new_distance_20261012_13.py` | Apply the specified +10.8 cm reference-plane offset without altering raw results. | 人工复核后再决定 |
| `yolo_related/audit_new_distance_data_20261012_13.py` | Audit the stereo distance samples captured on 2026-10-12 and 2026-10-13. | 人工复核后再决定 |
| `yolo_related/audit_project_code.py` | Create a non-destructive inventory and cleanup recommendation for project code. | 人工复核后再决定 |
| `yolo_related/bootstrap_metrics.py` | Bootstrap 统计显著性计算 ========================= 对每个(模型, 数据集)组合，通过逐帧 bootstrap 重采样 (1000 iterations) 计算各指标的均值 ± 标准差和 95% 置信区间。 用法: python bootstrap_metrics.py 输出: bootstrap_results.json + 终端表格 | 人工复核后再决定 |
| `yolo_related/build_batch_corrected_truth_manifest.py` | Create a traceable batch-corrected copy of the static stereo truth manifest. | 人工复核后再决定 |
| `yolo_related/build_controlled_scale_bd_dataset.py` | Build a paired, controlled target-scale diagnostic dataset for Models B and D. | 人工复核后再决定 |
| `yolo_related/build_corrected_static_truth_150_report.py` | Build the documented 150-pair, batch-corrected static stereo result table. | 人工复核后再决定 |
| `yolo_related/build_dynamic_hard_replay_v2.py` | Build replay dataset for the second dynamic hard-sample fine-tuning round. | 人工复核后再决定 |
| `yolo_related/build_dynamic_replay_v1.py` | Build a grouped dynamic-scene split mixed with the existing replay dataset. | 人工复核后再决定 |
| `yolo_related/build_equalized_3domain_n344.py` | Build a source-stratified 344-image three-domain training set. Sampling is grouped by capture id, so left/right views stay together. Any capture id present in the fixed Original validation split is excluded from all three training domains t | 人工复核后再决定 |
| `yolo_related/build_final_single_domain_independent_v2.py` | Build a symlink-only Original-domain view of the audited independent_v2 split. | 人工复核后再决定 |
| `yolo_related/build_new_data0826_anomaly_audit.py` | Build a reproducible anomaly audit without deleting difficult test cases. | 人工复核后再决定 |
| `yolo_related/build_new_data0826_manifest.py` | Build a corrected stereo-distance manifest for the 2026-08-26 frozen set. | 人工复核后再决定 |
| `yolo_related/build_phone_replay_dataset.py` | Build a symlink-only replay dataset for incremental real-phone fine-tuning. | 人工复核后再决定 |
| `yolo_related/build_static_stereo_truth_manifest.py` | Build the current manifest for all valid static stereo pairs on disk. | 人工复核后再决定 |
| `yolo_related/carafe_module.py` | CARAFE: Content-Aware ReAssembly of FEatures. Replaces nearest-neighbor upsampling with learned content-aware kernels. Reference: "CARAFE: Content-Aware ReAssembly of Features" Wang et al., ICCV 2019 Key design choices for Jetson Orin (spee | 人工复核后再决定 |
| `yolo_related/compile_results.py` | 编译评估结果，生成消融实验对比表 支持两种输出格式： 1. 文本表格（终端显示） 2. JSON（后续处理用） | 人工复核后再决定 |
| `yolo_related/compute_all_metrics.py` | 消融实验指标全量计算 ======================= 对 5 个模型 (A-E) × 3 个数据集 (original, tension_variation_heavy, light_and_clutter_heavy) 计算: 检测指标: Precision, Recall, mAP@0.5, F1 深度指标: MAE(mm), RMSE(mm), Max Error(mm), 成功率(<50mm) 速度指标: FPS, 平均推理时间(ms) 输出: - 终 | 人工复核后再决定 |
| `yolo_related/cuda_remap.py` | GPU 加速立体校正 — 使用自定义 CUDA kernel (Texture Memory + 硬件双线性插值) 替换 cv2.remap，解决 Jetson 上 OpenCV 无 CUDA 支持的问题。 用法: import cuda_remap rectified = cuda_remap.remap(image, map_x, map_y) rect_l, rect_r = cuda_remap.remap_stereo(left, right, map_lx, ma | 人工复核后再决定 |
| `yolo_related/depth_drift_analysis.py` | 审稿人意见 专家1-第2条：连续帧深度漂移统计 ==================================================== 使用已有的三域测试集，按文件名时间戳顺序排列， 计算相邻帧的深度跳变量，绘制深度波动曲线图。 同时计算： - 相邻帧深度变化量 (Δdepth / frame) - 连续100帧的深度标准差 - 深度漂移率 (cumulative drift) | 人工复核后再决定 |
| `yolo_related/draw_improved_arch.py` | 最终版 — 严格对照 yolov8_designed.yaml 每条连线 | 人工复核后再决定 |
| `yolo_related/draw_network_architectures.py` | Draw YOLOv8n baseline and Model F (improved) neural network architecture diagrams. Reference style: LAR-YOLOv8 paper Fig.2, Fig.4. Usage: python draw_network_architectures.py Output: network_yolov8n_baseline.png, network_model_f_improved.pn | 人工复核后再决定 |
| `yolo_related/engineering_table_chart.py` | 审稿人意见 专家2-第1/3条：工程工况→退化参数→成像效应三列映射表 + SSIM直方图 ====================================================================== 产出: engineering_mapping_table.png + ssim_histogram.png 用法: python engineering_table_chart.py | 人工复核后再决定 |
| `yolo_related/eval_all_3domain.py` | Evaluate all models (A/B/C/E/F) trained on 3-domain dataset. Unified YOLOv8 val() on all 3 test datasets for fair comparison. | 人工复核后再决定 |
| `yolo_related/eval_both_models.py` | Fair comparison: evaluate BOTH models using the SAME YOLOv8 val() method. - Model F (original 348): target_detector_model_f - Model F (3-domain 923): target_detector_augmented | 人工复核后再决定 |
| `yolo_related/exclusion_experiments_chart.py` | 审稿人意见 专家2-第4条：排除性实验补充图表 ==================================================== 将排除性实验的七个方向整理为一张表（排除方案/核心结果/排除原因/设计教训）， 正文展开1-2个最有代表性的。补充 loss 曲线和 mAP 对比图。 排除性实验七方向： 1. NWD Loss (替代 CIoU) 2. Wise-IoU (WIoU) Loss 3. CARAFE 上采样 (替代最近邻) 4. SimAM  | 人工复核后再决定 |
| `yolo_related/experiment_registry/add_submission_audit_20261016.py` | 代码中无模块说明，需结合调用关系复核 | 人工复核后再决定 |
| `yolo_related/experiment_registry/finalize_master_structure_sheet.py` | Add Models G/H to the paper-facing structure comparison sheet. | 人工复核后再决定 |
| `yolo_related/experiment_registry/sync_authoritative_20261015.py` | Create the 2026-10-15 authoritative workbook without overwriting history. | 人工复核后再决定 |
| `yolo_related/experiment_registry/sync_master_20260911.py` | Synchronize verified experiment outputs into experiment_data_master.xlsx. | 人工复核后再决定 |
| `yolo_related/experiment_registry/update_backbone_registry.py` | 代码中无模块说明，需结合调用关系复核 | 人工复核后再决定 |
| `yolo_related/experiment_registry/update_final_D_dynamic_retest_20261016.py` | 代码中无模块说明，需结合调用关系复核 | 人工复核后再决定 |
| `yolo_related/experiment_registry/update_master_registry_20260904.py` | 代码中无模块说明，需结合调用关系复核 | 人工复核后再决定 |
| `yolo_related/export_backbone_frozen_test_excel.py` | Export six-backbone frozen-test metrics to a chart-ready workbook. | 人工复核后再决定 |
| `yolo_related/export_current_ad_onnx.py` | Export current A/D weights to ONNX without creating a CUDA context. | 人工复核后再决定 |
| `yolo_related/export_current_ad_tensorrt_fp16.py` | Export the final paper Models A and D to fixed-shape TensorRT FP16 engines. | 人工复核后再决定 |
| `yolo_related/export_final_D_independent_v2_tensorrt_fp16.py` | Export the audited final independent_v2 Model D as a fixed 640 FP16 engine. | 人工复核后再决定 |
| `yolo_related/export_final_D_onnx_deploy.py` | Export final D with the TensorRT-stable illumination statistic graph. | 人工复核后再决定 |
| `yolo_related/export_tensorrt.py` | TensorRT 引擎导出 + FP16/INT8 优化 ==================================== 将训练好的 PyTorch 模型导出为 TensorRT 引擎，大幅加速推理。 支持的量化模式: - FP16: Jetson Orin 上可获 2-3x 加速，精度几乎无损 - INT8: 额外 1.3-1.5x 加速，需校准数据 用法: # 导出 Model E → FP16 TensorRT python export_tensorrt.p | 人工复核后再决定 |
| `yolo_related/export_trt_int8.py` | Export all 6 models to TensorRT INT8 with calibration. | 人工复核后再决定 |
| `yolo_related/extreme_degradation_boundary.py` | 审稿人意见 专家1-第3条：极端工况性能边界测试 ==================================================== - PSF 核半径 10-60 像素运动模糊 - 对比 Model A (基线YOLOv8n) + Model D (SPDConv+P2+C2fAttn) + Model F (完整提案) - 指标: mAP@50 + 检测成功率 - 产出: psf_vs_map_dual.png (A/D/F 三线) + JSON 用 | 人工复核后再决定 |
| `yolo_related/fill_excel.py` | 自动填写消融实验 Excel 表格 (v2) 从 metrics_all.json 读取指标，填入 计划.xlsx 的「消融试验」表。 Sheet 结构: 数据集块1 (rows 2-7): original 数据集块2 (rows 10-15): tension_variation_heavy 数据集块3 (rows 18-23): light_and_clutter_heavy 数据集块4 (rows 26-30): light_and_clutter_heavy (备用 | 人工复核后再决定 |
| `yolo_related/freeze_dataset_split.py` | Create a reproducible, source-stratified stereo split for the combined dataset. | 人工复核后再决定 |
| `yolo_related/gen_missing_preds.py` | Generate missing single-domain predictions (E_v2, F_v2). | 人工复核后再决定 |
| `yolo_related/generate_3domain_predictions.py` | Generate pred_boxes for 3-domain trained models on all test datasets. | 人工复核后再决定 |
| `yolo_related/gmm_depth_chart.py` | 审稿人意见 专家2-第4条：GMM vs SGBM 深度估计 MAE 柱状对比图 ======================================================== 数据源: depth_comparison.json + exclusion 硬编码 + 原始日志 用法: python gmm_depth_chart.py | 人工复核后再决定 |
| `yolo_related/gmm_depth_estimator.py` | Model G: Model F 检测 + GMM 点集匹配深度估计 ================================================ 基于胡欣等[4]的GMM图像点集匹配方法，替换Model F的中心视差深度估计。 算法流程: 1. 使用Model F的检测框 (pred_boxes_F) 2. 在左目ROI内提取Shi-Tomasi角点 3. 沿对极线(已校正)在右目搜索NCC最佳匹配 4. 使用2分量GMM分离内点/外点视差 5. 取内点 | 人工复核后再决定 |
| `yolo_related/image_size_utils.py` | Resolve the real image size associated with a YOLO label or prediction. | 人工复核后再决定 |
| `yolo_related/make_real_scene_4x4_qualitative.py` | Build the paper 4x4 qualitative grid with a shared enlarged crop per scene. | 人工复核后再决定 |
| `yolo_related/model_f_illumination.py` | Model F detection with optional brightness-aware input enhancement. Coordinates remain in original-image pixels. Pass ORIGINAL rectified images, not enhanced detector inputs, to stereo matching. | 人工复核后再决定 |
| `yolo_related/nwd_loss.py` | NWD (Normalized Wasserstein Distance) Loss for small object detection. Reference: "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection" Wang et al., arXiv:2110.13389 NWD models bboxes as 2D Gaussian distributions and measur | 人工复核后再决定 |
| `yolo_related/optimize_inference.py` | Optimized Inference Pipeline for Model C Optimizations: 1. GPU-accelerated remap (grid_sample) 2. TensorRT export + inference 3. FP16 half-precision 4. Reduced resolution 5. Pipeline (overlap CPU/GPU) | 人工复核后再决定 |
| `yolo_related/pgi_collapse_chart.py` | 审稿人意见 专家2-第4条：YOLOv9t PGI 梯度崩溃曲线 ============================================= 从 v9t 训练日志提取 val/cls_loss 崩溃过程，绘制异常曲线图。 用法: python pgi_collapse_chart.py | 人工复核后再决定 |
| `yolo_related/prepare_combined_plus_synthetic_lowlight_100.py` | Add 100 synthetic low-light images to Combined train; keep val unchanged. | 人工复核后再决定 |
| `yolo_related/prepare_dynamic_annotation_pool.py` | Build a deterministic stereo annotation pool from selected dynamic clips. | 人工复核后再决定 |
| `yolo_related/prepare_illumination_aug_dataset.py` | Create a training-only illumination-randomized replay dataset. Validation and test images remain byte-identical symlinks to the existing split. One synthetic low-light image is generated for each unique training source so replayed samples d | 人工复核后再决定 |
| `yolo_related/prepare_lowlight_real_0828_v1.py` | Create a pair-grouped split for the real low-light stereo images. | 人工复核后再决定 |
| `yolo_related/prepare_lowlight_replay_0828_v1.py` | Mix real low-light training pairs with the existing Model F replay set. | 人工复核后再决定 |
| `yolo_related/prepare_real_phone_v1.py` | Copy, group-split and version the real-phone monocular detection dataset. | 人工复核后再决定 |
| `yolo_related/prepare_synthetic_lowlight_100.py` | Generate 100 training-only synthetic low-light images from 50 stereo pairs. | 人工复核后再决定 |
| `yolo_related/prune_and_distill.py` | 模型压缩工具: 结构化剪枝 + 知识蒸馏 + 轻量训练 三种策略: 1. prune — L1 结构化剪枝 (保留架构, 裁剪通道) 2. distill — 知识蒸馏 (大模型→小模型) 3. train — 直接训练 pico 变体 (最快) 用法: python prune_and_distill.py prune --ratio 0.3 # 剪掉 30% 通道 python prune_and_distill.py distill # 蒸馏训练 pico python | 人工复核后再决定 |
| `yolo_related/qualitative_comparison.py` | 定性对比图生成 ================ 为每个数据集选取代表性帧，叠加显示: - Ground Truth (绿色实线) - Model A 基线 (红色虚线) - Model F 最优 (蓝色实线) - ORB 传统CV (黄色点线) - SIFT 传统CV (青色点线) - 如果可用 输出: qualitative_comparison.png 用法: python qualitative_comparison.py | 人工复核后再决定 |
| `yolo_related/retrain_A_and_evaluate_final_abcd_v2.py` | Retrain a clean Model A, then evaluate final A/B/C/D on frozen tests. | 人工复核后再决定 |
| `yolo_related/scale_stratified_detection_experiment.py` | Measure 640-letterboxed target scales and evaluate Models A/F by scale. | 人工复核后再决定 |
| `yolo_related/select_real_scene_qualitative_samples.py` | Select one shared image per real scene for the A/B/C/D qualitative grid. | 人工复核后再决定 |
| `yolo_related/summarize_new_data0826_stereo.py` | Summarize new_data0826 stereo results by distance and theoretical scale. | 人工复核后再决定 |
| `yolo_related/summarize_static_stereo_corrected.py` | Apply documented batch reference offsets and summarize current stereo results. | 人工复核后再决定 |
| `yolo_related/summarize_stereo_fixed_reference.py` | Recompute stereo errors after applying a fixed reference-plane offset. The offset belongs to the measurement setup, not to the stereo algorithm. Raw estimates are preserved and corrected values are written to separate files. | 人工复核后再决定 |
| `yolo_related/target_size_boundary.py` | 审稿人意见 专家1-第3条：靶标尺寸递减性能边界测试 ==================================================== 方法: 按靶标在 640 推理空间中的自然像素尺寸分箱 (natural binning) - 不进行人工缩放（避免引入 artifacts） - 直接用原始测试集图像，按目标像素宽度分箱评估 - 每个 bin 内目标尺寸分布均匀，反映真实退化趋势 - 指标: mAP@50 + 检测成功率 - 产出: target_s | 人工复核后再决定 |
| `yolo_related/traditional_cv_comparison.py` | Traditional CV vs YOLOv8n Comparison ===================================== 对比方法: - ORB: ORB 多尺度模板匹配 + Homography → 中心视差深度 - SIFT: SIFT 多尺度模板匹配 + Homography → 中心视差深度 - ORB+SGBM: ORB 检测 + SGBM 稠密视差 (ROI) - SIFT+SGBM: SIFT 检测 + SGBM 稠密视差 (ROI) | 人工复核后再决定 |
| `yolo_related/train_acf_equalized_3domain_n344.py` | Sequentially train Models A, C and F on the equalized 344-image 3-domain set. | 人工复核后再决定 |
| `yolo_related/train_architecture_design_ablation.py` | Train controlled architecture ablations requested by the reviewer. | 人工复核后再决定 |
| `yolo_related/train_backbone_comparison_full.py` | 审稿人意见 专家1-第1条：骨干选型对比不完整 ==================================================== 补充 YOLOv5s（v5系列轻量标杆）和 YOLOv11n（最新代际） 在 original 数据集上的 mAP 和 FPS 对比。 已有: YOLOv8n, YOLOv9t, YOLOv10n 新增: YOLOv5s, YOLOv11n 输出到: runs/compare/ 和 小论文/images/v8_v9_v10_ | 人工复核后再决定 |
| `yolo_related/train_ef_single.py` | Continue training E and F on single-domain (original 348, no MixUp). | 人工复核后再决定 |
| `yolo_related/train_h_equalized_3domain_n344.py` | Train Model H on the 344-image equalized three-domain control set. | 人工复核后再决定 |
| `yolo_related/train_v8_v9_v10.py` | YOLOv8n vs YOLOv9t vs YOLOv10n 对比训练脚本 使用相同的数据集和超参数，确保公平对比 | 人工复核后再决定 |
| `yolo_related/training_monitor.py` | Small local-only dashboard for Ultralytics training progress. | 人工复核后再决定 |
| `yolo_related/trt_inference.py` | TensorRT 推理引擎 — YOLOv8 FP16 加速 使用 ctypes CUDA runtime 分配显存，避免 Jetson 统一内存上 torch.empty 的 Bus error。 支持 YOLOv8 letterbox 预处理 (保持宽高比 + padding)。 用法: engine = TRTInference('model_f_640x640.engine') boxes = engine.infer(image_np) # image_np: (H | 人工复核后再决定 |
| `yolo_related/validate_dataset_fid.py` | 审稿人意见 专家2-第3条：退化数据集有效性验证 ==================================================== 1. 计算退化前后图像的 FID (Fréchet Inception Distance) 分布距离指标 2. 生成退化参数→工程工况映射表 3. 交叉验证：同一退化等级下仿真数据训练、真实数据测试 FID 通过 torchvision 的 InceptionV3 特征提取器计算。 | 人工复核后再决定 |
| `yolo_related/visualize_3domain_figures.py` | 三域训练消融实验图表 — 完整版本 A/B/C/D/E/F | 人工复核后再决定 |
| `yolo_related/visualize_ambiguous_dynamic_candidates.py` | Draw all detections on the eight ambiguous dynamic frames. | 人工复核后再决定 |
| `yolo_related/visualize_paper_figures.py` | 论文图表生成: PR曲线 + 深度误差分布直方图 ============================================ 生成适用于学术论文的: 图1: Precision-Recall 曲线 (按数据集分行, 按模型分色) 图2: 深度误差分布直方图 (按数据集分行, 按模型分色) 用法: python visualize_paper_figures.py 输出: /home/nvidia/Documents/bincular_vision/小论文/ima | 人工复核后再决定 |
| `yolo_related/visualize_stereo_detection_errors.py` | Visualize Model F detections on stereo frames with large depth errors. | 人工复核后再决定 |
| `yolo_related/wiou_loss.py` | Wise-IoU (WIoU) Loss Implementation. WIoU v1: Attention-based IoU loss that uses distance attention to reduce the penalty on low-quality examples, improving generalization. Reference: Tong et al., "Wise-IoU: Bounding Box Regression Loss wit | 人工复核后再决定 |
| `yolo_related/yolo_detect.py` | 基于 YOLOv8 的实时靶标检测 + ZED 双目深度定位 支持 ZED 相机（自动获取深度）和普通 USB 摄像头（仅检测） | 人工复核后再决定 |
| `yolo_related/yolo_detect_static.py` | 双目靶标定位（支持立体校正 + 预测框保存 + 耗时记录）： - 从保存的左右图像对中检测靶标并计算距离 - 自动加载标定参数进行畸变校正和极线对齐（如提供） - 距离结果保存到 distance_detect/ 文件夹 - 左目预测框（YOLO 格式）保存到 pred_boxes/ 文件夹 - 每帧推理耗时记录到 inference_time.txt | 人工复核后再决定 |
| `yolo_related/yolo_detect_static_1.py` | 双目靶标定位：从保存的左右图像对中检测靶标并计算距离 输出保存到 distance_detect/ 文件夹，与真值文件同名 改进： - 自适应光照归一化（CLAHE + Gamma） - 动态高分辨率裁剪：远距离目标先全局检测，若未检出则裁剪中心区域放大再次检测 | 人工复核后再决定 |
| `yolo_related/yolo_detected_improved.py` | 代码中无模块说明，需结合调用关系复核 | 人工复核后再决定 |
| `yolo_related/yolo_result.py` | 代码中无模块说明，需结合调用关系复核 | 人工复核后再决定 |

## 采集与标定（保护）

| 文件 | 用途 | 建议 |
|---|---|---|
| `calibration.py` | 代码中无模块说明，需结合调用关系复核 | 保留/禁止删除 |
| `extract_selected_stereo_frames.py` | Extract selected synchronized left/right frames from a stereo recording. | 保留/禁止删除 |
| `extract_stereo_frames.py` | Extract synchronized continuous frames from a recorded stereo MP4 pair. | 保留/禁止删除 |
| `stereo_video_imu_recorder.py` | Record synchronized ZED X stereo video and IMU data for vibration tests. Controls -------- B: begin recording left/right video and IMU data Q: stop recording (if active), save metadata, and exit The camera frame timestamps and IMU timestamp | 保留/禁止删除 |
| `stereo_video_recorder.py` | Record synchronized, timestamped left/right video from a ZED X camera. The camera setup follows takephotos.py: HD1200, 15 FPS and DEPTH_MODE.NONE. Press Q in the preview window to stop recording safely. | 保留/禁止删除 |
| `takephotos.py` | 代码中无模块说明，需结合调用关系复核 | 保留/禁止删除 |
| `takephotos_withdistance.py` | 代码中无模块说明，需结合调用关系复核 | 保留/禁止删除 |
