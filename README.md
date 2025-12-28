# DINOv2 Crack Detection with Dual-Teacher Distillation & Lipschitz Constraints
This repository contains the official implementation of a DINOv2-based crack detection method that integrates heterogeneous dual-teacher knowledge distillation and Lipschitz constraints, designed to tackle challenges such as fragile elongated structures, difficult multi-scale fusion, and environmental noise interference in crack detection.
# Key Features
🧠 Heterogeneous Dual-Teacher Knowledge Distillation Framework: Combines the high-level semantic comprehension of DINOv2-large with the detailed feature preservation capabilities of DINOv2-base.
🔗 Channel Attention Feature Fusion Module (CAFM): Adaptively fuses multi-source heterogeneous features from different teachers.
🛡️ Lipschitz-Constrained Decoder: Enhances model robustness against noise and adversarial perturbations through Spectral Normalization (SN) and Gradient Penalty (GP).
📈 High Performance: Achieves 61.91% Crack_IoU and 75.49% F1-score on the DeepCrack dataset, with improved stability under adversarial attacks.
