# APEM
# Farmland Segmentation

**APEM: An Adaptive Prior-Enhanced Model for Language-Driven Remote Sensing Farmland Segmentation**

This is a multimodal farmland segmentation model based on the Swin-V2-B backbone, Agriculture-BERT text encoder, Mamba global modeling, local self-attention and cross-modal attention, and attention-guided adaptive fusion. The model features a specially designed boundary enhancement branch to further improve edge accuracy.

**Key Features**
Backbone: Swin-Transformer V2-B (Visual) & Agriculture-BERT (Text).

Core Mechanism: Attention-Guided Adaptive Fusion based on Kalman Filter principles to handle cross-modal uncertainty.

Decoder: A hybrid decoder leveraging Mamba (SSM) for efficient global modeling and local self-attention for detail refinement.

Boundary Branch: A dedicated branch to sharpen segmentation edges, critical for farmland delineation.

### Performance (on the FarmSeg-VL dataset)

- Test Set mIoU: ~88.92
- Test Set Dice: ~94.12
- Test Set ACC: ~95.23
- Test Set Recall: ~94.54
