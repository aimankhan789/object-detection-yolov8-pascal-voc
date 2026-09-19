# YOLOv8 Object Detection using Transfer Learning

Object detection project using **YOLOv8**, **PyTorch**, and the **PASCAL VOC 2012** dataset.

## Project Overview

This project implements multi-class object detection using YOLOv8 transfer learning.

The model is trained to detect **20 object classes** from the PASCAL VOC 2012 dataset and predict their locations using bounding boxes.

## Dataset

**Dataset:** PASCAL VOC 2012

* Images: 17,125
* Object classes: 20
* Original annotation format: XML
* Converted annotation format: YOLO TXT
* Train images: 13,700
* Validation images: 3,425

## VOC Classes

1. aeroplane
2. bicycle
3. bird
4. boat
5. bottle
6. bus
7. car
8. cat
9. chair
10. cow
11. diningtable
12. dog
13. horse
14. motorbike
15. person
16. pottedplant
17. sheep
18. sofa
19. train
20. tvmonitor

## Technologies

* Python
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* NumPy
* Matplotlib
* Gradio
* ONNX
* FastAPI
* Kaggle GPU (Tesla T4)

## Model Training

A pretrained **YOLOv8n** model was fine-tuned on the PASCAL VOC dataset.

Training configuration:

* Epochs: 50
* Batch size: 16
* Image size: 640 × 640
* GPU: Tesla T4
* Transfer learning: COCO-pretrained YOLOv8n

## Evaluation Results

The final validation evaluation produced:

| Metric    | Result |
| --------- | -----: |
| Precision | 70.04% |
| Recall    | 57.43% |
| mAP@50    | 64.46% |
| mAP@50-95 | 47.84% |

## Project Features

* PASCAL VOC XML annotation parsing
* XML to YOLO format conversion
* Normalized bounding box coordinates
* YOLOv8 transfer learning
* Object detection on validation images
* Detection grid visualization
* Per-class AP analysis
* Confidence threshold experiment
* Ground Truth vs Prediction comparison
* Precision-Recall curve
* ONNX model export
* Gradio web interface

## Project Structure

```text
object-detection-yolov8-pascal-voc/
│
├── README.md
├── notebook.ipynb
├── app.py
├── data.yaml
├── convert_voc_to_yolo.py
├── requirements.txt
│
├── api/
│   └── api.py
│
├── weights/
│   └── best.onnx
│
└── images/
    ├── class_distribution.png
    ├── bbox_size_scatter.png
    ├── detection_grid.png
    ├── gt_vs_pred.png
    ├── per_class_ap.png
    ├── training_curves.png
    ├── pr_curve.png
    └── nms_comparison.png
```

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Gradio application:

```bash
python app.py
```

## Results

The trained YOLOv8 model can detect multiple PASCAL VOC object categories and draw bounding boxes around detected objects.

Visual evaluation was performed using detection grids, ground-truth comparisons, per-class AP, and confidence-threshold experiments.

## Limitations

The achieved mAP@50 is slightly below the project's target range of 0.65–0.80. Performance varies between object classes because of differences in object size, appearance, background complexity, and occlusion.

## Future Improvements

* Train for more epochs
* Experiment with larger YOLOv8 models
* Improve data augmentation
* Tune confidence and NMS thresholds
* Perform additional hyperparameter tuning
* Increase training data

## Author

YOLOv8 Object Detection Project — PASCAL VOC 2012

