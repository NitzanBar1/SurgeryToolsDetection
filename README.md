# SurgeryToolsDetection
Detect Surgery tools using YOLOS model from huggingface
This project perform detection of surgery tools using YOLOS model.  
This is a class project as part of 097222 Computer Vision Seminar @ Technion.  

<p align="center">
    <a href="https://www.linkedin.com/in/nitzan-bar-9ab896146/">Nitzan Bar</a>
</p>


- [SurgeryToolsDetection](#surgery-tools-detection)
  * [Files in The Repository](#files-in-the-repository)
  * [Dataset](#dataset) 
  * [YOLOS Model](#yolos-model)
  * [Results](#results)
  * [References](#references)



## Files in the repository
|File name         | Purpsoe |
|----------------------|------|
|`clean_env.yml`| conda enviornment file|
|`predict.py`| predict on a single image|
|`video.py`| predict on a single video|
|`train_yolos.py`| train yolos model|
|`my_coco_eval.py`| modified coco_eval file for evaluation|
|`images`| Images used for preview in README.md file|



## Dataset
![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/vis.png)
![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/insights.png)



## YOLOS Model
•	The YOLOS model was proposed in You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection by Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, Wenyu Liu. YOLOS proposes to just leverage the plain Vision Transformer (ViT) for object detection, inspired by DETR. It turns out that a base-sized encoder-only Transformer can also achieve 42 AP on COCO, like DETR and much more complex frameworks such as Faster R-CNN.

•	The abstract from the paper is the following:
Can Transformer perform 2D object- and region-level recognition from a pure sequence-to-sequence perspective with minimal knowledge about the 2D spatial structure? To answer this question, we present You Only Look at One Sequence (YOLOS), a series of object detection models based on the vanilla Vision Transformer with the fewest possible modifications, region priors, as well as inductive biases of the target task. We find that YOLOS pre-trained on the mid-sized ImageNet-1k dataset only can already achieve quite competitive performance on the challenging COCO object detection benchmark, e.g., YOLOS-Base directly adopted from BERT-Base architecture can obtain 42.0 box AP on COCO val. We also discuss the impacts as well as limitations of current pre-train schemes and model scaling strategies for Transformer in vision through YOLOS.

•	We replaced COCO classification head with a custom head.

An illustration of the YOLOS architecture is shown below:

![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/yolos.png)



## Results
The model was compiled in Microsoft Azure using PyTorch packages with 25 epochs. 
We chose YOLOS as our base network due to its short run-time. This model performed very well on our dataset and achieved a mean average precision (mAP) of 73.6 for tool detection.
The loss and mAP@K graphs are shown below.

![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/graphs1.png)
![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/graphs2.png)

Model detections:
![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/results.png)

Evaluation metrics:\n
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.593\n
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.7360 
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.713
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.595
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.591
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.732
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.777
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.790
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.749
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.792 



## References
[1] Goldbraikh, A., D’Angelo, A.L., Pugh, C.M. and Laufer, S., 2022. Video-based fully automatic assessment of open surgery suturing skills. International Journal of Computer Assisted Radiology and Surgery, 17(3), pp.437-448.

[2] Fang, Y., Liao, B., Wang, X., Fang, J., Qi, J., Wu, R., Niu, J. and Liu, W., 2021. You only look at one sequence: Rethinking transformer in vision through object detection. Advances in Neural Information Processing Systems, 34, pp.26183-26197.
