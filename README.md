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
  * [Models](#models)
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



## Models
•	*YOLOS architecture:*  
YOLOS model fine-tuned on COCO 2017 object detection (118k annotated images). It was introduced in the paper You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection by Fang et al. and first released in this repository.  
YOLOS is a Vision Transformer (ViT) trained using the DETR loss. Despite its simplicity, a base-sized YOLOS model is able to achieve 42 AP on COCO validation 2017 (similar to DETR and more complex frameworks such as Faster R-CNN).  
The model is trained using a "bipartite matching loss": one compares the predicted classes + bounding boxes of each of the N = 100 object queries to the ground truth annotations, padded up to the same length N (so if an image only contains 4 objects, 96 annotations will just have a "no object" as class and "no bounding box" as bounding box). The Hungarian matching algorithm is used to create an optimal one-to-one mapping between each of the N queries and each of the N annotations. Next, standard cross-entropy (for the classes) and a linear combination of the L1 and generalized IoU loss (for the bounding boxes) are used to optimize the parameters of the model.  
We replaced COCO classification head with a custom head.

An illustration of the YOLOS architecture is shown below:

![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/yolos.png)


•	*DETR Architecture:*  
DEtection TRansformer (DETR) model trained end-to-end on COCO 2017 object detection (118k annotated images). It was introduced in the paper End-to-End Object Detection with Transformers by Carion et al. and first released in this repository.  
The DETR model is an encoder-decoder transformer with a convolutional backbone. Two heads are added on top of the decoder outputs in order to perform object detection: a linear layer for the class labels and a MLP (multi-layer perceptron) for the bounding boxes. The model uses so-called object queries to detect objects in an image. Each object query looks for a particular object in the image. For COCO, the number of object queries is set to 100.

An illustration of the DETR architecture is shown below:

![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/detr.png)




## Results
The model was compiled in Microsoft Azure using PyTorch packages with 25 epochs. 
We chose YOLOS as our base network due to its short run-time. This model performed very well on our dataset and achieved a mean average precision (mAP) of 73.6 for tool detection.
The loss and mAP@K graphs are shown below.

![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/graphs1.png)
![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/graphs2.png)

Model detections:
![alt text](https://github.com/NitzanBar1/SurgeryToolsDetection/blob/main/images/results.png)

Evaluation metrics:

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.593  
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
