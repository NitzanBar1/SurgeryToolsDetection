# -*- coding: utf-8 -*-
import os
import torch
import pytorch_lightning as pl
from transformers import AutoFeatureExtractor
from transformers import AutoModelForObjectDetection
import cv2
import bbox_visualizer as bbv
import argparse


"""## Image Classification Collator
To apply our transforms to images, we'll use a custom collator class. We'll initialize it using an instance of `ViTFeatureExtractor` and pass the collator instance to `torch.utils.data.DataLoader`'s `collate_fn` kwarg.
"""
class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.feature_extractor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['labels'] = labels
        return batch


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size[0], size[1]
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def check_current_tool(tool_usage, i):
    for elem in tool_usage:
        if i >= int(elem[0]) and i <= int(elem[1]):
            return elem[2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Predict on image')
    parser.add_argument('--image_file', required=True)
    parser.add_argument('--model_path', required=True)
    args = parser.parse_args()

    # Set global seed
    global_seed = 42
    pl.seed_everything(global_seed)

    """## Init Feature Extractor, Model"""
    # Init Feature Extractor   
    feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-small", size=640, max_size=640)

    # Define Model
    num_labels = 8
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny", 
    num_labels=num_labels, ignore_mismatched_sizes=True)

    """## Load fine-tuned model for inference """
    #model_path = os.path.join('models/yolos', '08122022_143316_model.pth')
    model_path = args.model_path
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Read input image
    image = cv2.imread(args.image_file)

    # colors for visualization
    colors = [[0, 128, 255], [255, 0, 0], [255, 128, 0],
    [128, 0, 128], [128, 255, 0], [0, 0, 255]]

    id2label = {
        0: 'surgery-tools',
        1: 'Left_Empty',
        2: 'Left_Forceps',
        3: 'Left_Needle_driver',
        4: 'Left_Scissors',
        5: 'Right_Empty',
        6: 'Right_Needle_driver',
        7: 'Right_Scissors'
    }

    # apply featue extractor on input image
    pixel_values = feature_extractor(image, return_tensors="pt")['pixel_values']
    pixel_values = pixel_values.to(device)
    image_size = torch.zeros(1,2)
    image_size[0,0] = image.shape[0]
    image_size[0,1] = image.shape[1]
    image_size = image_size.to(device)

    # predict bounding boxes
    outputs = model(pixel_values=pixel_values)
    
    # convert outputs of model to COCO api
    results = feature_extractor.post_process(outputs, image_size) 

    # keep the two boxes with highest scores
    preds = []
    for res_dict in results:
        scores = res_dict['scores']
        idx = torch.argsort(scores, descending=True)
        boxes = res_dict['boxes'][idx[0:2]]
        scores = res_dict['scores'][idx[0:2]]
        labels = res_dict['labels'][idx[0:2]]
        preds.append(
            dict(
                boxes=boxes,
                scores=scores,
                labels=labels
            )
        )

    prob = preds[0]['scores']
    boxes = preds[0]['boxes']
    labels = preds[0]['labels']

    for p, (xmin, ymin, xmax, ymax), c, cl in zip(prob, boxes.tolist(), colors, labels):
        if p <= 0.001: # drop bboxes with score less than 0.001
            continue
        box = [int(xmin), int(ymin), int(xmax), int(ymax)]
        image = bbv.draw_rectangle(image, box, bbox_color=[0,0,255])
        text = id2label[cl.item()] + ': ' + f'{p.item():.3f}'
        image = bbv.add_T_label(image, text, box, text_bg_color=[0,0,255])
    
    # Display the resulting frame
    if not(os.path.exists('predict_on_images')):
        os.makedirs('predict_on_images')
    output_file = os.path.join('predict_on_images', os.path.basename(args.image_file))

    cv2.imwrite(output_file, image)

# Run Command: python3 predict.py --image_file HW1_dataset/HW1_dataset/images/P016_balloon1_9.jpg --model_path models/yolos/08122022_143316_model.pth