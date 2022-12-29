# -*- coding: utf-8 -*-
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor
from transformers import AutoModelForObjectDetection
import torchvision
from roboflow import Roboflow
import sys
from git import Repo
import argparse


# Dataset class - COCO format
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "_annotations.coco.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target


"""## Image Classification Collator
To apply our transforms to images, we'll use a custom collator class. We'll initialize it using an instance of `ViTFeatureExtractor` and pass the collator instance to `torch.utils.data.DataLoader`'s `collate_fn` kwarg.
"""
class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.feature_extractor(pixel_values, return_tensors="pt")
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
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes, colors_arg):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = colors_arg * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
    plt.savefig('test_results.png')


def visualize_predictions(image, outputs, threshold, colors_arg):
  # keep only predictions with confidence >= threshold
  probas = outputs.logits.softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold
  
  # convert predicted boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

  # plot results
  plot_results(image, probas[keep], bboxes_scaled, colors_arg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate Model')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--hf_model', required=True)
    args = parser.parse_args()

    # Set global seed
    global_seed = 42
    pl.seed_everything(global_seed)

    """## Init Feature Extractor, Model, Data Loaders"""
    # Import data from roboflow
    rf = Roboflow(api_key="YxuydqSbMvRyuPTb9wMG")
    project = rf.workspace("hw1").project("surgery-tools-detect")
    dataset = project.version(1).download("coco")
    print(dataset.location)

    # Init Feature Extractor, Model, Data Loaders    
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.hf_model, size=640, max_size=640)
    train_dataset = CocoDetection(img_folder=(dataset.location + '/train'), feature_extractor=feature_extractor)
    val_dataset = CocoDetection(img_folder=(dataset.location + '/valid'), feature_extractor=feature_extractor, train=False)
    test_dataset = CocoDetection(img_folder=(dataset.location + '/test'), feature_extractor=feature_extractor, train=False)
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))
    print("Number of test examples:", len(test_dataset))

    # Create dataloaders
    collator = ImageClassificationCollator(feature_extractor)
    num_workers = 12
    BATCH_SIZE = 2
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collator, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collator, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collator, shuffle=False, num_workers=num_workers)

    # Define Model
    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}
    model = AutoModelForObjectDetection.from_pretrained(args.hf_model, 
    num_labels=len(id2label), ignore_mismatched_sizes=True)


    """## Load fine-tuned model for inference """
    #model_path = os.path.join('models/yolos', '17122022_134101_model.pth')
    model_path = args.model_path
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evalute
    if not(os.path.exists('/home/student/detr')):
        Repo.clone_from('https://github.com/facebookresearch/detr.git', '/home/student/detr')
    sys.path.insert(0, '/home/student/detr')
    from my_coco_eval import CocoEvaluator
    from detr.datasets import get_coco_api_from_dataset

    base_ds = get_coco_api_from_dataset(test_dataset) # this is actually just calling the coco attribute
    iou_types = ['bbox']
    coco_evaluator = CocoEvaluator(base_ds, iou_types) # initialize evaluator with ground truths

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    print("Running evaluation...")

    for idx, batch in enumerate(test_dataloader):
        # get the inputs
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

        # forward pass
        outputs = model(pixel_values=pixel_values)

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
        res = {target['image_id'].item(): output for target, output in zip(labels, results)}
        coco_evaluator.update(res)
        

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()
    print(stats)
    #the evaluation here prints out mean average precision details
    #learn more - https://blog.roboflow.com/mean-average-precision/

    """# Visualizing Inference on Test Images
    """
    #We can use the image_id in target to know which image it is
    pixel_values, target = test_dataset[np.random.randint(0, len(test_dataset))]
    pixel_values = pixel_values.unsqueeze(0).to(device)
    outputs = model(pixel_values=pixel_values)


    #lower confidence yields more, but less accurate predictions
    confidence=0.2

    # colors for visualization
    colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    
    image_id = target['image_id'].item()
    image = test_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join(dataset.location + '/test', image['file_name']))

    visualize_predictions(image, outputs, confidence, colors)

# run command: python3 evaluate_model.py --model_path models/yolos/29122022_162027_model.pth --hf_model facebook/detr-resnet-101
# tensorboard --logdir=lightning_logs