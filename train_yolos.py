# -*- coding: utf-8 -*-
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from PIL import Image, ImageDraw
import numpy as np
#from tensorboard import program
from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor
from transformers import AutoModelForObjectDetection
import torchvision
from roboflow import Roboflow
import sys
from git import Repo
from datetime import datetime
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
        encoding = self.feature_extractor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['labels'] = labels
        return batch


"""# Training
⚡ We'll use [PyTorch Lightning](https://pytorchlightning.ai/) to fine-tune our model.
"""
class Detector(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, batch_size, hparams):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.iou_types = ['bbox']
        self.save_hyperparameters(hparams)
        self.train_ds = get_coco_api_from_dataset(train_dataset) # this is actually just calling the coco attribute
        self.val_ds = get_coco_api_from_dataset(val_dataset) # this is actually just calling the coco attribute
        self.coco_evaluator_train = CocoEvaluator(self.train_ds, self.iou_types) # initialize evaluator with ground truths
        self.coco_evaluator_val = CocoEvaluator(self.val_ds, self.iou_types) # initialize evaluator with ground truths

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs
     
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
        res = {target['image_id'].item(): output for target, output in zip(labels, results)}
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict, res

    def training_step(self, batch, batch_idx):
        loss, loss_dict, res = self.common_step(batch, batch_idx) 
        self.coco_evaluator_train.update(res)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=self.batch_size)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item(), on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=self.batch_size)
        return_dict = {'loss': loss}
        return return_dict
    
    def training_epoch_end(self, outputs) -> None:
        gathered = self.all_gather(outputs)
        if self.global_rank == 0:
            loss = sum(output['loss'].mean() for output in gathered) / len(outputs)
            self.logger.experiment.add_scalars('loss', {'train': loss}, self.current_epoch)
        self.coco_evaluator_train.synchronize_between_processes()
        self.coco_evaluator_train.accumulate()
        stats = self.coco_evaluator_train.summarize()
        self.logger.experiment.add_scalars('mAP@25', {'train': stats[0][0]}, self.current_epoch)
        self.logger.experiment.add_scalars('mAP@50', {'train': stats[0][1]}, self.current_epoch)
        self.logger.experiment.add_scalars('mAP@75', {'train': stats[0][2]}, self.current_epoch)
        self.coco_evaluator_train = CocoEvaluator(self.train_ds, self.iou_types) # initialize evaluator with ground truths

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, res = self.common_step(batch, batch_idx)
        self.coco_evaluator_val.update(res)     
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=self.batch_size)
    
        for k,v in loss_dict.items():
            self.log("val_" + k, v.item(), on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=self.batch_size)
        return_dict = {'val_loss': loss}
        return return_dict

    def validation_epoch_end(self, outputs) -> None:
        gathered = self.all_gather(outputs)
        if self.global_rank == 0:
            loss = sum(output['val_loss'].mean() for output in gathered) / len(outputs)
            self.logger.experiment.add_scalars('loss', {'val': loss}, self.current_epoch)
        self.coco_evaluator_val.synchronize_between_processes()
        self.coco_evaluator_val.accumulate()
        stats = self.coco_evaluator_val.summarize()
        self.logger.experiment.add_scalars('mAP@25', {'val': stats[0][0]}, self.current_epoch)
        self.logger.experiment.add_scalars('mAP@50', {'val': stats[0][1]}, self.current_epoch)
        self.logger.experiment.add_scalars('mAP@75', {'val': stats[0][2]}, self.current_epoch)
        self.coco_evaluator_val = CocoEvaluator(self.val_ds, self.iou_types) # initialize evaluator with ground truths
            

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, 
                                      weight_decay=self.weight_decay)
        return optimizer


# --------- MAIN --------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser('train')
    parser.add_argument('--batch_size', required=True)
    parser.add_argument('--lr', required=True)
    parser.add_argument('--weight_decay', required=True)
    parser.add_argument('--max_epochs', required=True)
    args = parser.parse_args()

    if not(os.path.exists('/home/student/detr')):
        Repo.clone_from('https://github.com/facebookresearch/detr.git', '/home/student/detr')
    sys.path.insert(0, '/home/student/detr')
    #from detr.datasets.my_coco_eval import CocoEvaluator
    from my_coco_eval import CocoEvaluator
    from detr.datasets import get_coco_api_from_dataset

    # Set global seed
    global_seed = 42
    pl.seed_everything(global_seed)

    # Import data from roboflow
    rf = Roboflow(api_key="YxuydqSbMvRyuPTb9wMG")
    project = rf.workspace("hw1").project("surgery-tools-detect")
    dataset = project.version(1).download("coco")
    print(dataset.location)

    # Init Feature Extractor, Model, Data Loaders    
    feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-small", size=640, max_size=640)
    train_dataset = CocoDetection(img_folder=(dataset.location + '/train'), feature_extractor=feature_extractor)
    val_dataset = CocoDetection(img_folder=(dataset.location + '/valid'), feature_extractor=feature_extractor, train=False)
    test_dataset = CocoDetection(img_folder=(dataset.location + '/test'), feature_extractor=feature_extractor, train=False)
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))
    print("Number of test examples:", len(test_dataset))

    # Create dataloaders
    collator = ImageClassificationCollator(feature_extractor)
    num_workers = 12
    batch_size = int(args.batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collator, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator, shuffle=False, num_workers=num_workers)

    # Display images with their labels
    batch = next(iter(train_dataloader))
    images = batch['pixel_values']
    print('pixel_values shape: ', images[0].shape)
    labels = batch['labels']
    print('labels: ', labels)

    # Hyperparameters
    lr = float(args.lr) # 2.5e-5
    weight_decay = float(args.weight_decay) #1e-4
    max_epochs = int(args.max_epochs) #30
    gradient_clip_val = 0.1
    accumulate_grad_batches = 8
    hparams = {
        'learning_rate': lr, 
        'weight_decay': weight_decay, 
        'batch_size': batch_size,
        'max_epochs': max_epochs, 
        'gradient_clip_val': gradient_clip_val, 
        'accumulate_grad_batches': accumulate_grad_batches}

    # Define Model
    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny", 
    num_labels=len(id2label), ignore_mismatched_sizes=True)
    detector = Detector(model=model, lr=lr, weight_decay=weight_decay, batch_size=batch_size, hparams=hparams)
    # for debug
    outputs = model(pixel_values=batch['pixel_values'])
    print(outputs.logits.shape)

    # Training
    now = datetime.now()
    timestr = now.strftime("%d%m%Y_%H%M%S")
    devices = -1
    version_name = 'version_' + timestr
    logger = pl.loggers.TensorBoardLogger(
                save_dir='.',
                version=version_name,
                name='lightning_logs'
            )
    trainer = Trainer(logger=logger, accelerator='gpu', devices=devices, max_epochs=max_epochs, 
                    gradient_clip_val=gradient_clip_val, 
                    accumulate_grad_batches=accumulate_grad_batches, 
                    callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
    trainer.fit(detector, train_dataloader, val_dataloader)
    
    # Save model for inference
    model_path = os.path.join('models/yolos')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model.state_dict(), os.path.join(model_path, timestr + '_model.pth'))

    """
    tracking_address = 'lightning_logs'
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    """


# run command: python3 train_yolos.py --batch_size 4 --weight_decay 1e-4 --lr 2e-5 --max_epochs 25
# tensor board command:  tensorboard --logdir=lightning_logs