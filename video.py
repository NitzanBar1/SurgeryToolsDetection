# -*- coding: utf-8 -*-
import os
import torch
import pytorch_lightning as pl
from transformers import AutoFeatureExtractor
from transformers import AutoModelForObjectDetection
import torchvision
import cv2
import numpy as np
import bbox_visualizer as bbv
import argparse
import torchmetrics
from sklearn import metrics


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
    parser = argparse.ArgumentParser('smoothing process')
    parser.add_argument('--video_file', required=True)
    parser.add_argument('--num_previous_frames', required=True)
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

    cap = cv2.VideoCapture(os.path.join('videos', args.video_file))

    file_name = os.path.splitext(args.video_file)[0] + '.txt'
    # right
    with open(os.path.join('/home/student/HW1_dataset/HW1_dataset/tool_usage/tools_right', file_name)) as f:
        tool_usage_right = []
        for x in f:
            tool_usage_right.append(list(map(lambda s: s, x.split())))
    # left
    with open(os.path.join('/home/student/HW1_dataset/HW1_dataset/tool_usage/tools_left', file_name)) as f:
        tool_usage_left = []
        for x in f:
            tool_usage_left.append(list(map(lambda s: s, x.split())))

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # Define the fps to be equal to 10. Also frame size is passed.
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_video = os.path.splitext(args.video_file)[0] + '.avi'
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

    # colors for visualization
    colors = [[0, 128, 255], [255, 0, 0], [255, 128, 0],
    [128, 0, 128], [128, 255, 0], [0, 0, 255]]

    #id2label = {0: 'Right_Scissors',
    #            1: 'Left_Scissors',
    #            2: 'Right_Needle_driver',
    #            3: 'Left_Needle_driver',
    #            4: 'Right_Forceps',
    #            5: 'Left_Forceps',
    #            6: 'Right_Empty',
    #            7: 'Left_Empty'}

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


    tool_usage = {'T0': "no tool in hand",
                  'T1': "needle_driver",
                  'T2': "forceps",
                  'T3': "scissors"}

    tool_label_right = {'T0': 5,
                        'T1': 6,
                        'T2': 0,
                        'T3': 7}

    tool_label_left = {'T0': 1,
                       'T1': 3,
                       'T2': 2,
                       'T3': 4}
    # Read until video is completed
    i=0
    tool_usage_left_preds = []
    tool_usage_right_preds = []
    tool_usage_left_targets = []
    tool_usage_right_targets = []
    tools_hist_left = [0]*8
    tools_hist_right = [0]*8
    while (cap.isOpened()):
        i += 1
        # every #num_previous_frames frames - reset the tool usage histogram 
        if (i % int(args.num_previous_frames) == 0):
            tools_hist_left = [0]*8
            tools_hist_right = [0]*8

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # apply featue extractor on input image
            pixel_values = feature_extractor(frame, return_tensors="pt")['pixel_values']
            pixel_values = pixel_values.to(device)
            frame_size = torch.zeros(1,2)
            frame_size[0,0] = frame.shape[0]
            frame_size[0,1] = frame.shape[1]
            frame_size = frame_size.to(device)

            # predict bounding boxes
            outputs = model(pixel_values=pixel_values)
            
            # convert outputs of model to COCO api
            results = feature_extractor.post_process(outputs, frame_size) 

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
                frame = bbv.draw_rectangle(frame, box, bbox_color=[0,0,255])
                if cl.item() >=1 and cl.item() <=4: # left hand  
                    tools_hist_left[cl.item()] += 1
                    idx = tools_hist_left.index(max(tools_hist_left))
                    tool_usage_left_preds.append(idx)
                    tool_left = check_current_tool(tool_usage_left, i) 
                    tool_usage_left_targets.append(tool_label_left[tool_left])
                else: # right hand
                    tools_hist_right[cl.item()] += 1
                    idx = tools_hist_right.index(max(tools_hist_right))
                    tool_usage_right_preds.append(idx)
                    tool_right = check_current_tool(tool_usage_right, i)
                    tool_usage_right_targets.append(tool_label_right[tool_right]) 

                text = id2label[idx] + ': ' + f'{p.item():.3f}'
                #text = id2label[cl.item()] + ': ' + f'{p.item():.3f}'
                frame = bbv.add_T_label(frame, text, box, text_bg_color=[0,0,255])

            # Add tool usage text (ground truth)
            tool_right = check_current_tool(tool_usage_right, i)
            tool_left = check_current_tool(tool_usage_left, i) 
            text_right = 'Right: ' + tool_usage[tool_right]
            text_left = 'Left: ' + tool_usage[tool_left]
            cv2.putText(img=frame, text=text_right, org=(10, 30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0), thickness=1)
            cv2.putText(img=frame, text=text_left, org=(10, 70), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0), thickness=1)

            """
            # Display the resulting frame
            if not(os.path.exists(os.path.join('output_frames', args.video_file))):
                os.makedirs(os.path.join('output_frames', args.video_file))
            cv2.imwrite(os.path.join('output_frames', args.video_file, 'Frame' + str(i)+ '.jpg'), frame)
            """

            # Write the frame into the file 'output.avi'
            out.write(frame)

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    out.release()

    # Evaluation merics - tool usage
    preds  = torch.tensor(tool_usage_right_preds + tool_usage_left_preds)
    target = torch.tensor(tool_usage_right_targets + tool_usage_left_targets)

    F1 = torchmetrics.F1Score(task="multiclass", num_classes=8, average='macro')
    print('Macro F1 Score: ', round(F1(preds, target).item(), 2))

    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=8)
    print('Accuracy: ', round(accuracy(preds, target).item(), 2))

    print(metrics.classification_report(target, preds, digits=3))


# Run Command: python3 video.py --video_file P025_tissue2.wmv --num_previous_frames 100 --model_path models/yolos/08122022_143316_model.pth