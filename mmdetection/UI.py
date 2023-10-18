from tkinter import *
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import os
import time
import json
import numpy as np
from mmdet.apis import (inference_detector, init_detector)

## verify AP50 IoU correctness

img_folder = ''
img_filename = ''
cur = os.getcwd()
checkpoint_file = os.path.join(cur, 'work_dirs/myconfig/f04_best_bbox_mAP_50_epoch_90_0.956_0.969.pth')
config_file = os.path.join(cur, 'configs/myconfig.py')
model = init_detector(config_file, checkpoint_file, device='cuda:0')
cat = ['powder_uncover', 'powder_uneven', 'scratch']
total_img_list = []
fps = 0.
gt_cat = ''
pre_cat = ''
iou = 0.
dice = 0.
a, b, c, d, e = '', '', '', '', ''
ap = [0., 0., 0.]

def imgfolder():
    global img_folder
    img_folder = filedialog.askdirectory(initialdir = "/",title = "Select directory")
    print(img_folder)
    AP50_FPS()
    

def imgfile():
    global img_filename
    img_filename =  filedialog.askopenfilename(initialdir = img_folder, title = "Select file", filetypes = (("png files","*.png"),("all files","*.*")))
    print(img_filename)
    imgshow()
    

def imgshow():
    global gt_cat
    gt_cat = img_filename.split('/')[-3]   ## GT category
    img_name = img_filename.split('/')[-1]
    img = Image.open(img_filename)
    img = img.resize((300, 300), Image.Resampling.LANCZOS)
    tk_img = ImageTk.PhotoImage(img)
    imageLabel.configure(image = tk_img)
    imageLabel.image = tk_img
    textLabel.configure(text='Original Image')
    #textLabel.text = 'Original Image'
    global a, b
    a = f"Current Image: {total_img_list.index(img_name)+1}/{len(total_img_list)}\n"
    b = f"Type(GT): {gt_cat} \n"
    textlabel3.configure(text=a+e+b+c+d)
    #textlabel3.text = f"{gt_cat}"

def AP50_FPS():
    totaltime = 0
    for i in cat:
        totalap = 0
        img_list = os.listdir(os.path.join(img_folder, f'{i}/image'))
        global total_img_list
        total_img_list.extend(img_list)
        for j in img_list:
            sec = time.time()
            bbox_result, segm_result = inference_detector(model, os.path.join(img_folder, f'{i}/image', j))
            totaltime += time.time() - sec
            totalap += AP50(bbox_result, os.path.join(img_folder, f'{i}/label', j[:-4]) + '.json', cat.index(i))
        totalap /= len(img_list)
        print(f'{i}: {totalap}')
        global ap
        ap[cat.index(i)] = totalap
    totaltime = 60 / (totaltime / len(img_list))

    global fps
    fps = totaltime
    print(f"FPS: {totaltime}")
    global d, e
    e = f"FPS: {round(fps, 3)}\n"
    d = f"Folder(Mean): {round((ap[0] + ap[1] + ap[2])/3, 3)}\n \
                Evaluation Metric\n \
                AP50(uncover): {round(ap[0], 3)}\n \
                AP50(uneven): {round(ap[1], 3)}\n \
                AP50(scratch): {round(ap[2], 3)}\n"
    textlabel3.configure(text=a+e+b+c+d)


def AP50(bbox_result, label_path, category):
    bbox_list = []
    if bbox_result[category].size == 0:
        return 0
    for i in bbox_result[category]:
        if i[-1] > 0.5:
            bbox_list.append(i[:-1])
    gt_bbox = []
    with open(label_path) as f:
        data = json.load(f)
        for i in data['shapes']:
            if i['points'][0][1] > i['points'][1][1]:
                gt_bbox.append([i['points'][0][0], i['points'][1][1], i['points'][1][0], i['points'][0][1]])
            else:
                gt_bbox.append(i['points'][0] + i['points'][1])
    tp = 0
    fp = 0
    for i in gt_bbox:
        tmp = get_max_iou(np.array(bbox_list), i)
        if tmp < 0.5:
            fp += 1
        else:
            tp += 1
    if len(bbox_list) - len(gt_bbox) > 0:
        fp += len(bbox_list) - len(gt_bbox)
    
    #print(tp / (tp + fp))
    return tp / (tp + fp)

def get_max_iou(pred_boxes, gt_box):
    if pred_boxes.shape[0] > 0:
        ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
        ixmax = np.minimum(pred_boxes[:, 2], gt_box[2])
        iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
        iymax = np.minimum(pred_boxes[:, 3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

        inters = iw * ih

        uni = ((pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) +
               (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
               inters)

        iou = inters / uni
        iou_max = np.max(iou)
        #nmax = np.argmax(iou)
        return iou_max

def IoU_cat():
    img_name = img_filename.split('/')[-1]
    bbox_result, segm_result = inference_detector(model, img_filename)

    global pre_cat
    ss = 0
    for i in range(len(bbox_result)):
        if bbox_result[i].size != 0 and bbox_result[i].size > ss:
            pre_cat = cat[i]
        ss = bbox_result[i].size
    bbox_list = []
    #if bbox_result[category].size == 0:
    #    return 0
    category = cat.index(pre_cat)
    for i in bbox_result[category]:
        if i[-1] > 0.5:
            bbox_list.append(i[:-1])
    gt_bbox = []
    str1 = "/"
    with open(os.path.join(str1.join(img_filename.split('/')[:-2]), 'label', img_name[:-4]) + '.json') as f:
        data = json.load(f)
        for i in data['shapes']:
            if i['points'][0][1] > i['points'][1][1]:
                gt_bbox.append([i['points'][0][0], i['points'][1][1], i['points'][1][0], i['points'][0][1]])
            else:
                gt_bbox.append(i['points'][0] + i['points'][1])
    global iou
    iou = 0
    for i in gt_bbox:
        iou += get_max_iou(np.array(bbox_list), i)
    iou /= len(gt_bbox)
    print(f'IoU: {iou}')
    global c
    c = f"Predict: {pre_cat} \n \
            IoU: {round(iou, 3)}\n \
            Dice Coefficient: {round(dice, 3)}\n"
    textlabel3.configure(text=a+e+b+c+d)
    return bbox_list

def dice_cat():
    img_name = img_filename.split('/')[-1]
    bbox_result, segm_result = inference_detector(model, img_filename)

    global pre_cat
    ss = 0
    for i in range(len(bbox_result)):
        if bbox_result[i].size != 0 and bbox_result[i].size > ss:
            pre_cat = cat[i]
        ss = bbox_result[i].size
    #if len(segm_result[category]) == 0:
    #    return 0
    category = cat.index(pre_cat)
    w, h = segm_result[category][0].shape
    bmp = []
    bmp.extend([[False]*h]*w)
    for i in segm_result[category]:
        bmp |= i
    bmp = np.array(bmp)
    str1 = "/"
    print(str1.join(img_filename.split('/')[:-2]))
    gt_bmp = Image.open(os.path.join(str1.join(img_filename.split('/')[:-2]), 'mask', img_name))
    gt_bmp = np.array(gt_bmp, dtype=bool)
    global dice
    dice = 2 * np.count_nonzero(bmp & gt_bmp) / (np.count_nonzero(bmp & gt_bmp) + np.count_nonzero(bmp | gt_bmp))
    print(f'dice: {dice}')
    global c
    c = f"Predict: {pre_cat} \n \
            IoU: {round(iou, 3)}\n \
            Dice Coefficient: {round(dice, 3)}\n"
    textlabel3.configure(text=a+e+b+c+d)
    return bmp


def inference_bbox():
    bbox_list = IoU_cat()
    img = Image.open(img_filename).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i in bbox_list:
        draw.rectangle(((i[0], i[1]), (i[2], i[3])), width = 5, outline = "red", fill = None)
    img = img.resize((300, 300), Image.Resampling.LANCZOS)
    tk_img = ImageTk.PhotoImage(img)
    imageLabel1.configure(image = tk_img)
    imageLabel1.image = tk_img
    textLabel1.configure(text='Detection Result')
    #textLabel1.text = 'Detection Result'


def inference_seg():
    bmp = dice_cat()
    img = Image.fromarray(bmp)
    img = img.resize((300, 300), Image.Resampling.LANCZOS)
    tk_img = ImageTk.PhotoImage(img)
    imageLabel2.configure(image = tk_img)
    imageLabel2.image = tk_img
    textLabel2.configure(text='Segmentation Result')
    #textLabel2.text = 'Segmentation Result'



if __name__ == '__main__':
    root = Tk()
    root.title('FinalProject')
    root.geometry('1500x500')

    frame = tk.Frame(root, width=300, height=300)
    frame.pack(side=LEFT, anchor=tk.N)
    frame1 = tk.Frame(root, width=300, height=300)
    frame1.pack(side=LEFT, anchor=tk.N)
    frame2 = tk.Frame(root, width=300, height=300)
    frame2.pack(side=LEFT, anchor=tk.N)


    textLabel = tk.Label(frame, font=('Arial', 16))
    textLabel.pack()
    textLabel1 = tk.Label(frame1, font=('Arial', 16))
    textLabel1.pack()
    textLabel2 = tk.Label(frame2, font=('Arial', 16))
    textLabel2.pack()


    imageLabel = tk.Label(frame)
    imageLabel.pack()
    imageLabel1 = tk.Label(frame1)
    imageLabel1.pack()
    imageLabel2 = tk.Label(frame2)
    imageLabel2.pack()

    a = f"Current Image: \n"
    e = f"FPS: {round(fps, 3)}\n"
    b = f"Type(GT): {gt_cat} \n"
    c = f"Predict: {pre_cat} \n \
            IoU: {iou}\n \
            Dice Coefficient: {dice}\n"

    d = f"Folder(Mean): {(ap[0] + ap[1] + ap[2])/3}\n \
                Evaluation Metric\n \
                AP50(uncover): {ap[0]}\n \
                AP50(uneven): {ap[1]}\n \
                AP50(scratch): {ap[2]}\n"

    textlabel3 = tk.Label(root, text=a+e+b+c+d, font=('Arial', 16))  # 放入 Label
    textlabel3.pack(anchor=tk.NE)

    button3 = tk.Button(root, text='Segment', bg='yellow', font=('Arial', 16), command=inference_seg).pack(side=BOTTOM, anchor=tk.SE)
    button2 = tk.Button(root, text='Detect defects', bg='yellow', font=('Arial', 16), command=inference_bbox).pack(side=BOTTOM, anchor=tk.SE)
    button1 = tk.Button(root, text='Select Image', bg='yellow', font=('Arial', 16), command=imgfile).pack(side=BOTTOM, anchor=tk.SE)
    button = tk.Button(root, text='Select Image Folder', bg='yellow', font=('Arial', 16), command=imgfolder).pack(side=BOTTOM, anchor=tk.SE)


    root.mainloop()
