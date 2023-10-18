import mmcv
from itertools import groupby
from pycocotools import mask as mutils
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image                                      # (pip install Pillow)
import numpy as np                                         # (pip install numpy)
import os
import json
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
from mmcv import Config
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

def dice_coefficient(seg, gt):
    return np.sum(seg[gt==1]==1)*2.0 / (np.sum(seg[seg==1]==1) + np.sum(gt[gt==1]==1))

def dice_coefficient_each(seg, gt):
    return np.sum(seg[gt==1]==1)*2.0 / (np.sum(seg[gt==1]==1) + np.sum(gt[gt==1]==1))

def color_val_matplotlib(color):
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)

def show_result(img,
            result,
            classname=None,
            score_thr=0.7,
            bbox_color=(72, 101, 241),
            text_color=(72, 101, 241),
            mask_color=(255, 0, 0),
            thickness=2,
            font_size=13,
            win_name='',
            show=True,
            wait_time=0,
            out_file=None):
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        segms = np.stack(segms, axis=0)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms,
        class_names=classname,
        score_thr=score_thr,
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    

def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=13,
                      win_name='',
                      show=True,
                      wait_time=0,
                      out_file=None):
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:
            # Get random state before set seed, and restore random state later.
            # Prevent loss of randomness.
            # See: https://github.com/open-mmlab/mmdetection/issues/5844
            state = np.random.get_state()
            # random color
            np.random.seed(42)
            mask_colors = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            np.random.set_state(state)
        else:
            # specify  color
            mask_colors = [
                np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
            ] * (
                max(labels) + 1)

    bbox_color = color_val_matplotlib(bbox_color)
    text_color = color_val_matplotlib(text_color)

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + 1e-2) / dpi, (height + 1e-2) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    polygons = []
    color = []
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(bbox_color)
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        ax.text(
            bbox_int[0],
            bbox_int[1],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size,
            verticalalignment='top',
            horizontalalignment='left')
        if segms is not None:
            color_mask = mask_colors[labels[i]]
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    plt.imshow(img)

    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    ax.add_collection(p)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    #if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
    #    if wait_time == 0:
    #        plt.show()
    #    else:
    #        plt.show(block=False)
    #        plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img


cur = os.getcwd()
annFile = os.path.join(cur,'data/f01_val.json')
filterClasses=['powder_uncover'] # , 'powder_uneven', 'scratch'
config_file = os.path.join(cur, 'configs/myconfig.py')
checkpoint_file = os.path.join(cur, 'work_dirs/myconfig/f04_best_bbox_mAP_50_epoch_90_0.956_0.969.pth')
image_root = os.path.join(cur,'data/f01/image/')
path = 'result/output.txt'

# Initialize the COCO api for instance annotations
coco=COCO(annFile)
# Load the categories in a variable
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)
# Fetch class IDs only corresponding to the filterClasses
catIds = coco.getCatIds(catNms=filterClasses)
# Get all images containing the above Category IDs
imgIds = coco.getImgIds(catIds=catIds)
print("Number of images containing all the  classes:", len(imgIds))
test_cfg = Config.fromfile(config_file)


f = open(path, 'w')
for i in imgIds:
    img = coco.loadImgs(i)[0]
    
    img_path = str(os.path.join(image_root, img['file_name']))  # or img = mmcv.imread(img), which will only load it once
    img_mmcv = mmcv.imread(os.path.join(image_root, img['file_name'])).astype(np.uint8)
    I = Image.open(img_path).convert("RGB")

    model = init_detector(test_cfg, checkpoint_file, device='cuda:0')
    result = inference_detector(model, img_path)

    spine_class_name = model.CLASSES
    print(list(spine_class_name))


#print("Original image:")
#plt.imshow(I)
#plt.axis('off')

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    gt_mask = coco.annToMask(anns[0])
    coco.showAnns(anns, draw_bbox=False)

    bbox_result, segm_result = result
    show_result(str(os.path.join(image_root, img['file_name'])), bbox_result, score_thr=0.5)
    print("Original image with prediction bbox:")
    show_result(str(os.path.join(image_root, img['file_name'])), bbox_result, score_thr=0.5, out_file='result/bbox'+img['file_name'])
    print("Original image with prediction segmentation:")
    show_result( str(os.path.join(image_root, img['file_name'])), result, score_thr=0.5, out_file='result/'+img['file_name'])

    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns, draw_bbox=False)
# Load and display instance annotations
#show_result( str(os.path.join(image_root, img['file_name'])), result, score_thr=0.7, mask_color=(0, 0, 255))

    print("Number of vertebra ::")
    print(f"GT : {len(anns)}")
    Detected = 0
    bbox_result_filtered = []
    segm_result_filtered = []
    index = 0

    for bbox_list in bbox_result[0]:
        if bbox_list[-1] > 0.5:
            bbox_result_filtered.append(bbox_list)
            segm_result_filtered.append(segm_result[0][index].astype(int))
            Detected += 1
            index += 1

    print(f"Detected : {Detected}")
    print("DC :: ")

    result = np.zeros(gt_mask.shape)
    for area in segm_result_filtered:
        #print(area)
        result += area

    dc_sum = 0
    for i in range(0, len(anns)):
        rt_map = coco.annToMask(anns[i])
        dc_each = dice_coefficient_each(result, rt_map)
        dc_sum += dc_each
        print(f'V{i}: {dc_each}')

    dc_sum /= len(anns)

# Show DC of Ground Truth and the result
    gt_mask = coco.annToMask(anns[0])
    rt_map = np.zeros(gt_mask.shape)
    for rt in anns:
        rt_map += coco.annToMask(rt)
    dc = dice_coefficient(result, rt_map)
    print(f"Average : {dc_sum}")
    print(f"whole : {dc}")
    
    f.writelines("Number of vertebra ::\n")
    f.writelines(f"GT : {len(anns)}\n")
    f.writelines(f"Detected : {Detected}\n")
    f.writelines("DC :: \n")
    for i in range(0, len(anns)):
        rt_map = coco.annToMask(anns[i])
        dc_each = dice_coefficient_each(result, rt_map)
        f.writelines(f'V{i}: {dc_each}\n')
    f.writelines(f"Average : {dc_sum}\n")
    f.writelines(f"whole : {dc}\n")
f.close()
