import os
cur = os.getcwd()
#_base_ = os.path.join(cur,'configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco.py')
#_base_ = os.path.join(cur,'configs/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py')
_base_ = os.path.join(cur,'configs/mask_rcnn/mask_rcnn_r101_caffe_fpn_mstrain-poly_3x_coco.py')



model = dict(
    roi_head=dict(bbox_head=dict(num_classes=3),
                  mask_head=dict(num_classes=3)))

dataset_type = 'CocoDataset'     # 数据类型
classes = ('powder_uncover', 'powder_uneven', 'scratch',)           # 检测类别为balloon



number_of_sample = 2
data = dict(
    shuffle=True,
    samples_per_gpu=number_of_sample,
    workers_per_gpu=number_of_sample,
    val_dataloader=dict(shuffle=False, samples_per_gpu=number_of_sample),
    test_dataloader=dict(shuffle=False, samples_per_gpu=number_of_sample),
    train=dict(
        type=dataset_type,
        img_prefix=os.path.join(cur,'data/image/'),
        classes=classes,
        ann_file=os.path.join(cur,'data/t04_train.json')),
    val=dict(
        type=dataset_type,
        img_prefix=os.path.join(cur,'data/image/'),
        classes=classes,
        ann_file=os.path.join(cur,'data/f04_val.json')),
    test=dict(
        type=dataset_type,
        img_prefix=os.path.join(cur,'data/image/'),
        classes=classes,
        ann_file=os.path.join(cur,'data/f04_val.json')))


#load_from = os.path.join(cur,'checkpoints/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco.pth')
#load_from = os.path.join(cur,'checkpoints/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.pth')
load_from = os.path.join(cur,'checkpoints/mask_rcnn_r101_caffe_fpn_mstrain-poly_3x_coco.pth')
#load_from = os.path.join(cur,'checkpoints/mask_rcnn_r101_fpn_2x_coco.pth')

#workflow = [('train', 2), ('valid', 1)]
checkpoint_config = dict(interval=1000)
log_config = dict(interval=171) #352

#`base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=number_of_sample)

## 300 data