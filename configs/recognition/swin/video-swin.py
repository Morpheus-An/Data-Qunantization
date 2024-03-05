_base_ = [
    'swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py'
]

model = dict(
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=101,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob')
)

dataset_type = 'VideoDataset'
data_root = 'data/ucf101/videos'
# data_root = 'data/kinetics400_tiny/train'
data_root_val = 'data/ucf101/videos'
# data_root_val = 'data/kinetics400_tiny/val'
ann_file_train = 'data/ucf101/ucf101_train_split_1_videos.txt'
# ann_file_train = 'data/kinetics400_tiny/kinetics_tiny_train_video.txt'
ann_file_val = 'data/ucf101/ucf101_val_split_1_videos.txt'
# ann_file_val = 'data/kinetics400_tiny/kinetics_tiny_val_video.txt'

train_dataloader = dict(
    dataset=dict(
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root)))
val_dataloader = dict(
    dataset=dict(
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val)))
test_dataloader = dict(
    dataset=dict(
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val)))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,  # 将 100 修改为 50
    val_begin=1,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,  # 将 100 修改为 50
        by_epoch=True,
        milestones=[20, 40],  # 修改 milestones
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.002, # 将 0.01 修改为 0.005
        momentum=0.8,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

load_from = '/home/wangkai/big_space/ant/mmaction2-main/work_dirs/video-swin/best_acc_top1_epoch_44.pth'
resume = True  # 是否从 `load_from` 中定义的权重恢复训练。如果 `load_from` 为 None，则会从 `work_dir` 中恢复最新的权重。
