_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/chestxray_binary.py',
    '../_base_/default_runtime.py' 
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)),
    decode_head=dict(
        num_classes=1,
        threshold=.3,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True)
        )
    )

# define custom crop_size
crop_size = (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
data = dict(
    train=dict(pipeline=train_pipeline))


# optimizer (default)
#optimizer = dict(
#    _delete_=True,
#    type='AdamW',
#    lr=0.00006,
#    betas=(0.9, 0.999),
#    weight_decay=0.01,
#    paramwise_cfg=dict(
#        custom_keys={
#            'pos_block': dict(decay_mult=0.),
#            'norm': dict(decay_mult=0.),
#            'head': dict(lr_mult=10.)
#        }))

# lr cfg
#lr_config = dict(
#    _delete_=True,
#    policy='poly',
#    warmup='linear',
#    warmup_iters=1500,
#    warmup_ratio=1e-6,
#    power=1.0,
#    min_lr=0.0,
#    by_epoch=False)

# default
data = dict(samples_per_gpu=1, workers_per_gpu=1)


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

#wandb = dict(project="MedImg", entity="eeplater", name='fcn_test')
log_config = dict(
    interval=1, # interval to print log
    by_epoch=True,
    hooks=[
        dict(type='TextLoggerHook')
#        dict(type="WandbLoggerHook", init_kwargs=wandb)
    ])
# runtime settings for validating correct workflow
runner = dict(type='EpochBasedRunner', max_epochs=1) 
checkpoint_config = dict(by_epoch=True, interval=1)
evaluation = dict(interval=1, metric=['mIoU', 'mDice','mFscore'], 
                  by_epoch=True,
                  pre_eval=True)


workflow = [('train',1)] 
user = 'eplatero'
work_dir = f'/home/{user}/Downloads/LungLogs/segformer/trainval/' # path to save checkpoints (creates symlink, which is not supported by my drive)
resume_from = None
load_from = None
# CONFIG_FILE=configs/segformer/segformer_mit-b0_8x1_1024x1024_10_chestxray.py
# bash ./tools/dist_train.sh $CONFIG_FILE 1