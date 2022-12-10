_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/chestxray_binary.py',
    '../_base_/default_runtime.py'
]
model = dict(
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
        num_classes=1,
        threshold=.3,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True)

    ),
    auxiliary_head=dict(
        in_channels=256, 
        channels=64,
        num_classes=1,
        threshold=.3,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True)
    )
        )


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

wandb = dict(project="MedImg", entity="eeplater", name='fcn_test')
log_config = dict(
    interval=1, # interval to print log
    by_epoch=True,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type="WandbLoggerHook", init_kwargs=wandb)
    ])
# runtime settings for validating correct workflow
runner = dict(type='EpochBasedRunner', max_epochs=1) 
checkpoint_config = dict(by_epoch=True, interval=1)
evaluation = dict(interval=1, metric=['mIoU', 'mDice','mFscore'], 
                  by_epoch=True,
                  pre_eval=True)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1
)
workflow = [('train',1)] 
user = 'erick'
work_dir = f'/home/{user}/Downloads/LungLogs/pspnet/trainval/' # path to save checkpoints (creates symlink, which is not supported by my drive)
resume_from = None
load_from = None
# CONFIG_FILE=configs/pspnet/pspnet_r18b-d8_512x1024_10_chestxray.py
# bash ./tools/dist_train.sh $CONFIG_FILE 1