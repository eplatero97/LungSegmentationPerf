_base_ = ['./fcn_r50-d8_512x1024_80k_cityscapes.py']
model = dict(
#    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
        num_classes=2,
    ),
    auxiliary_head=dict(
        num_classes=2,
        in_channels=256,
        channels=64
    ))

wandb = dict(project="MedImg", entity="eeplater")
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type="WandbLoggerHook", init_kwargs=wandb, by_epoch=False)
        
    ])
# 20k optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = [dict(interval=10, metric='mIoU', pre_eval=True),
              dict(interval=10, metric="mFscore", pre_eval=True)] # 10 for testing purposes, then adjust to 2000 (as was the default)
meta = dict(CLASSES = ('lung', 'not lung'))