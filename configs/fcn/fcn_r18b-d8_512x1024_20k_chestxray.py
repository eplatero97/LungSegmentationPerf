_base_ = ['./fcn_r50-d8_512x1024_80k_cityscapes.py', '../_base_/schedules/schedule_20k.py']
model = dict(
#    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
        num_classes=2,
    ),
    auxiliary_head=dict(
        num_classes=2
    ),
    auxiliary_head=dict(in_channels=256, channels=64))

wandb = dict(project="MedImg", entity="eeplater")
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type="WandbLoggerHook", init_kwargs=wandb, by_epoch=False)
        
    ])
evaluation = [dict(interval=2000, metric='mIoU', pre_eval=True),
              dict(interval=2000, metric="mFscore", pre_eval=True)]
evaluation = dict(interval=10, metric='mIoU', pre_eval=True) # 10 for testing purposes, then adjust