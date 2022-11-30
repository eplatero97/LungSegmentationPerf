_base_ = ['./fcn_r50-d8_512x1024_20k_chestxray_binary.py']
model = dict(
#    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18),
    decode_head=dict(
        in_channels=512,
        threshold=.3,
        channels=128,
        num_classes=1,
    ),
    auxiliary_head=dict(
        num_classes=1,
        threshold=.3,
        in_channels=256,
        channels=64
    ))

#wandb = dict(project="MedImg", entity="eeplater")
log_config = dict(
    interval=1, # interval to print log
    by_epoch=True,
    hooks=[
        dict(type='TextLoggerHook')
#        dict(type="WandbLoggerHook", init_kwargs=wandb)
    ])
# runtime settings for validating correct workflow
runner = dict(type='EpochBasedRunner', max_epochs=1, max_iters=None) 
checkpoint_config = dict(by_epoch=True, interval=1)
evaluation = dict(interval=1, metric=['mIoU', 'mDice','mFscore'], 
                  by_epoch=True,
                  pre_eval=True)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1
)
workflow = [('train',1)] 
work_dir = '/home/erick/Downloads/LungLogs/trainval/' # path to save checkpoints (creates symlink, which is not supported by my drive)
resume_from = None
load_from = None