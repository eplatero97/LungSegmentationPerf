_base_ = ['./fcn_r50-d8_512x1024_20k_chestxray_cityscapes.py']


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1)

#wandb = dict(project="MedImg", entity="eeplater")
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False)
#        dict(type="WandbLoggerHook", init_kwargs=wandb, by_epoch=False)
        
    ])
# runtime settings for validating correct workflow
runner = dict(type='IterBasedRunner', max_iters=10)
checkpoint_config = dict(by_epoch=False, interval=10)
#evaluation = [dict(interval=1, metric='mIoU', pre_eval=True),
#              dict(interval=1, metric="mFscore", pre_eval=True)] # 1 for testing purposes, then adjust to 2000 (as was the default)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True)