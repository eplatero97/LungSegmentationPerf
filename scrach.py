
from mmcv.utils import Config
from mmseg.apis import train_segmentor
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.utils import build_dp, setup_multi_processes

config = r'configs/fcn/fcn_r18b-d8_512x1024_20k_chestxray.py'



cfg = Config.fromfile(config)
cfg.gpu_ids = range(1)
cfg.seed = 255

setup_multi_processes(cfg)

model = build_segmentor(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'))
#model.init_weights()

dataset = [build_dataset(cfg.data.train)]

distributed = False
validate = True
#train_segmentor(
#    model,
#    datasets,
#    cfg,
#    distributed=False,
#    validate=True)



# The default loader config
loader_cfg = dict(
    # cfg.gpus will be ignored if distributed
    num_gpus=len(cfg.gpu_ids),
    dist=distributed,
    seed=cfg.seed,
    drop_last=True)
# The overall dataloader settings
loader_cfg.update({
    k: v
    for k, v in cfg.data.items() if k not in [
        'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
        'test_dataloader'
    ]
})
# The specific dataloader settings
train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}


#from torch.utils.data import DataLoader, IterableDataset
#dl = DataLoader(dataset)
#print(len(dl))
#print(next(iter(dl)))



data_loaders = [build_dataloader(dataset, **train_loader_cfg)] # build PyTorch data loader
from mmcv.runner import IterLoader
iter_loaders = [IterLoader(data_loaders)]

print(dataset, train_loader_cfg)
print(iter_loaders)
#print(next(iter(iter_loaders[0])))
dl = next(iter_loaders[0])
print(dl)
dl_ = iter(dl)
print(dl_)
print(next(dl_))