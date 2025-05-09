import argparse
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import fix_seed, load_specific_dict
from utils.logger import set_log
from data_loader.loader import IAMDataset
import torch
from trainer.trainer import Trainer
from models.unet import UNetModel
from torch import optim
import torch.nn as nn
from models.diffusion import Diffusion, EMA
import copy
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from models.loss import SupConLoss


def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)
    """ prepare log file """
    logs = set_log(cfg.OUTPUT_DIR, opt.cfg_file, opt.log_name)

    """ set device """
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    """ set dataset"""
    train_dataset = IAMDataset(
        cfg.DATA_LOADER.IAMGE_PATH, cfg.DATA_LOADER.STYLE_PATH, cfg.DATA_LOADER.LAPLACE_PATH, cfg.TRAIN.TYPE)
    print('number of training images: ', len(train_dataset))
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.IMS_PER_BATCH,
                              drop_last=False,
                              collate_fn=train_dataset.collate_fn_,
                              num_workers=cfg.DATA_LOADER.NUM_THREADS,
                              pin_memory=True)

    test_dataset = IAMDataset(
        cfg.DATA_LOADER.IAMGE_PATH, cfg.DATA_LOADER.STYLE_PATH, cfg.DATA_LOADER.LAPLACE_PATH, cfg.TEST.TYPE)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.TEST.IMS_PER_BATCH,
                             drop_last=False,
                             collate_fn=test_dataset.collate_fn_,
                             pin_memory=True,
                             num_workers=cfg.DATA_LOADER.NUM_THREADS)

    """build model architecture"""
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM,
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
                     attention_resolutions=(1, 1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS,
                     context_dim=cfg.MODEL.EMB_DIM).to(device)

    """load pretrained one_dm model"""
    if len(opt.one_dm) > 0:
        unet.load_state_dict(torch.load(opt.one_dm, map_location=torch.device('cpu')))
        print('load pretrained one_dm model from {}'.format(opt.one_dm))

    """load pretrained resnet18 model"""
    if len(opt.feat_model) > 0:
        checkpoint = torch.load(opt.feat_model, map_location=torch.device('cpu'))
        checkpoint['conv1.weight'] = checkpoint['conv1.weight'].mean(1).unsqueeze(1)
        miss, unexp = unet.mix_net.Feat_Encoder.load_state_dict(checkpoint, strict=False)
        assert len(unexp) <= 32, "failed to load the pretrained model"
        print('load pretrained model from {}'.format(opt.feat_model))

    """build criterion and optimizer"""
    criterion = dict(nce=SupConLoss(contrast_mode='all'), recon=nn.MSELoss())
    optimizer = optim.AdamW(unet.parameters(), lr=cfg.SOLVER.BASE_LR)

    diffusion = Diffusion(device=device, noise_offset=opt.noise_offset)

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    """Freeze vae and text_encoder"""
    vae.requires_grad_(False)
    vae = vae.to(device)

    """build trainer"""
    trainer = Trainer(diffusion, unet, vae, criterion, optimizer, train_loader, logs, test_loader, device)
    trainer.train()


if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='path to stable diffusion')
    parser.add_argument('--cfg', dest='cfg_file', default='configs/dss.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--feat_model', dest='feat_model', default='', help='pre-trained resnet18 model')
    parser.add_argument('--one_dm', dest='one_dm', default='', help='pre-trained one_dm model')
    parser.add_argument('--log', default='debug',
                        dest='log_name', required=False, help='the filename of log')
    parser.add_argument('--noise_offset', default=0, type=float, help='control the strength of noise')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    opt = parser.parse_args()
    main(opt)
