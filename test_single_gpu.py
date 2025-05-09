import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import Random_StyleIAMDataset, ContentData, generate_type
from models.unet import UNetModel
from tqdm import tqdm
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
import torchvision
from utils.util import fix_seed

def main(opt):
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    fix_seed(cfg.TRAIN.SEED)

    load_content = ContentData()

    text_corpus = generate_type[opt.generate_type][1]
    with open(text_corpus, 'r') as _f:
        texts = _f.read().split()

    temp_texts = texts  # no need to split for single GPU

    style_dataset = Random_StyleIAMDataset(
        os.path.join(cfg.DATA_LOADER.STYLE_PATH, generate_type[opt.generate_type][0]),
        os.path.join(cfg.DATA_LOADER.LAPLACE_PATH, generate_type[opt.generate_type][0]),
        len(temp_texts)
    )

    print('Handling characters:', len(style_dataset))

    style_loader = torch.utils.data.DataLoader(
        style_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        pin_memory=True
    )

    target_dir = os.path.join(opt.save_dir, opt.generate_type)
    diffusion = Diffusion(device=opt.device)

    unet = UNetModel(
        in_channels=cfg.MODEL.IN_CHANNELS,
        model_channels=cfg.MODEL.EMB_DIM,
        out_channels=cfg.MODEL.OUT_CHANNELS,
        num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=cfg.MODEL.NUM_HEADS,
        context_dim=cfg.MODEL.EMB_DIM
    ).to(opt.device)

    if len(opt.one_dm) > 0:
        unet.load_state_dict(torch.load(f'{opt.one_dm}', map_location='cpu'))
        print('Loaded pretrained one_dm model from {}'.format(opt.one_dm))
    else:
        raise IOError('Input the correct checkpoint path')

    unet.eval()

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae").to(opt.device)
    vae.requires_grad_(False)

    loader_iter = iter(style_loader)

    for x_text in tqdm(temp_texts, position=0, desc='batch_number'):
        data = next(loader_iter)
        data_val, laplace, wid = data['style'][0], data['laplace'][0], data['wid']

        data_loader = []
        if len(data_val) > 224:
            data_loader.append((data_val[:224], laplace[:224], wid[:224]))
            data_loader.append((data_val[224:], laplace[224:], wid[224:]))
        else:
            data_loader.append((data_val, laplace, wid))

        for (data_val, laplace, wid) in data_loader:
            style_input = data_val.to(opt.device)
            laplace = laplace.to(opt.device)
            text_ref = load_content.get_content(x_text).to(opt.device)
            text_ref = text_ref.repeat(style_input.shape[0], 1, 1, 1)

            x = torch.randn((text_ref.shape[0], 4, style_input.shape[2] // 8, (text_ref.shape[1] * 32) // 8)).to(opt.device)

            if opt.sample_method == 'ddim':
                ema_sampled_images = diffusion.ddim_sample(
                    unet, vae, style_input.shape[0], x, style_input, laplace, text_ref,
                    opt.sampling_timesteps, opt.eta
                )
            elif opt.sample_method == 'ddpm':
                ema_sampled_images = diffusion.ddpm_sample(
                    unet, vae, style_input.shape[0], x, style_input, laplace, text_ref
                )
            else:
                raise ValueError('Sample method not supported')

            for index in range(len(ema_sampled_images)):
                im = torchvision.transforms.ToPILImage()(ema_sampled_images[index])
                image = im.convert("L")
                out_path = os.path.join(target_dir, wid[index][0])
                os.makedirs(out_path, exist_ok=True)
                image.save(os.path.join(out_path, x_text + ".png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='Generated',
                        help='Target dir for storing the generated characters')
    parser.add_argument('--one_dm', dest='one_dm', default='', required=True,
                        help='Pre-trained model for generating')
    parser.add_argument('--generate_type', dest='generate_type', required=True,
                        help='Four generation settings: iv_s, iv_u, oov_s, oov_u')
    parser.add_argument('--device', type=str, default='cuda', help='Device for test')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--sample_method', type=str, default='ddim', help='Choose the method for sampling')
    parser.add_argument('--eta', type=float, default=0.0)
    opt = parser.parse_args()
    main(opt)
