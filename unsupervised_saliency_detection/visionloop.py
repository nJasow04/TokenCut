# |<---------------     gradio_inpaint.py imports    --------------->|
import sys
sys.path.append("../../..")

from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random


from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

# |<---------------     singleimage.py imports    --------------->|

sys.path.remove("../../..")
sys.path.append('./model')
import dino # model

import object_discovery as tokencut
import argparse
import utils
import bilateral_solver
import os

from shutil import copyfile
import PIL.Image as Image
import cv2
from tqdm import tqdm

from torchvision import transforms
import metric
import skimage

print("Passed Imports")

# |<-----------    singleimage.py     ---------->|

# Image transformation applied to all images
ToTensor = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225)),])

def get_tokencut_binary_map(img_pth, backbone,patch_size, tau, resize) :
    I = Image.open(img_pth).convert('RGB')
    # print(resize)
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I, patch_size, resize)

    tensor = ToTensor(I_resize).unsqueeze(0).cuda('cuda:1')
    # print(tensor.shape)
    feat = backbone(tensor)[0]

    seed, bipartition, eigvec = tokencut.ncut(feat, [feat_h, feat_w], [patch_size, patch_size], [h,w], tau)
    
    threshold_percentage = 0.05
    is_significant = np.mean(bipartition > 0.99) >= threshold_percentage
    
    return bipartition, eigvec, is_significant

def mask_color_compose(org, mask, mask_color = [173, 216, 230]) :

    mask_fg = mask > 0.5
    rgb = np.copy(org)
    rgb[mask_fg] = (rgb[mask_fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)

    return Image.fromarray(rgb)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## input / output dir
parser.add_argument('--out-dir', type=str, default='Test', help='output directory')

parser.add_argument('--vit-arch', type=str, default='small', choices=['base', 'small'], help='which architecture')

parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')

parser.add_argument('--patch-size', type=int, default=16, choices=[16, 8], help='patch size')

parser.add_argument('--tau', type=float, default=0.2, help='Tau for tresholding graph')

parser.add_argument('--sigma-spatial', type=float, default=16, help='sigma spatial in the bilateral solver')

parser.add_argument('--sigma-luma', type=float, default=16, help='sigma luma in the bilateral solver')

parser.add_argument('--sigma-chroma', type=float, default=8, help='sigma chroma in the bilateral solver')


parser.add_argument('--dataset', type=str, default=None, choices=['ECSSD', 'DUTS', 'DUT', None], help='which dataset?')

parser.add_argument('--nb-vis', type=int, default=1, choices=[1, 200], help='nb of visualization')

parser.add_argument('--img-path', type=str, default=None, help='single image visualization')

parser.add_argument('--resize', type=int, nargs='+', default=[256, 256], help='specify input resolution')

## args declared
args = parser.parse_args()
print (args)

## feature net

if args.vit_arch == 'base' and args.patch_size == 16:
    url = "/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    feat_dim = 768
elif args.vit_arch == 'base' and args.patch_size == 8:
    url = "/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    feat_dim = 768
elif args.vit_arch == 'small' and args.patch_size == 16:
    url = "/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    feat_dim = 384
elif args.vit_arch == 'base' and args.patch_size == 8:
    url = "/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"

backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)
#    resume_path = './model/dino_vitbase16_pretrain.pth' if args.patch_size == 16 else './model/dino_vitbase8_pretrain.pth'

#    feat_dim = 768
#    backbone = dino.ViTFeat(resume_path, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)
#
#else :
#    resume_path = './model/dino_deitsmall16_pretrain.pth' if args.patch_size == 16 else './model/dino_deitsmall8_pretrain.pth'
#    feat_dim = 384
#    backbone = dino.ViTFeat(resume_path, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)


msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
print (msg)
backbone.eval()
backbone.cuda('cuda:1')

# Probably don't need this
if args.dataset == 'ECSSD' :
    args.img_dir = '../datasets/ECSSD/img'
    args.gt_dir = '../datasets/ECSSD/gt'
elif args.dataset == 'DUTS' :
    args.img_dir = '../datasets/DUTS_Test/img'
    args.gt_dir = '../datasets/DUTS_Test/gt'
elif args.dataset == 'DUT' :
    args.img_dir = '../datasets/DUT_OMRON/img'
    args.gt_dir = '../datasets/DUT_OMRON/gt'
elif args.dataset is None :
    args.gt_dir = None
print(args.dataset)


if args.out_dir is not None and not os.path.exists(args.out_dir) :
    os.mkdir(args.out_dir)

# Image path: img_list
if args.img_path is not None:
    args.nb_vis = 1
    img_list = [args.img_path]
else:
    img_list = sorted(os.listdir(args.img_dir))

def tokencutting (img_path):
    # Image path: img_list
    img_list = [img_path]
    
    mask_lost = []
    mask_bfs = []
    gt = []
    for img_name in tqdm(img_list) :
        img_pth = img_name
        img_name = img_name.split("/")[-1]
        print(img_name)
        
        # print(img_pth)
        bipartition, eigvec, is_significant = get_tokencut_binary_map(img_pth, backbone, args.patch_size, args.tau, args.resize)
        mask_lost.append(bipartition)
        
        if not is_significant:
            return None

        output_solver, binary_solver = bilateral_solver.bilateral_solver_output(img_pth, bipartition, sigma_spatial = args.sigma_spatial, sigma_luma = args.sigma_luma, sigma_chroma = args.sigma_chroma, resize=args.resize)
        mask1 = torch.from_numpy(bipartition).cuda('cuda:1')
        mask2 = torch.from_numpy(binary_solver).cuda('cuda:1')
        if metric.IoU(mask1, mask2) < 0.5:
            binary_solver = binary_solver * -1
        mask_bfs.append(output_solver)

        if args.gt_dir is not None :
            mask_gt = np.array(Image.open(os.path.join(args.gt_dir, img_name.replace('.jpg', '.png'))).convert('L'))
            gt.append(mask_gt)


        print(f'args.out_dir: {args.out_dir}, img_name: {img_name}')
        out_name = os.path.join(args.out_dir, img_name)
        # out_lost = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut.jpg'))
        out_bfs = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut_bfs.jpg'))
        #out_eigvec = os.path.join(args.out_dir, img_name.replace('.jpg', '_tokencut_eigvec.jpg'))

        copyfile(img_pth, out_name)
        # org = np.array(Image.open(img_pth).convert('RGB'))
        
        img_temp = Image.open(img_pth).convert('RGB') 
        
        org = np.array(img_temp)
        
        binary_solver_image = Image.fromarray(binary_solver)
        resized_binary_solver = binary_solver_image.resize(img_temp.size, Image.NEAREST)
        binary_solver_resized = np.array(resized_binary_solver)

        mask_color_compose(org, binary_solver_resized).save(out_bfs)
        if args.gt_dir is not None :
            out_gt = os.path.join(args.out_dir, img_name.replace('.jpg', '_gt.jpg'))
            mask_color_compose(org, mask_gt).save(out_gt)
    
    return binary_solver_resized





# |<-----------   gradio_inpaint.py   ---------->|

model_name = 'control_v11p_sd15_inpaint'
device = torch.device("cuda:0")

model = create_model(f'../../../models/{model_name}.yaml')
print("Passes Model name")
model.load_state_dict(load_state_dict('../../../models/v1-5-pruned.ckpt'), strict=False)
print("Passes Model name")
model.load_state_dict(load_state_dict(f'../../../models/{model_name}.pth'), strict=False)
print("Passes Model name")
model.to(device)
print("Passes Model name")
ddim_sampler = DDIMSampler(model)


def process(input_image_and_mask, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, mask_blur):
    with torch.no_grad():
        input_image = HWC3(input_image_and_mask['image'])
        input_mask = input_image_and_mask['mask']

        img_raw = resize_image(input_image, image_resolution).astype(np.float32)
        H, W, C = img_raw.shape

        mask_pixel = cv2.resize(input_mask, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mask_pixel = cv2.GaussianBlur(mask_pixel, (0, 0), mask_blur)

        mask_latent = cv2.resize(mask_pixel, (W // 8, H // 8), interpolation=cv2.INTER_AREA)

        detected_map = img_raw.copy()
        detected_map[mask_pixel > 0.5] = - 255.0

        control = torch.from_numpy(detected_map.copy()).float().cuda('cuda:0') / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        mask = 1.0 - torch.from_numpy(mask_latent.copy()).float().cuda('cuda:0')
        mask = torch.stack([mask for _ in range(num_samples)], dim=0)
        mask = einops.rearrange(mask, 'b h w -> b 1 h w').clone()

        x0 = torch.from_numpy(img_raw.copy()).float().cuda('cuda:0') / 127.0 - 1.0
        x0 = torch.stack([x0 for _ in range(num_samples)], dim=0)
        x0 = einops.rearrange(x0, 'b h w c -> b c h w').clone()

        mask_pixel_batched = mask_pixel[None, :, :, None]
        img_pixel_batched = img_raw.copy()[None]

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=True)
        x0 = model.get_first_stage_encoding(model.encode_first_stage(x0))

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond, x0=x0, mask=mask)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().astype(np.float32)
        x_samples = x_samples * mask_pixel_batched + img_pixel_batched * (1.0 - mask_pixel_batched)

        results = [x_samples[i].clip(0, 255).astype(np.uint8) for i in range(num_samples)]
    return [detected_map.clip(0, 255).astype(np.uint8)] + results



# |<------------        My Loop       ----------->|

# Variables
maxloops = 5
output_filename = 'output_with_mask.jpg'
input_filename = args.img_path

print("Entering Loop")

prompt = ''
a_prompt = 'absolutely zero creativity, zero imagination'
n_prompt = ''
num_samples = 1

# Create masks directory if not exists
os.makedirs('./masks', exist_ok=True)
os.makedirs('./results', exist_ok=True)

# Loop for processing each mask
for i in range(maxloops):
    print(f"Iteration: {i + 1}")
    
    image = cv2.imread(input_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("Token Cutting")
    mask = tokencutting(img_path=input_filename)
    if mask is None:
        print("No more significant mask found. Finishing Process.")
        break
    print(type(mask))
    print(mask.shape)
    
    
    mask = mask.astype(np.uint8)*255
    
    kernel = np.ones((20,20), np.uint8)  # Adjust the kernel size as needed
    dilated_mask = cv2.dilate(mask, kernel, iterations=5)
    
    cv2.imwrite(f'./masks/justlmi{i+1}.png', dilated_mask)
    
    print(f'lmi {i + 1} saved')

    # Inpaint using the selected mask
    print("Begin Inpainting")
    inpainted_result = process({'image': image, 'mask': dilated_mask}, 
                                prompt, a_prompt, n_prompt, num_samples, 
                                512, 50, False, 1.0, 7.0, 12345, 1.0, 5.0)
    print("Finished inpainting")


    # Save the image
    print("Saving File")
    cv2.imwrite(f'./results/output_iteration_{i + 1}.png', cv2.cvtColor(inpainted_result[1], cv2.COLOR_RGB2BGR))
    input_filename = f'./results/output_iteration_{i + 1}.png'
    input_filenamet = os.path.abspath(input_filename)
    print(f"Process completed for iteration {i + 1}")

