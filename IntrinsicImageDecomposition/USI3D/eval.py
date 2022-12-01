__author__ = "Prateek Gulati"
__date__ = "10/10/22"

'''
Similar to test.py 
Modularized the code to return the evaluation results
'''

from __future__ import print_function

from tqdm import tqdm

from utils import get_config
from trainer import UnsupIntrinsicTrainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
from argparse import Namespace

opts_dict = {
    'config':'USI3D/configs/intrinsic_MIX_IIW.yaml', 
    'input_dir':'./USI3D/test', 
    'output_folder':'./USI3D/results/', 
    'checkpoint':'./USI3D/pretrained_model/gen-MIX.pt', 
    'output_only':False,
    'seed':10
}
opts = Namespace(**opts_dict)



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


wo_fea = 'wo_fea' in opts.checkpoint



torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)

# Setup model and data loader
trainer = UnsupIntrinsicTrainer(config)

state_dict = torch.load(opts.checkpoint, map_location='cuda:0')
trainer.gen_i.load_state_dict(state_dict['i'])
trainer.gen_r.load_state_dict(state_dict['r'])
trainer.gen_s.load_state_dict(state_dict['s'])
trainer.fea_s.load_state_dict(state_dict['fs'])
trainer.fea_m.load_state_dict(state_dict['fm'])

trainer.cuda()
trainer.eval()

if 'new_size' in config:
    new_size = config['new_size']
else:
    new_size = config['new_size_i']

intrinsic_image_decompose = trainer.inference

def main(config = opts.config, input_dir=opts.input_dir, output_folder = opts.output_folder, checkpoint = opts.checkpoint):
    with torch.no_grad():
        transform = transforms.Compose([transforms.Resize(new_size),
                                        transforms.ToTensor(),
    #                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    # Make sure the vaule range of input tensor be consistent to the training time
                                       ])

        image_paths = os.listdir(opts.input_dir)
        image_paths = [x for x in image_paths if is_image_file(x)]
        t_bar = tqdm(image_paths)
        t_bar.set_description('Processing')
        pred_list=[]
        for image_name in t_bar:
            image_pwd = os.path.join(opts.input_dir, image_name)

            out_root = os.path.join(opts.output_folder, image_name.split('.')[0])

            if not os.path.exists(out_root):
                os.makedirs(out_root)

            image = Variable(transform(Image.open(image_pwd).convert('RGB')).unsqueeze(0).cuda())

            # Start testing
            im_reflect, im_shading = intrinsic_image_decompose(image, wo_fea)
            # im_reflect = (im_reflect + 1) / 2.
            im_shading = (im_shading + 1) / 2.

            path_reflect = os.path.join(out_root, 'output_r.jpg')
            path_shading = os.path.join(out_root, 'output_s.jpg')

            vutils.save_image(im_reflect.data, path_reflect, padding=0, normalize=True)
            vutils.save_image(im_shading.data, path_shading, padding=0, normalize=True)

            pred_dict = {'pred_alb': im_reflect.data.squeeze(),
                             'img': image.squeeze(),
                             'pred_shd': im_shading.data.squeeze(),
                            }
            pred_list.append(pred_dict)

            if not opts.output_only:
                # also save input images
                vutils.save_image(image.data, os.path.join(out_root, 'input.jpg'), padding=0, normalize=True)
    return pred_list

