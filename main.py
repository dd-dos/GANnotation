import logging
import GANnotation
from PIL import Image
from pathlib import Path
import glob
import numpy as np
import tqdm

# img_list = ['test_images/300VW_Dataset_2015_12_14/001/image/000001.jpg']
# pts_list = ['test_images/300VW_Dataset_2015_12_14/001/annot/000001.pts']

logging.info('=> Read data path...')
pts_list = np.array(list(Path('../300VW-3D_cropped').glob('**/*.mat'))).astype(str)
img_list = [pts_list[i].replace('mat', 'jpg') for i in range(len(pts_list))]
logging.info('=> Done reading.')

sub_pts_list = [pts_list[i] for i in range(len(pts_list)) if i%5==0][:3]
sub_img_list = [img_list[i] for i in range(len(img_list)) if i%5==0][:3]

assert len(sub_pts_list) == len(sub_img_list)

model = GANnotation.GANnotation(path_to_model='models/myGEN.pth', enable_cuda=True)

split_size=1000
div_batch = len(sub_pts_list) // split_size
mod_batch = len(sub_pts_list) % split_size

import shutil
shutil.rmtree('300VW-3D_cropped_non_closed_eyes_foo', ignore_errors=True)
for idx in tqdm.tqdm(range(div_batch)):
    model.gen_close_eyes_batch(
        sub_img_list[idx*split_size:(idx+1)*split_size], 
        sub_pts_list[idx*split_size:(idx+1)*split_size], 
        batch_size=64, 
        closed_eyes=True,
        out_dir='300VW-3D_cropped_non_closed_eyes_foo'
    )
if mod_batch > 0:
    model.gen_close_eyes_batch(
        sub_img_list[-mod_batch:], 
        sub_pts_list[-mod_batch:], 
        batch_size=64, 
        closed_eyes=True,
        out_dir='300VW-3D_cropped_non_closed_eyes_foo'
    )
