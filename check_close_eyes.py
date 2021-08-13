import os
import glob
from pathlib import Path

import utils
import shutil
import tqdm

if __name__=='__main__':
    annot_list = Path('test_images/300VW_Dataset_2015_12_14').glob('**/*.pts')
    for item in tqdm.tqdm(annot_list):
        item = str(item)
        pts = utils.read_pts(item)
        eyes = utils.get_eyes(pts)
        if utils.check_close_eye(eyes['left']) and utils.check_close_eye(eyes['right']):
            folder_id = item.split('/')[-3]
            file_id = item.split('/')[-1].split('.')[0]

            img_path = item.replace('annot', 'image').replace('pts', 'jpg')

            shutil.copyfile(img_path, f'close_eyes/{folder_id}_{file_id}.jpg')

