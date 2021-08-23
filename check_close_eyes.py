import os
import glob
from pathlib import Path

import utils
import shutil
import tqdm
import scipy.io as sio
import multiprocessing as mp

def task(item):
    item = str(item)
    pts_2d = sio.loadmat(item)['pt2D']
    eyes_2d = utils.get_eyes(pts_2d)

    pts_3d = sio.loadmat(item)['pt3d']
    eyes_3d = utils.get_eyes(pts_3d)

    folder_id = item.split('/')[-2]
    file_id = item.split('/')[-1]

    if utils.check_close_eye(eyes_2d['left']) and utils.check_close_eye(eyes_2d['right']):
        shutil.copyfile(item.replace('mat', 'jpg'), f'test_images/{folder_id}_{file_id}.jpg')
    elif utils.check_close_eye(eyes_3d['left']) and utils.check_close_eye(eyes_3d['right']):
        shutil.copyfile(item.replace('mat', 'jpg'), f'test_images/{folder_id}_{file_id}_3D.jpg')
    else:
        shutil.copyfile(item.replace('mat', 'jpg'), f'300VW-3D_cropped_non_closed_eyes/{folder_id}/{file_id}.jpg')
        shutil.copyfile(item, f'300VW-3D_cropped_non_closed_eyes/{folder_id}/{file_id}.mat')

if __name__=='__main__':
    shutil.rmtree('300VW-3D_cropped_non_closed_eyes', ignore_errors=True)
    shutil.rmtree('test_images', ignore_errors=True)

    os.makedirs('300VW-3D_cropped_non_closed_eyes', exist_ok=True)
    os.makedirs('test_images', exist_ok=True)

    bag = list(Path('../300VW-3D_cropped').glob('**/*.mat'))

    for folder_path in glob.glob('../300VW-3D_cropped/*'):
        folder_name = folder_path.split('/')[-1]
        os.makedirs(
            os.path.join('300VW-3D_cropped_non_closed_eyes', folder_name),
            exist_ok=True
        )

    with mp.Pool(mp.cpu_count()) as p:
        r = list(tqdm.tqdm(p.imap(task, bag), total=len(bag)))

