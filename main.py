import GANnotation
from PIL import Image

img_list = ['test_images/300VW_Dataset_2015_12_14/001/image/000001.jpg']
pts_list = ['test_images/300VW_Dataset_2015_12_14/001/annot/000001.pts']

model = GANnotation.GANnotation(path_to_model='models/myGEN.pth', enable_cuda=False)
import time
t0 = time.time()
res = model.gen_close_eyes_batch(img_list, pts_list, batch_size=16)
print(time.time()-t0)
import cv2
img = cv2.resize(res[0], (256,256))
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow('',img)
cv2.waitKey(0)
cv2.destroyAllWindows()