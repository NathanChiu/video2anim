import cv2
import numpy as np
import os
img=[]
for i in range(0,99+1):
    img.append(cv2.imread(os.path.join('ebsynth','nathan_test','out', f"{i}.png")))
print(len(img))
print(img[0])
height,width,layers=img[1].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(os.path.join('ebsynth/nathan_test/out', 'video.mp4'),fourcc,30,(width,height))

for j in range(0,99+1):
    # video.write(img)
    video.write(img[j])

cv2.destroyAllWindows()
video.release()