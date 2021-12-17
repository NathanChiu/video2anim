
import cv2
import time
import os

timestr = time.strftime("%Y%m%d_%H%M%S")
vidcap = cv2.VideoCapture('videos/test.mp4')
print(f"FPS: {vidcap.get(cv2.CAP_PROP_FPS)}")
success, image = vidcap.read()
count = 0

output_file_dir = os.path.join("videos", "output_frames", timestr)
os.makedirs(output_file_dir, exist_ok=True)
while success:
    # Phone records at 29.957979731399853 fps
    small = cv2.resize(image, (0, 0), fx=1/3.75, fy=1/3.75)
    cv2.imwrite(os.path.join(output_file_dir, f"{count}.jpg"), small)     # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1