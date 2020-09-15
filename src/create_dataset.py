import run_full
import cv2
import os, shutil

def clear_folder():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','pic')
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

vidcap = cv2.VideoCapture('../../../4/vid4.mp4')
success,image = vidcap.read()
count = 0

while success:
    cv2.imwrite("../../pic/frame%s.jpg" % str(count).zfill(5), image)
    if count%40 == 0 and count != 0:
        run_full.demo_images()
        clear_folder()
    success,image = vidcap.read()
    print(' Read a new frame: ', success)
    count += 1