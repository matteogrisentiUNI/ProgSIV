import os
from LOBES import LOBES


if __name__ == "__main__":

    video_path = 'test/Video/Car2.mp4'
    output_folder = os.path.join('test/Global/Car2')
    object_detected = 'car'

    LOBES(video_path, object_detected, vertical=False, output_folder=output_folder,  saveVideo=True, debugPrint=False)

