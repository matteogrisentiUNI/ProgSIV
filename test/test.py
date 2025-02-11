import os
from LOB_S import LOB_S


if __name__ == "__main__":

    video_path = 'test/Video/Boat.mp4'
    output_folder = os.path.join('test/Global/Boat')
    object_detected = 'boat'

    LOB_S(video_path, object_detected, vertical=False, output_folder=output_folder,  saveVideo=True, debugPrint=False)

