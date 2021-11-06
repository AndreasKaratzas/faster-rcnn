

import os
import cv2
import math
import numpy as np

from typing import Dict
from tqdm import tqdm


def get_cv2_stats() -> Dict[str, int]:
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    return dict(Major=major_ver, Minor=minor_ver, Sub=subminor_ver)


def print_cv2_stats(cvs_stats: Dict[str, int]):
    print(
        f"OpenCV version {cvs_stats.get('Major')}.{cvs_stats.get('Minor')}.{cvs_stats.get('Sub')}")


def print_video_stats(video_filepath: str, video_stats: Dict[str, float]):
    print(f"Stats for video {video_filepath}\n"
          f"\tWidth: {video_stats.get('Width')}, Height: {video_stats.get('Height')}\n"
          f"\tFrames per Second: {video_stats.get('FPS')}\n"
          f"\tTotal frame count: {video_stats.get('Number_of_frames')}")


def get_video_stats(video_filepath: str) -> Dict[str, float]:
    video = cv2.VideoCapture(video_filepath)

    if video.isOpened():
        width = math.ceil(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = math.ceil(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = math.ceil(video.get(cv2.CAP_PROP_FPS))
        frame_count = math.ceil(video.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        raise FileNotFoundError("Video requested could not be accessed.")

    return dict(Width=width, Height=height, FPS=fps, Number_of_frames=frame_count)


def frame_iter(capture, description, frames, unit):
    def _iterator():
        while capture.grab():
            yield capture.retrieve()[1]

    return tqdm(
        _iterator(),
        desc=description,
        total=frames,
        unit=unit
    )


def rgb_converter(input_filepath: str, output_dir: str, video_id: str, video_stats: Dict[str, float]):
    video_capture = cv2.VideoCapture(input_filepath)
    main_out_path = os.path.join(output_dir, video_id)

    if not os.path.exists(main_out_path):
        os.makedirs(main_out_path, exist_ok=True)
    
    i = 0
    for frame in frame_iter(video_capture, 'Extracting images from RGB video ', video_stats.get('Number_of_frames'), ' frames extracted'):
        i += 1
        cv2.imwrite(os.path.join(main_out_path, 'frame_' + str(i) + '.png'), frame)


def gray_converter(input_filepath: str, output_dir: str, video_id: str, video_stats: Dict[str, float]):
    video_capture = cv2.VideoCapture(input_filepath)
    main_out_path = os.path.join(output_dir, video_id)

    if not os.path.exists(main_out_path):
        os.makedirs(main_out_path, exist_ok=True)
    
    i = 0
    for frame in frame_iter(video_capture, 'Extracting images in grayscale from RGB video ', video_stats.get('Number_of_frames'), ' frames extracted'):
        i += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(main_out_path, 'frame_' + str(i) + '.png'), gray)


def spatiotemporal_converter(input_filepath: str, output_dir: str, video_id: str, video_stats: Dict[str, float]):
    video_capture = cv2.VideoCapture(input_filepath)
    main_out_path = os.path.join(output_dir, video_id)

    if not os.path.exists(main_out_path):
        os.makedirs(main_out_path, exist_ok=True)

    img = np.zeros((3, video_stats.get('Height'),
                   video_stats.get('Width')), dtype=np.uint8)
    
    i = 0
    for frame in frame_iter(video_capture, 'Extracting images in grayscale from RGB video ', video_stats.get('Number_of_frames'), ' frames extracted'):
        img[i % 3] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (i % 3 == 0) and (i > 0):
            spatiotemporal = np.transpose(spatiotemporal, (1, 2, 0))
            cv2.imwrite(os.path.join(main_out_path, 'frame_' + str(i) + '.png'), spatiotemporal)
        
        spatiotemporal = img.copy()

        i += 1


def main():
    input_video_filepaths = [
        'E:\\videos\\example_1.MP4',
        'E:\\videos\\example_2.avi',
    ]

    output_root_dir = ""

    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)

    cv2_stats = get_cv2_stats()
    print_cv2_stats(cv2_stats)

    for input_video_filepath in input_video_filepaths:
        _, tail = os.path.split(input_video_filepath)
        video_id = os.path.splitext(tail)[0]
        video_stats = get_video_stats(input_video_filepath)
        print_video_stats(input_video_filepath, video_stats)
        
        gray_converter(
            input_filepath=input_video_filepath, 
            output_dir=output_root_dir, 
            video_id=video_id, 
            video_stats=video_stats
        )

        spatiotemporal_converter(
            input_filepath=input_video_filepath,
            output_dir=output_root_dir,
            video_id=video_id,
            video_stats=video_stats
        )


if __name__ == "__main__":
    main()
