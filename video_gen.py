import numpy as np
from image_gen import process_image
from tracker import LineTracker
import sys
import yaml
import pickle

from moviepy.editor import VideoFileClip

def process_video_clip(clip, dist_pickle,src, dst, thresholds, tracker):
    def process_frame(image):
        return process_image(image, dist_pickle,src, dst, thresholds, tracker)
    return clip.fl_image(process_frame) #NOTE: this function expects color images!!


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("usage: video_gen.py video_path cal_image_folder_path parameter_file_path\n  note: remember to use trailing '/' in folder paths. e.g. camera_cal/")
        sys.exit(1)

    cal_image_path = sys.argv[2]

    # Read in the saved objpoints and imgpoints
    dist_pickle = pickle.load(open(cal_image_path+"dist_pickle.p", "rb" ))

    Input_video = sys.argv[1]
    video_file = Input_video.split('/')[-1]
    Output_video = video_file.split('.')[0]+'_output.mp4'

    with open(sys.argv[3]) as f:
        config = yaml.load(f)

    thresholds = config['thresholds']

    src = np.array(config['src']).astype(np.float32)
    dst = np.array(config['dst']).astype(np.float32)

    tracker_params = config['tracker_params']

    # Set up the overall class to do all the tracking
    curve_centers = LineTracker(window_width = tracker_params['window_width'], window_height = tracker_params['window_height'], margin = tracker_params['margin'], ym = tracker_params['ym_per_pix'], xm = tracker_params['xm_per_pix'], smooth_factor=tracker_params['smooth_factor'])

    clip1 = VideoFileClip(Input_video)
    video_clip = clip1.fx(process_video_clip,dist_pickle, src, dst,thresholds,curve_centers)
    video_clip.write_videofile(Output_video, audio=False)
