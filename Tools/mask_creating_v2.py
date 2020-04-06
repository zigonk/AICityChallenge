#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:21:20 2019

@author: peterluong
"""

import numpy as np
import os
import cv2
import json
import imageio
import argparse

def apply_morphology(frame):
    """Applies morphological operations to remove noise and to segment vehicles
    """
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # kernel_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel=kernel_open)         # Noise removal
    # frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel=kernel_close)
    frame = cv2.morphologyEx(frame, cv2.MORPH_DILATE, kernel=kernel_dilate)     # Expands detected ROIs
    # frame = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel=kernel_grad)     # Creates ROI outline
    return frame

def save_mask(mask, vid, scene_id, frame):
  if not os.path.isdir(mask_path + '/masks_refine_non_expand/'):
    os.makedirs(mask_path + '/masks_refine_non_expand/')
  np.save(mask_path + '/masks_refine_non_expand/mask_%d_%d.npy' %(vid, scene_id), mask)
  if not os.path.isdir(mask_path + '/masks/'):
    os.makedirs(mask_path + '/masks/')
  imageio.imwrite(mask_path + '/masks/%d_%d.jpg' %(vid, scene_id), mask.reshape(410,800,1).astype(np.uint8) * frame)

def extractMask(video_id):
    for vid in range(video_id, video_id + 1):
      capture = cv2.VideoCapture(video_path + '/%d.mp4' %vid)
      scenes = json.load(open(data_path + '/unchanged_scene_periods.json'))
      bs = cv2.createBackgroundSubtractorMOG2()
      bs.setHistory(120)
      ret, frame = capture.read()
      scene_id = 0
      cur_vid_scenes = scenes['%d' %vid]
      cur_frame = 0
      mask = 0
      print("Start vid {} scene {}".format(vid, scene_id))
      while ret:
        start = cur_vid_scenes[scene_id][0]
        end   = cur_vid_scenes[scene_id][1]
        bs.apply(frame)
        bg_img = bs.getBackgroundImage()

        fg = cv2.subtract(frame,bg_img)
        fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        if (cur_frame in range(start, end + 1)):
          mask = cv2.bitwise_or(mask,fg)
          mask = cv2.medianBlur(mask, 3) # Clear the image, remove small spots
          mask = cv2.GaussianBlur(mask, (3, 3), 0) # Smooth the image
          # remove abnormal trajectory
          _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_TOZERO)
          _, mask = cv2.threshold(mask, 110, 255, cv2.THRESH_BINARY)
        elif cur_frame > end:
          mask = apply_morphology(mask)
          mask = (mask > 0).astype(np.uint8)
          save_mask(mask, vid, scene_id, frame)
          scene_id += 1
          if (scene_id == len(cur_vid_scenes)):
            break
          print("Start vid {} scene {}".format(vid, scene_id))
          mask = 0
        cur_frame += 1
        ret, frame = capture.read()

if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Preprocess cut files.')
    parser.add_argument('--video',
                        help='Directory containing video.',
                        type=str)
    parser.add_argument('--save',
                        help='Directory containing mask.',
                        type=str)
    parser.add_argument('--data',
                        help='Directory containing preprocessing data.',
                        type=str)
    parser.add_argument('--start_id',
                        help='Process start at <video_id>',
                        type=str)
    parser.add_argument('--stop_id',
                        help='Process stop at <video_id>',
                        type=str)
    args = parser.parse_args()
    mask_path = args.save
    video_path = args.video
    data_path = args.data
    start_id = int(args.start_id)
    stop_id  = int(args.stop_id)
    # extract mask
    # videos = [45, 61, 84, 89]
    # videos = [61, 45, 84]
    # videos = [51]
    for c in range(start_id, stop_id):
        extractMask(c)

    # expandMask(video_id = 46, scene_id = 1)

    #visualize extracted masks
    # verifyMask(video_id = 51, scene_id = 1, expand = True)