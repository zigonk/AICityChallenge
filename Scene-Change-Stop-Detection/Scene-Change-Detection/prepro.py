import json
import glob
import os
import cv2
import argparse

# cuts_dir = '~/PreProcessed/test/cuts'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess cut files.')
    parser.add_argument('--cuts_dir',
                        help='Directory containing cut video information.',
                        type=str)
    args = parser.parse_args()
    cuts_dir = args.cuts_dir
    dataset_cuts = {}
    cuts_files = sorted(list(glob.glob(os.path.join(cuts_dir, '*.mp4.json'))), key=lambda x: int(os.path.basename(x).split('.')[0]))
    for cuts_file in cuts_files:
        with open(cuts_file, 'r') as f:
            cuts = json.load(f)
        dirname = os.path.dirname(cuts_file)
        basename = os.path.basename(cuts_file)

        vid = cv2.VideoCapture(os.path.join(dirname[:-4], basename[:-5]))
        num_frms = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        vid.release()

        cur_frm = 0

        dataset_cuts[basename.split('.')[0]] = []
        for cut in cuts:
            #print cur_frm, cut
            if cur_frm < cut - 5*30:
                dataset_cuts[basename.split('.')[0]].append((cur_frm, cut - 30))
            cur_frm = cut + 30
        if cur_frm < num_frms - 5*30:
            dataset_cuts[basename.split('.')[0]].append((cur_frm, num_frms))
        print(basename.split('.')[0], len(cuts), dataset_cuts[basename.split('.')[0]])

    with open(os.path.join(cuts_dir, 'unchanged_scene_periods.json'), 'w') as f:
        json.dump(dataset_cuts, f)
