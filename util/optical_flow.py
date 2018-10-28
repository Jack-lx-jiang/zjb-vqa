import os
from os.path import basename, splitext

import cv2
import numpy as np
from skimage import transform
from tqdm import tqdm


def opt_flow(file, output_dir, every=1, limit=0, compressed=False, position=0, tot_bar=None):
    video_name = splitext(basename(file))[0]
    output_name = os.path.join(output_dir, video_name) + '_opt'
    if compressed:
        output_name += '.npz'
        save = np.savez_compressed
    else:
        output_name += '.npy'
        save = np.save

    if not os.path.exists(output_name):
        cap = cv2.VideoCapture(file)
        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = (tot_frames - 1) // every + 1
        if limit:
            total = min(total, limit)

        _, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        flow_array = np.zeros((total, 224, 224, 3), dtype='uint8')
        flow_array[0] = transform.resize(rgb, (224, 224), preserve_range=True)

        with tqdm(total=total, position=position, initial=1) as pbar:
            pbar.set_description(video_name)
            for _ in range(total - 1):
                for _ in range(every):
                    ret, frame2 = cap.read()
                    if not ret:
                        break

                if not ret:
                    break

                next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                flow_array[pbar.n] = transform.resize(rgb, (224, 224), preserve_range=True)
                # comment below code to test performance in slient mode
                # if not pbar.n % 15:
                #     cv2.imshow('frame2', rgb)
                #     k = cv2.waitKey(30) & 0xff
                #     if k == 27:
                #         break
                #     elif k == ord('s'):
                #         cv2.imwrite('opticalfb.png', frame2)
                #         cv2.imwrite('opticalhsv.png', rgb)
                # above

                prvs = next
                pbar.update(1)
        cap.release()
        save(output_name, flow_array)

    if tot_bar:
        tot_bar.update(1)  # this may have critical section problem?


def batch_opt_flow(file_list, video_dir, output_dir, every=1, limit=0, compressed=False, position=0, tot_bar=None):
    for file in file_list:
        filename = os.path.join(video_dir, file)
        opt_flow(filename, output_dir, every, limit, compressed, position, tot_bar)


if __name__ == '__main__':
    opt_flow('D:\\VQADatasetA_20180815\\train\\ZJL3554.mp4',
             output_dir='.', every=8, limit=12)
