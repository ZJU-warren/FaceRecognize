import cv2
import numpy as np
from B_Detect_Align.MTCNN_model import MtCNN
import DataSetLink as DLSet
from Tools import show_img, bgr2rgb


class DetectProxy:
    def __init__(self):
        self.model = MtCNN()
        self.threshold = [0.5, 0.6, 0.7]

    def detect(self, img):
        rectangles = self.model.detect_face(img, self.threshold)
        draw = img.copy()

        best_crop_img = img
        flag = False
        best_distance = 0

        # find the best face in img
        for rectangle in rectangles:
            if rectangle is not None:
                W = -int(rectangle[0]) + int(rectangle[2])
                H = -int(rectangle[1]) + int(rectangle[3])
                paddingH = 0.01 * W
                paddingW = 0.02 * H
                crop_img = img[int(rectangle[1] + paddingH):int(rectangle[3] - paddingH),
                           int(rectangle[0] - paddingW):int(rectangle[2] + paddingW)]
                if crop_img is None:
                    continue
                if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                    continue

                if crop_img.shape[0] < 1 or crop_img.shape[1] < 1:
                    continue

                eye_center = ((int(rectangle[5]) + int(rectangle[7])) / 2,
                              (int(rectangle[6]) + int(rectangle[8])) / 2)
                dy = int(rectangle[8]) - int(rectangle[6])
                dx = int(rectangle[7]) - int(rectangle[5])

                angle = cv2.fastAtan2(dy, dx)
                rot = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
                rot_img = cv2.warpAffine(crop_img, rot, dsize=(
                    crop_img.shape[1], crop_img.shape[0]))

                distance = np.sqrt((draw.shape[0] - (int(rectangle[0]) + int(rectangle[2])) / 2) ** 2
                                   + (draw.shape[1] - (int(rectangle[1]) + int(rectangle[3])) / 2) ** 2)
                if flag is False:
                    flag = True
                    best_crop_img = rot_img
                    best_distance = distance

                elif distance < best_distance:
                    best_crop_img = rot_img
                    best_distance = distance

                # cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                #               (0, 0, 255), 1)

                # for i in range(5, 15, 2):
                #     cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))

        return best_crop_img

    def mark(self, img):
        rectangles = self.model.detect_face(img, self.threshold)
        draw = img.copy()

        new_img = img
        flag = False
        best_distance = 0
        # find the best face in img
        for rectangle in rectangles:
            if rectangle is not None:
                W = -int(rectangle[0]) + int(rectangle[2])
                H = -int(rectangle[1]) + int(rectangle[3])
                paddingH = 0.01 * W
                paddingW = 0.02 * H
                crop_img = img[int(rectangle[1] + paddingH):int(rectangle[3] - paddingH),
                           int(rectangle[0] - paddingW):int(rectangle[2] + paddingW)]

                if crop_img is None:
                    continue

                if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                    continue

                if crop_img.shape[0] < 1 or crop_img.shape[1] < 1:
                    continue

                distance = np.sqrt((draw.shape[0] - (int(rectangle[0]) + int(rectangle[2])) / 2) ** 2
                                   + (draw.shape[1] - (int(rectangle[1]) + int(rectangle[3])) / 2) ** 2)
                
                need_flag = False
                if flag is False:
                    flag=True
                    best_distance=distance
                    need_flag = True

                elif distance < best_distance:
                    best_distance=distance
                    need_flag = True

                if need_flag:
                    best_img = img.copy()
                    cv2.rectangle(best_img , (int(rectangle[0]), int(rectangle[1])),
                              (int(rectangle[2]), int(rectangle[3])), (0, 0, 255), 1)
                    new_img = best_img
        return flag, new_img

    def app_detect(self, img):
        rectangles = self.model.detect_face(img, self.threshold)
        draw = img.copy()

        best_crop_img = img
        flag = False
        best_distance = 0
        best_rectangle = None
        # find the best face in img
        for rectangle in rectangles:
            if rectangle is not None:
                W = -int(rectangle[0]) + int(rectangle[2])
                H = -int(rectangle[1]) + int(rectangle[3])
                paddingH = 0.01 * W
                paddingW = 0.02 * H
                crop_img = img[int(rectangle[1] + paddingH):int(rectangle[3] - paddingH),
                           int(rectangle[0] - paddingW):int(rectangle[2] + paddingW)]
                if crop_img is None:
                    continue
                if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                    continue

                if crop_img.shape[0] < 1 or crop_img.shape[1] < 1:
                    continue

                eye_center = ((int(rectangle[5]) + int(rectangle[7])) / 2,
                              (int(rectangle[6]) + int(rectangle[8])) / 2)
                dy = int(rectangle[8]) - int(rectangle[6])
                dx = int(rectangle[7]) - int(rectangle[5])

                angle = cv2.fastAtan2(dy, dx)
                rot = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
                rot_img = cv2.warpAffine(crop_img, rot, dsize=(
                    crop_img.shape[1], crop_img.shape[0]))

                distance = np.sqrt((draw.shape[0] - (int(rectangle[0]) + int(rectangle[2])) / 2) ** 2
                                   + (draw.shape[1] - (int(rectangle[1]) + int(rectangle[3])) / 2) ** 2)
                if flag is False:
                    flag = True
                    best_crop_img = rot_img
                    best_distance = distance
                    best_rectangle = rectangle
                    
                elif distance < best_distance:
                    best_crop_img = rot_img
                    best_distance = distance
                    best_rectangle = rectangle
                    
                # cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                #               (0, 0, 255), 1)

                # for i in range(5, 15, 2):
                #     cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))

        return best_crop_img, best_rectangle


if __name__ == '__main__':
    img=cv2.imread(DLSet.dataset_link + '/timg.jpg')
    # cv2.imwrite(DLSet.dataset_link + "/out.jpg", draw)
    dp=DetectProxy()
    draw=dp.detect(img)
    b, g, r=cv2.split(draw)
    draw_new=cv2.merge([r, g, b])
    show_img(draw_new)
