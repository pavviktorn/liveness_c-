import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


def fit_img(img, out_w=256, out_h=256):
    src_h = img.shape[0]
    src_w = img.shape[1]
    exp_img = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255
    if src_h > src_w:
        new_h = out_h
        new_w = int(src_w * out_h / src_h)
        exp_scale = new_h / src_h
        cropped = cv2.resize(img, (new_w, new_h))
        x_pos = int(out_h / 2 - cropped.shape[0] / 2)
        y_pos = int(out_w / 2 - cropped.shape[1] / 2)
        exp_img[x_pos: x_pos + cropped.shape[0], y_pos: y_pos + cropped.shape[1], :] = cropped
    else:
        new_w = out_w
        new_h = int(src_h * new_w / src_w)
        exp_scale = new_w / src_w
        cropped = cv2.resize(img, (new_w, new_h))
        x_pos = int(out_h / 2 - cropped.shape[0] / 2)
        y_pos = int(out_w / 2 - cropped.shape[1] / 2)
        exp_img[x_pos: x_pos + cropped.shape[0], y_pos: y_pos + cropped.shape[1], :] = cropped

    return exp_img


def crop(img, bbox, top, bottom, left, right):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if abs(bottom - top) < 0.2 * h:
        bottom = int(bottom + 0.2 * h)
        top = int(top - 0.2 * h)

    if abs(right - left) < 0.2 * w:
        right = int(right + 0.2 * w)
        left = int(left - 0.2 * w)

    if left < 0: left = 0
    if right >= img.shape[1]: right = img.shape[1] - 1
    if bottom >= img.shape[0]: bottom = img.shape[0] - 1
    if top < 0: top = 0

    return img[top:bottom, left:right]


def get_parts(org_img, bbox, lmk, out_w=256, out_h=256):

    result = []

    kps = np.zeros((5, 2)).astype(int)
    kps[0] = lmk[38]
    kps[1] = lmk[88]
    kps[2] = lmk[86]
    kps[3] = lmk[52]
    kps[4] = lmk[61]

    dist_eyes = abs(kps[1][0] - kps[0][0])

    # left eye
    left = kps[0][0] - dist_eyes//2
    if left < 0: left = 0
    right = kps[0][0] + dist_eyes//2
    if right >= org_img.shape[1]: right = org_img.shape[1] - 1
    bottom = kps[0][1] + dist_eyes//3
    if bottom >= org_img.shape[0]: bottom = org_img.shape[0] - 1
    top = bottom - (right - left)
    if top < 0: top = 0
    cropped = crop(org_img, bbox, top, bottom, left, right)
    cropped = fit_img(cropped)
    result.append(cropped)
    # cv2.imwrite('0_left_eye.jpg', cropped)

    # right eye
    left = kps[1][0] - dist_eyes//2
    if left < 0: left = 0
    right = kps[1][0] + dist_eyes//2
    if right >= org_img.shape[1]: right = org_img.shape[1] - 1
    bottom = kps[1][1] + dist_eyes//3
    if bottom >= org_img.shape[0]: bottom = org_img.shape[0] - 1
    top = bottom - (right - left)
    if top < 0: top = 0
    cropped = crop(org_img, bbox, top, bottom, left, right)
    cropped = fit_img(cropped)
    cropped = cv2.flip(cropped, 1)
    result.append(cropped)
    # cv2.imwrite('1_right_eye.jpg', cropped)

    # forehead
    left = bbox[0]
    if left < 0: left = 0
    right = bbox[2]
    if right >= org_img.shape[1]: right = org_img.shape[1] - 1
    top = bbox[1] - (bbox[3] - bbox[1]) // 8
    if top < 0: top = 0
    bottom = bbox[1] + (bbox[3] - bbox[1]) // 4
    if bottom >= org_img.shape[0]: bottom = org_img.shape[0] - 1
    cropped = crop(org_img, bbox, top, bottom, left, right)
    cropped = fit_img(cropped)
    result.append(cropped)
    # cv2.imwrite('2_forehead.jpg', cropped)

    # left ear
    left = bbox[0] - (bbox[2] - bbox[0]) // 6
    if left < 0: left = 0
    right = bbox[0] + (bbox[2] - bbox[0]) // 6
    if right >= org_img.shape[1]: right = org_img.shape[1] - 1
    top = bbox[1]
    if top < 0: top = 0
    bottom = bbox[3]
    if bottom >= org_img.shape[0]: bottom = org_img.shape[0] - 1
    cropped = crop(org_img, bbox, top, bottom, left, right)
    cropped = fit_img(cropped)
    result.append(cropped)
    # cv2.imwrite('3_left_ear.jpg', cropped)

    # right ear
    left = bbox[2] - (bbox[2] - bbox[0]) // 6
    if left < 0: left = 0
    right = bbox[2] + (bbox[2] - bbox[0]) // 6
    if right >= org_img.shape[1]: right = org_img.shape[1] - 1
    top = bbox[1]
    if top < 0: top = 0
    bottom = bbox[3]
    if bottom >= org_img.shape[0]: bottom = org_img.shape[0] - 1
    cropped = crop(org_img, bbox, top, bottom, left, right)
    cropped = fit_img(cropped)
    cropped = cv2.flip(cropped, 1)
    result.append(cropped)
    # cv2.imwrite('4_right_ear.jpg', cropped)

    # chin
    left = bbox[0]
    if left < 0: left = 0
    right = bbox[2]
    if right >= org_img.shape[1]: right = org_img.shape[1] - 1
    top = int(bbox[3] - (bbox[3] - bbox[1]) / 2.5)
    if top < 0: top = 0
    bottom = bbox[3] + (bbox[3] - bbox[1]) // 10
    if bottom >= org_img.shape[0]: bottom = org_img.shape[0] - 1
    cropped = crop(org_img, bbox, top, bottom, left, right)
    cropped = fit_img(cropped)
    result.append(cropped)
    # cv2.imwrite('5_chin.jpg', cropped)

    return result



if __name__ == '__main__':
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    # img = ins_get_image('t1')
    img = cv2.imread('ffe4de56-c40f-4366-8aa0-7acace050526.jpg')
    faces = app.get(img)
    #assert len(faces)==6
    tim = img.copy()
    color = (200, 160, 75)
    for face in faces:
        bbox = np.round(face.bbox).astype(int)
        tim = cv2.rectangle(
            tim, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
            (255, 255, 255), 2
        )

        kps1 = np.round(face.kps).astype(int)
        for i in range(5):
            p = tuple(kps1[i])
            cv2.circle(tim, p, 2, (0,0,255), 2, cv2.LINE_AA)

        lmk = face.landmark_2d_106
        lmk = np.round(lmk).astype(int)
        for i in range(lmk.shape[0]):
            p = tuple(lmk[i])
            cv2.circle(tim, p, 1, color, 1, cv2.LINE_AA)

        kps = np.zeros((5,2)).astype(int)
        kps[0] = lmk[38]
        kps[1] = lmk[88]
        kps[2] = lmk[86]
        kps[3] = lmk[52]
        kps[4] = lmk[61]
        for i in range(5):
            p = tuple(kps[i])
            cv2.circle(tim, p, 2, (255,0,0), 2, cv2.LINE_AA)

        # checking bbox & lmk
        if bbox[0] < 0: bbox[0] = 0
        if bbox[0] >= img.shape[1]: bbox[0] = img.shape[1] - 1
        if bbox[2] < 0: bbox[2] = 0
        if bbox[2] >= img.shape[1]: bbox[2] = img.shape[1] - 1
        if bbox[1] < 0: bbox[1] = 0
        if bbox[1] >= img.shape[0]: bbox[1] = img.shape[0] - 1
        if bbox[3] < 0: bbox[3] = 0
        if bbox[3] >= img.shape[0]: bbox[3] = img.shape[0] - 1

        for iii in range(lmk.shape[0]):
            if lmk[iii][0] < 0: lmk[iii][0] = 0
            if lmk[iii][0] >= img.shape[1]: lmk[iii][0] = img.shape[1] - 1
            if lmk[iii][1] < 0: lmk[iii][1] = 0
            if lmk[iii][1] >= img.shape[0]: lmk[iii][1] = img.shape[0] - 1

        res = get_parts(img, bbox, lmk)
        merge = cv2.hconcat(res)
        cv2.imwrite(f'merge.jpg', merge)

    cv2.imwrite('./test_out.jpg', tim)

    get_parts(img, bbox, lmk)
