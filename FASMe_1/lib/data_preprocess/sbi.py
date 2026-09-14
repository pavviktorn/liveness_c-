
import numpy as np
import random
import cv2
import albumentations as alb

import warnings

warnings.filterwarnings('ignore')


from insightface.app import FaceAnalysis


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def apply(self, img, **params):
        return self.randomdownscale(img)

    def randomdownscale(self, img):
        keep_ratio = True
        keep_input_shape = True
        H, W, C = img.shape
        ratio_list = [2, 4]
        r = ratio_list[np.random.randint(len(ratio_list))]
        img_ds = cv2.resize(img, (int(W / r), int(H / r)), interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds = cv2.resize(img_ds, (W, H), interpolation=cv2.INTER_LINEAR)

        return img_ds


class Mask():
    """ Parent class for masks
        the output mask will be <mask_type>.mask
        channels: 1, 3 or 4:
                    1 - Returns a single channel mask
                    3 - Returns a 3 channel mask
                    4 - Returns the original image with the mask in the alpha channel """

    def __init__(self, landmarks, face, channels=4):
        # logger.info("Initializing %s: (face_shape: %s, channels: %s, landmarks: %s)",
        #              self.__class__.__name__, face.shape, channels, landmarks)
        self.landmarks = landmarks
        self.face = face
        self.channels = channels

        mask = self.build_mask()
        self.mask = self.merge_mask(mask)
        # logger.info("Initialized %s", self.__class__.__name__)

    def build_mask(self):
        """ Override to build the mask """
        raise NotImplementedError

    def merge_mask(self, mask):
        """ Return the mask in requested shape """
        # logger.info("mask_shape: %s", mask.shape)
        assert self.channels in (1, 3, 4), "Channels should be 1, 3 or 4"
        assert mask.shape[2] == 1 and mask.ndim == 3, "Input mask be 3 dimensions with 1 channel"

        if self.channels == 3:
            retval = np.tile(mask, 3)
        elif self.channels == 4:
            retval = np.concatenate((self.face, mask), -1)
        else:
            retval = mask

        # logger.info("Final mask shape: %s", retval.shape)
        return retval


class dfl_full(Mask):  # pylint: disable=invalid-name
    """ DFL facial mask """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)

        # nose_ridge = (self.landmarks[27:31], self.landmarks[33:34])
        nose_ridge = (self.landmarks[72:75], self.landmarks[86:87], self.landmarks[80:81])
        # jaw = (self.landmarks[0:17],
        #        self.landmarks[48:68],
        #        self.landmarks[0:1],
        #        self.landmarks[8:9],
        #        self.landmarks[16:17])
        jaw = (self.landmarks[0:33],
               self.landmarks[59:72],
               self.landmarks[9:10],
               self.landmarks[0:1],
               self.landmarks[25:26])
        # eyes = (self.landmarks[17:27],
        #         self.landmarks[0:1],
        #         self.landmarks[27:28],
        #         self.landmarks[16:17],
        #         self.landmarks[33:34])
        eyes = (self.landmarks[43:52],
                self.landmarks[97:106],
                self.landmarks[9:10],
                self.landmarks[72:73],
                self.landmarks[25:26],
                self.landmarks[80:81])
        parts = [jaw, nose_ridge, eyes]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
        return mask


class components(Mask):  # pylint: disable=invalid-name
    """ Component model mask """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)

        # r_jaw = (self.landmarks[0:9], self.landmarks[17:18])
        r_jaw = (self.landmarks[0:17], self.landmarks[43:44])
        # l_jaw = (self.landmarks[8:17], self.landmarks[26:27])
        l_jaw = (self.landmarks[17:33], self.landmarks[101:102])
        # r_cheek = (self.landmarks[17:20], self.landmarks[8:9])
        r_cheek = (self.landmarks[48:50], self.landmarks[0:1])
        # l_cheek = (self.landmarks[24:27], self.landmarks[8:9])
        l_cheek = (self.landmarks[104:106], self.landmarks[0:1])
        # nose_ridge = (self.landmarks[19:25], self.landmarks[8:9],)
        nose_ridge = (self.landmarks[49:51],
                      self.landmarks[102:105],
                      self.landmarks[0:1],)
        # r_eye = (self.landmarks[17:22],
        #          self.landmarks[27:28],
        #          self.landmarks[31:36],
        #          self.landmarks[8:9])
        r_eye = (self.landmarks[43:52],
                 self.landmarks[72:73],
                 self.landmarks[77:81],
                 self.landmarks[83:86],
                 self.landmarks[0:1])
        # l_eye = (self.landmarks[22:27],
        #          self.landmarks[27:28],
        #          self.landmarks[31:36],
        #          self.landmarks[8:9])
        l_eye = (self.landmarks[101:106],
                 self.landmarks[72:73],
                 self.landmarks[77:81],
                 self.landmarks[83:86],
                 self.landmarks[0:1])
        # nose = (self.landmarks[27:31], self.landmarks[31:36])
        nose = (self.landmarks[72:75],
                self.landmarks[86:87],
                self.landmarks[77:81],
                self.landmarks[83:86])
        parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
        return mask


class extended(Mask):  # pylint: disable=invalid-name
    """ Extended mask
        Based on components mask. Attempts to extend the eyebrow points up the forehead
    """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)

        landmarks = self.landmarks.copy()
        # mid points between the side of face and eye point
        # ml_pnt = (landmarks[36] + landmarks[0]) // 2
        ml_pnt = (landmarks[35] + landmarks[9]) // 2
        # mr_pnt = (landmarks[16] + landmarks[45]) // 2
        mr_pnt = (landmarks[93] + landmarks[25]) // 2

        # mid points between the mid points and eye
        # ql_pnt = (landmarks[36] + ml_pnt) // 2
        ql_pnt = (landmarks[35] + ml_pnt) // 2
        # qr_pnt = (landmarks[45] + mr_pnt) // 2
        qr_pnt = (landmarks[93] + mr_pnt) // 2

        # Top of the eye arrays
        # bot_l = np.array((ql_pnt, landmarks[36], landmarks[37], landmarks[38], landmarks[39]))
        bot_l = np.array((ql_pnt, landmarks[35], landmarks[40], landmarks[42], landmarks[39]))
        # bot_r = np.array((landmarks[42], landmarks[43], landmarks[44], landmarks[45], qr_pnt))
        bot_r = np.array((landmarks[89], landmarks[95], landmarks[94], landmarks[93], qr_pnt))

        # Eyebrow arrays
        # top_l = landmarks[17:22]
        top_l = np.array((landmarks[43], landmarks[48], landmarks[49], landmarks[51], landmarks[50]))
        # top_r = landmarks[22:27]
        top_r = np.array((landmarks[102], landmarks[103], landmarks[104], landmarks[105], landmarks[101]))

        # Adjust eyebrow arrays
        # landmarks[17:22] = top_l + ((top_l - bot_l) // 2)
        # landmarks[22:27] = top_r + ((top_r - bot_r) // 2)

        # r_jaw = (landmarks[0:9], landmarks[17:18])
        r_jaw = (landmarks[0:17], landmarks[43:44])
        # l_jaw = (landmarks[8:17], landmarks[26:27])
        l_jaw = (landmarks[17:33], landmarks[101:102])
        # r_cheek = (landmarks[17:20], landmarks[8:9])
        r_cheek = (landmarks[48:50], landmarks[0:1])
        # l_cheek = (landmarks[24:27], landmarks[8:9])
        l_cheek = (landmarks[104:106], landmarks[0:1])
        # nose_ridge = (landmarks[19:25], landmarks[8:9],)
        nose_ridge = (landmarks[49:51],
                      landmarks[102:105],
                      landmarks[0:1],)
        # r_eye = (landmarks[17:22], landmarks[27:28], landmarks[31:36], landmarks[8:9])
        r_eye = (landmarks[43:52],
                 landmarks[72:73],
                 landmarks[77:81],
                 landmarks[83:86],
                 landmarks[0:1])
        # l_eye = (landmarks[22:27], landmarks[27:28], landmarks[31:36], landmarks[8:9])
        l_eye = (landmarks[101:106],
                 landmarks[72:73],
                 landmarks[77:81],
                 landmarks[83:86],
                 landmarks[0:1])
        # nose = (landmarks[27:31], landmarks[31:36])
        nose = (landmarks[72:75],
                landmarks[86:87],
                landmarks[77:81],
                landmarks[83:86])
        parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
        return mask


class facehull(Mask):  # pylint: disable=invalid-name
    """ Basic face hull mask """
    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1, ), dtype=np.float32)
        hull = cv2.convexHull(  # pylint: disable=no-member
            np.array(self.landmarks).reshape((-1, 2)))
        cv2.fillConvexPoly(mask, hull, 255.0, lineType=cv2.LINE_AA)  # pylint: disable=no-member
        return mask


def get_source_transforms():
    return alb.Compose([
        alb.Compose([
            alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                   val_shift_limit=(-0.3, 0.3), p=1),
            alb.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1),
        ], p=1),

        alb.OneOf([
            RandomDownScale(p=1),
            alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
        ], p=1),

    ], p=1.)


def get_transforms():
    return alb.Compose([

        alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
        alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3),
                               p=0.3),
        alb.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
        alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),

    ],
        additional_targets={f'image1': 'image'},
        p=1.)


def random_get_hull(landmark,img1):
    hull_type = random.choice([0,1,2,3])
    if hull_type == 0:
        mask = dfl_full(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask/255
    elif hull_type == 1:
        mask = extended(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask/255
    elif hull_type == 2:
        mask = components(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask/255
    elif hull_type == 3:
        mask = facehull(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask/255


def randaffine(img, mask):
    f = alb.Affine(
        translate_percent={'x': (-0.03, 0.03), 'y': (-0.015, 0.015)},
        scale=[0.95, 1 / 0.95],
        fit_output=False,
        p=1)

    g = alb.ElasticTransform(
        alpha=50,
        sigma=7,
        alpha_affine=0,
        p=1,
    )

    transformed = f(image=img, mask=mask)
    img = transformed['image']

    mask = transformed['mask']
    transformed = g(image=img, mask=mask)
    mask = transformed['mask']
    return img, mask


def get_blend_mask(mask):
    H, W = mask.shape
    size_h = np.random.randint(192, 257)
    size_w = np.random.randint(192, 257)
    mask = cv2.resize(mask, (size_w, size_h))
    kernel_1 = random.randrange(5, 26, 2)
    kernel_1 = (kernel_1, kernel_1)
    kernel_2 = random.randrange(5, 26, 2)
    kernel_2 = (kernel_2, kernel_2)

    mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
    mask_blured = mask_blured / (mask_blured.max())
    mask_blured[mask_blured < 1] = 0

    mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5, 46))
    mask_blured = mask_blured / (mask_blured.max())
    mask_blured = cv2.resize(mask_blured, (W, H))

    return mask_blured.reshape((mask_blured.shape + (1,)))


def dynamic_blend(source,target,mask):
    mask_blured = get_blend_mask(mask)
    blend_list=[0.25,0.5,0.75,1,1,1]
    blend_ratio = blend_list[np.random.randint(len(blend_list))]
    mask_blured*=blend_ratio
    img_blended=(mask_blured * source + (1 - mask_blured) * target)

    return img_blended,mask_blured


def self_blending(img, landmark):
    H, W = len(img), len(img[0])

    transforms = get_transforms()
    source_transforms = get_source_transforms()

    mask = random_get_hull(landmark, img)[:, :, 0]
    # mask = np.zeros_like(img[:, :, 0])
    # cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

    source = img.copy()
    img_trans = img.copy()
    if np.random.rand() < 0.5:
        source = source_transforms(image=source.astype(np.uint8))['image']
    else:
        img_trans = source_transforms(image=img.astype(np.uint8))['image']

    source, mask = randaffine(source, mask)

    img_blended, mask = dynamic_blend(source, img_trans, mask)
    img_blended = img_blended.astype(np.uint8)

    transformed = transforms(image=img_blended.astype('uint8'))
    img_blended = transformed['image']

    return img_blended


if __name__ == '__main__':
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    img = cv2.imread('./IMG_20240119_080323.jpg')
    # img = np.array(Image.open('./IMG_20240119_080323.jpg'))
    faces = app.get(img)

    landmarks = list()  # save the landmark
    bboxs = list()
    size_list = list()  # save the size of the detected face
    for face_idx in range(len(faces)):
        landmark = faces[face_idx].landmark_2d_106
        bbox = faces[face_idx].bbox
        x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
        x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
        face_s = (x1 - x0) * (y1 - y0)
        size_list.append(face_s)
        landmarks.append(landmark)
        bboxs.append(bbox)
    # save the landmark with the biggest face
    landmarks = np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
    landmarks = landmarks[np.argsort(np.array(size_list))[::-1]][0]
    bboxs = np.concatenate(bboxs).reshape((len(size_list),) + bbox.shape)
    bboxs = bboxs[np.argsort(np.array(size_list))[::-1]][0]

    img_f = self_blending(img.copy(), landmarks.copy())

    cv2.imwrite('img_f.jpg', img_f)
