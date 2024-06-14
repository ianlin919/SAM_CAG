import random
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
from PIL import Image, ImageFilter

PARAMETER_MAX = 10

#def _float_parameter(v, max_v):
#    return float(v) * max_v / PARAMETER_MAX


#def _int_parameter(v, max_v):
#    return int(v * max_v / PARAMETER_MAX)

def hflip(img, _):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def Rotate(img, v):
    return img.rotate(v)

def Horizontal_Shift(img, v):
    # v > 0, shift right
    # v < 0, shift left
    assert -1 < v and v < 1
    width, height = img.size
    
    img = img.crop((0 - v * width, 0, width - v * width, height))

    return img

def Vertical_Shift(img, v):
    assert -1 < v and v < 1
    # v > 0, shift up
    # v < 0, shift down
    width, height = img.size
    

    img = img.crop((0, v * height, width, height + v * height))

    return img

def Zoom(img, v):
    assert -1 < v and v < 1
    width, height = img.size

    img = img.crop((v * width, v * height, width - v * width, height - v * height))
    img = img.resize((width, height), Image.BILINEAR)

    return img

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Sharpness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Identity(img, _):
    return img


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Posterize(img, v):
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Solarize(img, v):
    return PIL.ImageOps.solarize(img, v)


def Edge_enhance(img, _):
    return img.filter(ImageFilter.EDGE_ENHANCE)


def Edge_enhance_more(img, _):
    return img.filter(ImageFilter.EDGE_ENHANCE_MORE)


def Smooth(img, _):
    return img.filter(ImageFilter.SMOOTH)


def Smooth_more(img, _):
    return img.filter(ImageFilter.SMOOTH_MORE)


def Detail(img, _):
    return img.filter(ImageFilter.DETAIL)


def Blur(img, _):
    return img.filter(ImageFilter.BLUR)


def Sharpen(img, _):
    return img.filter(ImageFilter.SHARPEN)


def Emboss(img, _):
    return img.filter(ImageFilter.EMBOSS)


def Contour(img, _):
    return img.filter(ImageFilter.CONTOUR)


def Find_edges(img, _):
    return img.filter(ImageFilter.FIND_EDGES)


def SolarizeAdd(img, v, threshold=128):
    v = int(v)
    if random.random() > 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def gauss_noise(image, v):
    # row, col, ch = image.shape
    row, col, ch = 512, 512, 1
    mean, var = 0, v
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy

def pass_aug(img, _):
    return img

def augment_list_senior_13():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 1.2),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Posterize, 4, 8),
            (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            #(Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            # (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            # (Smooth_more, 0, 1),
            # (Sharpen, 0, 1),
            #(Identity, 0, 1),
            ]
    return augs

def augment_list():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 1.2),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Posterize, 4, 8),
            (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            #(Emboss, 0, 1),
            #(Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            (Smooth_more, 0, 1),
            (Sharpen, 0, 1),
            #(Identity, 0, 1),
            ]
    return augs

def augment_list1():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 3.0),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Posterize, 4, 8),
            (Sharpness, 0.1, 2.0), # (Sharpness, 0.5, 5.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            (Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            (Smooth_more, 0, 1),
            (Sharpen, 0, 1),
            #(Identity, 0, 1),
            # (gauss_noise, 0, 0.01), # 2021/12/28 加的
            ]
    return augs

def augment_list_16():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 3.0),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            # (Posterize, 4, 8),
            (Sharpness, 0.5, 5.0), # (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            (Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            (Smooth_more, 0, 1),
            (Sharpen, 0, 1),
            #(Identity, 0, 1),
            ]
    return augs

def augment_list_17():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 3.0),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            # (Posterize, 4, 8),
            (Sharpness, 0.5, 5.0), # (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            (Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            (Smooth_more, 0, 1),
            (Sharpen, 0, 1),
            #(Identity, 0, 1),
            (gauss_noise, 0, 0.01),
            ]
    return augs

def augment_list_18():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 3.0),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Posterize, 4, 8),
            (Sharpness, 0.5, 5.0), # (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            (Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            (Smooth_more, 0, 1),
            (Sharpen, 0, 1),
            #(Identity, 0, 1),
            (gauss_noise, 0, 0.01),
            ]
    return augs


def augment_list_best_9Aug():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 3.0),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            # (Posterize, 4, 8),
            # (Sharpness, 0.5, 5.0), # (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            # (Detail, 0, 1),
            (Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            # (Edge_enhance, 0, 1),
            # (Edge_enhance_more, 0, 1),
            # (Smooth, 0, 1),
            # (Smooth_more, 0, 1),
            # (Sharpen, 0, 1),
            #(Identity, 0, 1),
            ]
    return augs

def augment_list_best_1Aug():
    # Test
    augs = [(Contour, 0, 1),]
    return augs

def augment_list_new_best_2Aug():
    # Test
    augs = [(SolarizeAdd, 0, 80),
            (Contour, 0, 1),
            ]
    return augs

def augment_list_new_best_3Aug():
    # Test
    augs = [(SolarizeAdd, 0, 80),
            (Contour, 0, 1),
            (Invert, 0, 1),
            ]
    return augs    

def augment_list_new_best_4Aug():
    # Test
    augs = [(SolarizeAdd, 0, 80),
            (Contour, 0, 1),
            (Invert, 0, 1),
            (Blur, 0, 1),
            ]
    return augs    

def augment_list_best_2Aug():
    # Test
    augs = [(Blur, 0, 1),
            (Contour, 0, 1),
            ]
    return augs

def augment_list_best_3Aug():
    # Test
    augs = [(Blur, 0, 1),
            (Contour, 0, 1),
            (SolarizeAdd, 0, 80),
            ]
    return augs

def augment_list_best_4Aug():
    # Test
    augs = [(Blur, 0, 1),
            (Contour, 0, 1),
            (SolarizeAdd, 0, 80),
            (Invert, 0, 1),
            ]
    return augs

def augment_list_3Aug():
    # Test
    augs = [(Blur, 0, 1),
            (Contour, 0, 1),
            (Posterize, 4, 8),
            ]
    return augs

def augment_list_gauss_noise():
    # Test
    augs = [(gauss_noise, 0, 0.01),
            ]
    return augs

def augment_list_gauss_noise_Contour():
    # Test
    augs = [(gauss_noise, 0, 0.01),
            (Contour, 0, 1),
            ]
    return augs

def augment_list_gauss_noise_Contour_Blur():
    # Test
    augs = [(gauss_noise, 0, 0.01),
            (Contour, 0, 1),
            (Blur, 0, 1),
            ]
    return augs

def augment_list_Detail():
    # Test
    augs = [(Detail, 0, 1),
            ]
    return augs

def augment_list_space():
    augs = [(hflip, 0 , 1),
            (Rotate, -20, 20),
            (Horizontal_Shift, -0.1, 0.1),
            (Vertical_Shift, -0.1, 0.1),
            (Zoom, 0, 0.1),
            ]
    return augs

def augment_list_space_2aug():
    augs = [(hflip, 0 , 1),
            (Rotate, -20, 20),
            ]
    return augs

def augment_list_filter_2aug_space_3aug():
    augs = [(Contour, 0, 1),
            (Blur, 0, 1),
            (hflip, 0 , 1),
            (Rotate, -20, 20),
            (Zoom, 0, 0.1),
            ]
    return augs

def augment_list_M():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 1.2),
            ]
    return augs

def augment_list_senior_13_80_percent():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 1.2),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Posterize, 4, 8),
            (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            # (Blur, 0, 1),
            (Detail, 0, 1),
            #(Emboss, 0, 1),
            # (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            # (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            # (Smooth_more, 0, 1),
            # (Sharpen, 0, 1),
            #(Identity, 0, 1),
            ] + [
            (Blur, 0, 1),
            ] * 22 + [
            (Contour, 0, 1),
            ] * 22
    return augs

class RandAugment_senior_13_withspace:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_senior_13()
        self.spacelist = augment_list_space()

    def __call__(self, img):
        ops_filter = random.choices(self.augment_list, k=self.n)
        ops_space = random.choices(self.spacelist, k=self.n)

        for op, min_val, max_val in ops_filter:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        for op, min_val, max_val in ops_space:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_senior_13:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_senior_13()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_half_13:
    def __init__(self, n):
        self.n = n
        #self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_senior_13()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            if random.random() > 0.5:
                val = round(min_val + float(max_val - min_val) * random.random(), 2)
                img = op(img, val)

        return img

class RandAugment_half_15:
    def __init__(self, n):
        self.n = n
        #self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            if random.random() > 0.5:
                val = round(min_val + float(max_val - min_val) * random.random(), 2)
                img = op(img, val)

        return img

class RandAugment1:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list1()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_15:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_16:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_16()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_17:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_17()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

    
class RandAugment_18:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_18()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img


class RandAugment_single:
    def __init__(self, n, m,aug_num):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list1()
        self.aug_num = aug_num

    def __call__(self, img):
        # ops = random.choices(self.augment_list, k=self.n)
        ops = [self.augment_list[self.aug_num]]
        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_1aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_best_1Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_new_best_2aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_new_best_2Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_new_best_3aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_new_best_3Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_new_best_4aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_new_best_4Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_best_2aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_best_2Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_best_3aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_best_3Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_best_4aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_best_4Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_3aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_3Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_gauss_noise:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_gauss_noise()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_gauss_noise_Contour:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_gauss_noise_Contour()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_gauss_noise_Contour_Blur:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_gauss_noise_Contour_Blur()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_Detail:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_Detail()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img


class RandAugment_best9aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_best_9Aug()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_space_work_q:
    def __init__(self):
        self.work_q = []

    def append(self, op, val):
        self.work_q.append((op, val))

    def pop(self):
        return self.work_q.pop(0)
    
    def get_work_q_list(self):
        return self.work_q
    
    def load_work_q_list(self, op):
        self.work_q = op

class RandAugment_best_2aug_space_augment:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        # FIXME: space augmentation
        self.augment_list_filter = augment_list_best_2Aug()
        self.augment_list_space = augment_list_space()

    def __call__(self, img):
        ops_filter = random.choices(self.augment_list_filter, k=self.n)
        ops_space = random.choices(self.augment_list_space, k=self.n)

        for op, min_val, max_val in ops_filter:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        for op, min_val, max_val in ops_space:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img

class RandAugment_best_2aug_space_augment_2aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list_filter = augment_list_best_2Aug()
        self.augment_list_space = augment_list_space_2aug()

    def __call__(self, img):
        assert self.n == 1  # not support multiple choices
        ops_filter = random.choices(self.augment_list_filter, k=self.n)
        ops_space = random.choices(self.augment_list_space, k=self.n)

        for op, min_val, max_val in ops_filter:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        for op, min_val, max_val in ops_space:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img, (op, val)

class RandAugment_space_augment_2aug:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list_space = augment_list_space_2aug()

    def __call__(self, img):
        assert self.n == 1  # not support multiple choices
        ops_space = random.choices(self.augment_list_space, k=self.n)

        for op, min_val, max_val in ops_space:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img, (op, val)

class RandAugment_filter_space_select:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_filter_2aug_space_3aug()

    def __call__(self, img):
        assert self.n == 1  # not support multiple choices
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        # filter aug not need use `op` function
        if op == Contour or op == Blur:
            op = pass_aug

        return img, (op, val)

class RandAugmentM:
    def __init__(self, n):
        self.n = n
        #self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_M()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:            
            factor1 = np.random.randint(0, (max_val - min_val)/0.1)
            factor2 = np.random.randint(0, (max_val - min_val)/0.1)
            
            val1 = min_val + 0.1 * factor1
            val2 = min_val + 0.1 * factor2
            img1 = op(img, val1)
            img2 = op(img, val2)

        return img1, img2

class RandAugment_senior_13_80_percent:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list_senior_13_80_percent()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)

        for op, min_val, max_val in ops:
            # randomly choose a int from 0 to m
            factor = np.random.randint(0, self.m)
            # generate augmentation method's param
            val = round(min_val + float(max_val - min_val) * (factor/self.m), 2)
            img = op(img, val)

        return img