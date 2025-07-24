import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cmath
import os


def dImg(image, direction='x'):
    """
        :param image: image array
        :param direction: x or y
    """
    y = image.__len__()
    x = image[0].__len__()
    newImg = np.zeros((y - 1, y - 1), dtype=float)

    if direction == 'x':
        for i in range(1, y):
            for j in range(0, y - 1):
                newImg[i - 1][y - j - 1 - 1] = float(image[i][y - j - 1]) - float(image[i][y - j - 1 - 1])
    else:
        for i in range(0, y - 1):
            for j in range(1, x):
                newImg[i][j - 1] = float(image[i][j]) - float(image[i + 1][j])

    return newImg


def normalize(image):
    _max = 0
    _min = 999
    for i in range(0, image.__len__()):
        for j in range(0, image[0].__len__()):
            if _max < image[i][j]:
                _max = image[i][j]
            if _min > image[i][j]:
                _min = image[i][j]
    return (image - _min) / _max


def deNormalize(image):
    _max = -999
    _min = 999
    for i in range(0, image.__len__()):
        for j in range(0, image[0].__len__()):
            if _max < image[i][j]:
                _max = image[i][j]
            if _min > image[i][j]:
                _min = image[i][j]
    return 255 * (image - _min) / (_max - _min)


def operateList(l, operation='add'):
    res = 0 + 0j
    _real_over = 0
    _imag_over = 0
    if operation == 'add':
        for m in l:
            res += m
        return res, _real_over, _imag_over
    if operation == 'mul':
        res = 1 + 0j
        for m in l:
            res = res * m
            while 1:
                if (abs(res.real) > 1e10 or abs(res.imag) > 1e10 or abs(res.real) < (1 / 1e10) or abs(res.imag) < (1 / 1e10)) and abs(res.real) != 0 and abs(res.imag) != 0:
                    if abs(res.real) > 1e10:
                        _real_over += 1
                        res = res.real / 1e10 + res.imag * 1j
                    if abs(res.imag) > 1e10:
                        _imag_over += 1
                        res = res.real + res.imag / 1e10 * 1j
                    if abs(res.real) < (1 / 1e10):
                        _real_over -= 1
                        res = res.real * 1e10 + res.imag * 1j
                    if abs(res.imag) < (1 / 1e10):
                        _imag_over -= 1
                        res = res.real + res.imag * 1e10 * 1j
                else:
                    break
        _realDirection = 1 if res.real > 0 else -1
        _imagDirection = 1 if res.imag > 0 else -1
        return _realDirection, _imagDirection, _real_over, _imag_over

