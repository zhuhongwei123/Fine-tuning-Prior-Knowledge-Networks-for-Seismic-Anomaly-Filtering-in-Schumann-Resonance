import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ##
# @description 工具函数
# @date 2021-11-14
# @author 206149 阙名毅
# ##
#
#

"""
    Tip: Sorry... I know that the code here is bad...
         But I really do not want to refactor it...
"""

subRange = [[7, 9], [13, 15], [19, 21], [25, 30], [29, 33]]


def quickSort(array, low, high):
    arr = array
    left = low
    right = high
    middle = arr[left]
    if left < right:
        while left < right:
            while left < right:
                if arr[right] >= middle:
                    right -= 1
                else:
                    arr[left] = arr[right]
                    break
            while left < right:
                if arr[left] <= middle:
                    left += 1
                else:
                    arr[right] = arr[left]
                    break
            if left == right:
                arr[left] = middle
                break
        quickSort(arr, low, left - 1)
        quickSort(arr, right + 1, high)
    return arr


def normalize(img):
    vmin = np.min(img)
    vmax = np.max(img)
    return (img - vmin) / (vmax - vmin) * 255


def ArrNormalize(data):
    retData = []
    maxNum = max(data)
    minNum = min(data)
    for i in range(len(data)):
        retData.append((data[i] - minNum) / (maxNum - minNum))
    return retData


def fft(x):
    n = len(x)
    if n == 2:
        return [x[0] + x[1], x[0] - x[1]]

    G = fft(x[::2])
    H = fft(x[1::2])
    W = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    WH = W * H
    X = np.concatenate([G + WH, G - WH])
    return X


def fft2(img):
    h, w = img.shape
    img = normalize(img)
    res = np.zeros([h, w], 'complex128')
    for i in range(h):
        res[i, :] = fft(img[i, :])
    for j in range(w):
        res[:, j] = fft(res[:, j])
    return res


# 计算均值
def arrAverage(arr):
    arrSum = 0
    l = len(arr)
    for i in range(l):
        arrSum += arr[i]
    return arrSum / l


# 平滑处理
def dataSlider(data, pedding, step=3):
    retData = []
    dataLen = len(data)

    for i in range(dataLen):
        retData.append([])
        for rank in range(len(data[i])):
            averageMember = []
            if (i+1) < step:
                for ped in range(step - 1 - i):
                    averageMember.append(pedding[rank])
                for s in range(step - len(averageMember)):
                    averageMember.append(data[i-s][rank])
            else:
                for s in range(step):
                    averageMember.append(data[i-s-1][rank])
            retData[i].append(average(averageMember))
    return retData


# 平滑处理
def dataSlider2(data, pedding, step=3):
    retData = []
    dataLen = len(data)

    for i in range(dataLen):
        retData.append([])
        for j in range(len(data[i])):
            averageMember = []
            if j < step:
                for time in range(0, step - j - 1):
                    averageMember.append(pedding[i])
                for last in range(0, j + 1):
                    averageMember.append(data[i][j - last])
            else:
                for last in range(0, step):
                    averageMember.append(data[i][j - last])
            retData[i].append(average(averageMember))
    return retData


def average(data):
    numSum = 0
    for d in data:
        numSum += d
    return numSum / len(data)


# 一阶差分
def delta1(arr):
    newArr = []
    l = len(arr)
    for i in range(l - 1):
        newArr.append(arr[i + 1] - arr[i])
    return newArr


# 获取滑动四分位距标准值
def getSFD_distance(arr, ration, slide=False, initial=16, slideDays=1):
    # 深拷贝
    copy = arr.copy()
    up = []
    down = []
    if slide:
        for i in range(int(int(len(copy)) - initial) + 1):
            # 首先对数组进行排序
            temp = copy[i:i + initial]
            temp.sort()
            # 四等分化
            # Q3 25% Q2 50% Q1 75%
            if initial % 4 == 0:
                Q3 = temp[int(initial / 4)]
                Q2 = temp[int(initial * 2 / 4)]
                Q1 = temp[int(initial * 3 / 4)]
            else:
                Q3 = (temp[int(initial / 4)] + temp[
                    int(initial / 4) + slideDays]) / 2
                Q2 = (temp[int(initial * 2 / 4)] + temp[
                    int(initial * 2 / 4) + slideDays]) / 2
                Q1 = (temp[int(initial * 3 / 4)] + temp[
                    int(initial * 3 / 4) + slideDays]) / 2

            # 计算IQR
            IQR = Q1 - Q3
            if i == int(int(len(copy)) - initial):
                for j in range(int(initial / 2)):
                    up.append(Q2 + ration * IQR)
                    down.append(Q2 - ration * IQR)

            if i == 0:
                for j in range(initial - int(initial / 2)):
                    up.append(Q2 + ration * IQR)
                    down.append(Q2 - ration * IQR)
            else:
                up.append(Q2 + ration * IQR)
                down.append(Q2 - ration * IQR)
        return up, down

    else:
        # 四等分化
        # Q3 25% Q2 50% Q1 75%
        copy.sort()
        if len(copy) % 4 == 0:
            Q3 = copy[int(len(copy) / 4)]
            Q2 = copy[int(len(copy) * 2 / 4)]
            Q1 = copy[int(len(copy) * 3 / 4)]
        else:
            Q3 = (copy[int(len(copy) / 4)] + copy[int(len(copy) / 4) + 1]) / 2
            Q2 = (copy[int(len(copy) * 2 / 4)] + copy[int(len(copy) * 2 / 4) + 1]) / 2
            Q1 = (copy[int(len(copy) * 3 / 4)] + copy[int(len(copy) * 3 / 4) + 1]) / 2

        # 计算IQR
        IQR = Q1 - Q3
        for j in range(copy.__len__()):
            up.append(Q2 + ration * IQR)
            down.append(Q2 - ration * IQR)
        return up, down


# 由月份和日的时间转换为天数
def calcDaysByDate(time):
    result = 0
    dayNums = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = int(time[5:7])
    day = int(time[8:10])
    for i in dayNums[0:month - 1]:
        result += i
    result += day
    return result


# 由月份和日的时间转换为clip等分时间
def calcTimeByDate(time, clip, pre=0):
    result = 0
    dayNums = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = int(time[5:7])
    day = int(time[8:10])
    hour = int(time[11:13])
    for i in dayNums[0:month - 1]:
        result += i * clip
    result += (day - 1) * clip
    result += int(hour * clip / 24)
    return result - pre


# 返回两个字符的日期
def backCompleteDate(num):
    if num < 10:
        return '0' + str(num)
    else:
        return str(num)


# 滑动平均做曲线平滑化
def smoothly(arr, smooth=3, rank=0):
    arrLen = len(arr)
    ret = []
    for s in range(arrLen - smooth):
        temp = 0
        for i in range(smooth):
            if arr[s+i] < subRange[rank][0] or arr[s+i] > subRange[rank][1]:
                temp += (subRange[rank][0] + subRange[rank][1]) / 2
            else:
                temp += arr[s + i]
        ret.append(temp / smooth)
    return ret


# 计算差分
def diff(arr, rank=0):
    ret = [arr.copy()]
    for r in range(rank):
        temp = []
        for i in range(ret[r].__len__() - 1):
            temp.append(ret[r][i + 1] - ret[r][i])
        ret.append(temp.copy())
    return ret[rank]


def calcSFD_EuclideanDistance(data, up, down):
    retData = []
    for (index, num) in enumerate(data):
        if up[index] >= num >= down[index]:
            retData.append(0)
        else:
            if up[index] < num:
                retData.append(float(round(num - up[index], 3)))
            if num < down[index]:
                retData.append(float(round(down[index] - num, 3)))
    return retData


def normalize_v1(arr):
    _max = -99999
    _min = 99999
    for i in range(0, arr.__len__()):
        if _max < arr[i]:
            _max = arr[i]
        if _min > arr[i]:
            _min = arr[i]

    mon = abs(_min) + abs(_max)

    if mon == 0: return arr
    for i in range(0, arr.__len__()):
        arr[i] = (arr[i] - _min) / mon
    return arr


def normalize_v2(image):
    _max = 0
    _min = 999
    for i in range(0, image.__len__()):
        for j in range(0, image[0].__len__()):
            if _max < image[i][j]:
                _max = image[i][j]
            if _min > image[i][j]:
                _min = image[i][j]
    return (image - _min) / _max


def deNormalize_v2(image):
    _max = -999
    _min = 999
    for i in range(0, image.__len__()):
        for j in range(0, image[0].__len__()):
            if _max < image[i][j]:
                _max = image[i][j]
            if _min > image[i][j]:
                _min = image[i][j]
    return 255 * (image - _min) / (_max - _min)


def drawImage(y_data, x_data, vertical_lines=[], labels=[u'频率/Hz', u'时间'], save_name='image.svg', interval=25):
    # 正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 正常显示负号
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(20, 5))
    plt.plot(x_data, y_data)
    plt.ylabel(labels[0], fontproperties='SimHei', fontsize=20)
    plt.xlabel(labels[1], fontproperties='SimHei', fontsize=20)
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    # 关闭科学记数法
    plt.ticklabel_format(style='plain', useOffset=False, axis='y')

    # 设置x轴参数
    # xfmt = matplotlib.dates.DateFormatter('%m-%d')
    # x_major_locator = plt.MultipleLocator(interval)

    # 获取坐标轴实例
    ax = plt.gca()
    # ax.xaxis.set_major_formatter(xfmt)
    # ax.xaxis.set_major_locator(x_major_locator)
    # 设置横坐标垂线
    for i in range(len(vertical_lines)):
        ax.axvline(x=vertical_lines[i], ymin=0.05, ymax=0.95, color='red')
    # 设置图例
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, format='svg')
