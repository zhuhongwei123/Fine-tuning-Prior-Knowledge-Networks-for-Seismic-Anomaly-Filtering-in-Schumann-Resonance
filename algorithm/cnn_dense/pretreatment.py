import util

constNormalFreq = [7.8, 13.8, 19.7]


def extractFrequency(image):
    freq = [[], [], []]
    SFDs = []
    for x in range(640):
        # B G R
        r1L, r2L, r3L = 0, 0, 0
        maxAmplitude = [0, 0, 0]
        for y in range(560):
            pixImg = image[y][x]
            # 一阶舒曼谐振
            if 500 >= y > 440 and pixImg >= maxAmplitude[0]:
                maxAmplitude[0] = pixImg
                r1L = (560 - y) * 50 / 560
            # 二阶舒曼谐振
            if 440 >= y > 370 and pixImg >= maxAmplitude[1]:
                maxAmplitude[1] = pixImg
                r2L = (560 - y) * 50 / 560
            # 三阶舒曼谐振
            if 370 >= y > 300 and pixImg >= maxAmplitude[2]:
                maxAmplitude[2] = pixImg
                r3L = (560 - y) * 50 / 560

        if r1L < 7 or r1L > 9:
            r1L = 7.8
        if r2L < 12 or r2L > 15:
            r2L = 13.8
        if r3L < 18 or r3L > 22:
            r3L = 19.7
        freq[0].append(r1L)
        freq[1].append(r2L)
        freq[2].append(r3L)
    slidedData = util.dataSlider2(freq, constNormalFreq, 5)
    for r in range(len(freq)):
        # 获取滑动四分位距法阈值
        # 12 个像素点相当于以半小时为粒度
        up, down = util.getSFD_distance(slidedData[r], 1.73, True, 12, 12)
        # 获取滑动四分位距法阈值与数据的欧式距离
        dataSFDed = util.calcSFD_EuclideanDistance(slidedData[r], up, down)
        SFDs.append(dataSFDed)
    return [freq, SFDs]
