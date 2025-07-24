
import urllib.request as urllib
import cv2
import numpy as np

# 读取图像文件
url = 'http://localhost:5000/ORIGIN_IMAGE_20230621.jpg'
req = urllib.urlopen(url)
image_data = req.read()

# 将图像数据解码为彩色图像
color_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

# 转换为灰度图像
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
print(gray_image)
# import cv2
# import requests
# from PIL import Image
# from io import BytesIO

# # 图片 URL
# image_url = "http://localhost:5000/ORIGIN_IMAGE_20230621.jpg"

# # 发送 HTTP 请求获取图片数据
# response = requests.get(image_url)

# # 将图片数据读取为二进制格式
# image_data = response.content

# # 打开图片
# image = Image.open(BytesIO(image_data))

# # 获取图像信息
# width, height = image.size
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print(gray_image)
# # 显示图像信息
# print(f"图像宽度: {width}")
# print(f"图像高度: {height}")

# # 处理图像数据
# resp = urllib.urlopen(url)
# image = np.asarray(bytearray(resp.read()), dtype="uint8")
# print("look heressssssss")
# print(image.size)
# image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
# print(image)