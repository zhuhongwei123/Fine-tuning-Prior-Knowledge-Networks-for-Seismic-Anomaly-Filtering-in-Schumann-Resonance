import main.server as main
import uuid
from configs.index import MISSION_STATE
import cv2
import urllib.request as urllib
import numpy as np
import algorithm.cnn_dense as cnn_dense
import algorithm.autoencoder as auto_encoder
from threading import Thread
import requests
from io import BytesIO
from PIL import Image
missions = {}


def get_uuid():
    get_timestamp_uuid = uuid.uuid1()
    return get_timestamp_uuid


class SchummanAnalyserHandler(main.t_web.RequestHandler, main.ABC):
    """
        SchummanAnalyserHandler
        Post method only
        params:
            time - the time of schumman data
            url - the url of original schumman data
    """
    def get(self):
        self.set_status(404)

    def post(self):
        p_time = self.get_body_argument('time')
        p_url = self.get_body_argument('url')
        p_ip = self.get_body_argument('ip')
        """
            Tip: 理论上此处接口应完善成任务处理状态的保存，并且可以通过查询接口获取当前任务的状态
                 并且定期每日24点清除已完成的任务 id，避免内存溢出
                 从简 ----- 仅做触发
        """
        _uuid = str(get_uuid())
        missions[_uuid] = MISSION_STATE.ANALYSING

        """
            Main logic
            Here not combined too much conditional judgment.
            Because this server will only used by Device-Pi.
            
            Besides, we should use the format of "${deviceName}_${area}_${time}" to save the result xls
        """
        if p_url.startswith('https://') or p_url.startswith("http://"):
            # resp = urllib.urlopen(p_url)
            params ={
                'type':'ORIGIN_IMAGE',
                'name':'ORIGIN_IMAGE'+'_'+ p_time +'.jpg',
                'urlt': p_ip
            }
            resp = requests.get('http://localhost:5000/v1/module/resource/get',params = params)
            
            # img_data = resp.content  # 获取响应数据
            # image = Image.open(BytesIO(img_data))  # 将响应数据转换为图像对象

            #image.show()
            image = np.asarray(bytearray(resp.content), dtype="uint8")
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            cnn_thread = Thread(target=cnn_dense.CnnDense, args=(p_ip,p_time, image))
            autoEncoder_thread = Thread(target=auto_encoder.AutoEncoder, args=(p_ip,p_time, image))

            cnn_thread.start()
            autoEncoder_thread.start()

            self.write({
                'id': _uuid
            })
        else:
            self.write({
                'msg': 'Url not legal.'
            })


class MissionFetcherHandler(main.t_web.RequestHandler, main.ABC):
    """
        MissionFetcherHandler
        Get method only
        params:
            id - mission id, which fetched from API '/schumman_analyse'
    """
    def get(self):
        _id = self.get_argument('id')
        try:
            _state = missions[_id]
            self.write({
                "id": _id,
                "state": _state,
            })
        except Exception as e:
            print(e)
            self.write({
                "msg": "Mission is not exist."
            })


    def post(self):
        self.set_status(404)
