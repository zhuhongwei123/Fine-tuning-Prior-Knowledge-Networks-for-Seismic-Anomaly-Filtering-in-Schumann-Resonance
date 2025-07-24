import pandas as pd
import requests
import os


def ResourceSave(path, ip, file_type='FREQUENCY', delete=False):
    """
        ResourceSave
        use for saving resource to target server
    """
    requests.post(
        'http://localhost:5000/v1/module/resource/save',
        files={'file': open(path, 'rb')},
        data={'type': file_type ,'ip':ip},
    )
    if delete:
        os.remove(path)


def CreateExcel(data, path):
    """
        CreateExcel
        :param data: [key]:[value]
        :param path: the path of excel to save
    """
    dataFrame = pd.DataFrame(
        data=data,
        columns=data.keys(),
        index=None
    )
    data = pd.DataFrame(dataFrame)
    writer = pd.ExcelWriter(path)
    data.to_excel(writer, 'database', float_format='%.5f')
    writer.save()
    writer.close()
