import os
from time import sleep

import torch
import algorithm.cnn_dense.pretreatment as pretreatment
import algorithm.cnn_dense.cnn as cnn
import configs.index as conf
import util.resource as resourceUtil
import requests
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
analysed_rank = 1


class FILE_TYPE:
    ANN = 'ANN'
    FREQUENCY = 'FREQUENCY'


def RecordValue(value, doc_name, _type, excel_name,ip):
    """
        RecordValue
        cache sumQ and Ann(5), you can see the issue to understand what Ann(5) means
    """
    cache_file = open( conf.CACHE_POSITION + ip+"_" + doc_name, 'a+')
    cache_file.seek(0)
    content = cache_file.read()
    if content == '':
        cache_file.write(str([value]))
    else:
        cache_queue = eval(content)
        # SumQ 值满 5 个后计算 1 个 Ann 值
        # Ann 值满 30 个后对文件进行存储
        if _type == 'SumQ':
            
            if cache_queue.__len__() == 5:
                # 满 5 更新最新后计算 Ann(5)
                del(cache_queue[0])
                cache_queue.append(value)
                ann = 0
                for i in range(cache_queue.__len__()):
                    ann += cache_queue[i]
                RecordValue(value=ann, doc_name='Ann5.txt', _type='Ann', excel_name=excel_name,ip=ip)
            else:
                cache_queue.append(value)
        if _type == 'Ann':
            if cache_queue.__len__() == 40:
                # 满 30 上传
                del(cache_queue[0])
                cache_queue.append(value)
                
            else:
                cache_queue.append(value)
                resourceUtil.CreateExcel(data={"Rank1": cache_queue}, path=excel_name)
                resourceUtil.ResourceSave(excel_name,ip=ip, file_type=FILE_TYPE.ANN, delete=True)
        cache_file.seek(0)
        cache_file.truncate()
        cache_file.write(str(cache_queue))
    cache_file.close()


def PreSolve(name, image,ip):
    """
        PreSolve
        pre solve the schumman data in cnn_dense module
    """
    # analyse the schumman image
    [frequencies, SFDs] = pretreatment.extractFrequency(image)
    xlsData = {}
    for rank in range(frequencies.__len__()):
        xlsData["f" + str(rank + 1)] = frequencies[rank]
        
    xlsData["db"]="Schumman"
    json_data = json.dumps(xlsData)
    headers = {"Content-Type": "application/json"}
    # response = requests.post('http://localhost:5000/v1/module/resource/create', data=json_data, headers=headers)
    resourceUtil.CreateExcel(data=xlsData, path=name)
    resourceUtil.ResourceSave(name, ip=ip,delete=True, file_type=FILE_TYPE.FREQUENCY)
    return SFDs


def CnnDense(ip,name, image):
    """
        CnnDense
        main logic of the cnn_dense module
    """
    xlsName = name + '.xls'
    frequency_xls_path = './' + FILE_TYPE.FREQUENCY + '_' + xlsName
    ann_xls_path = './' + FILE_TYPE.ANN + '_' + xlsName
    SFDs = PreSolve(frequency_xls_path, image,ip)
    # then use model
    model = cnn.ConvSelector(mode='CutUsage', deleteNum=3).to(device)
    model.load_state_dict(torch.load('./algorithm/cnn_dense/model/20230211_1d_cnn_fc_rank1_1x.pkl'))
    matrix = torch.tensor([SFDs[analysed_rank - 1]]).to(device)
    matrix = torch.unsqueeze(matrix, 0)
    out = model(matrix)
    sumQ = 0    
    for i in range(out.__len__()):
        sumQ += abs(out[i].item())
    RecordValue(value=sumQ, doc_name='sumQ.txt', _type='SumQ', excel_name=ann_xls_path,ip=ip)
