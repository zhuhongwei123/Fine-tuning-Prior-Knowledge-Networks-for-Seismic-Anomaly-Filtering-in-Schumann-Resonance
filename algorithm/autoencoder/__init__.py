import torch
import numpy as np
import algorithm.autoencoder.unet_autoencoder as ua
import algorithm.autoencoder.util.util as util
import util.resource as resourceUtil
import configs.index as conf

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
analysed_rank = 1


class FILE_TYPE:
    AUTO_ENCODER = 'AUTOENCODER'


headers = [
    ['img_det', 'img_order', 'img_trace'],
    ['dt_det', 'dt_order', 'dt_trace'],
    ['d2t_det', 'd2t_order', 'd2t_trace'],
    ['df_det', 'df_order', 'df_trace'],
    ['d2f_det', 'd2f_order', 'd2f_trace'],
    ['fx1_det', 'fx1_order', 'fx1_trace'],
    ['fx2_det', 'fx2_order', 'fx2_trace'],
]


def RecordValue(value,ip, doc_name, _type, excel_name):
    """
        RecordValue
        cache values, once the record length >
    """
    cache_file = open(conf.CACHE_POSITION +ip+"_"+ doc_name, 'a+')
    cache_file.seek(0)
    content = cache_file.read()
    if content == '':
        cache_file.write(str([value]))
    else:
        cache_queue = eval(content)
        # record 值满 30 个后对文件进行存储
        if cache_queue.__len__() == 30:
            del (cache_queue[0])
            cache_queue.append(value)
            excel_data = {}
            # 拉平特征值
            values = [
                [[], [], []],  # img
                [[], [], []],  # dt
                [[], [], []],  # d2t
                [[], [], []],  # df
                [[], [], []],  # d2f
                [[], [], []],  # fx1
                [[], [], []],  # fx2
            ]
            for i in range(cache_queue.__len__()):
                sub_record = cache_queue[i]
                for j in range(sub_record.__len__()):
                    for k in range(sub_record[j].__len__()):
                        values[j][k].append(sub_record[j][k][0])
            for i in range(values.__len__()):
                for j in range(values[i].__len__()):
                    excel_data[
                        headers[i][j]
                    ] = values[i][j]
            resourceUtil.CreateExcel(data=excel_data, path=excel_name)
            resourceUtil.ResourceSave(excel_name,ip=ip, file_type=FILE_TYPE.AUTO_ENCODER, delete=True)
        else:
            cache_queue.append(value)
            excel_data = {}
            # 拉平特征值
            values = [
                [[], [], []],  # img
                [[], [], []],  # dt
                [[], [], []],  # d2t
                [[], [], []],  # df
                [[], [], []],  # d2f
                [[], [], []],  # fx1
                [[], [], []],  # fx2
            ]
            for i in range(cache_queue.__len__()):
                sub_record = cache_queue[i]
                for j in range(sub_record.__len__()):
                    for k in range(sub_record[j].__len__()):
                        values[j][k].append(sub_record[j][k][0])
            for i in range(values.__len__()):
                for j in range(values[i].__len__()):
                    excel_data[
                        headers[i][j]
                    ] = values[i][j]
            resourceUtil.CreateExcel(data=excel_data, path=excel_name)
            resourceUtil.ResourceSave(excel_name,ip=ip, file_type=FILE_TYPE.AUTO_ENCODER, delete=True)
            print(cache_queue.__len__())
        cache_file.seek(0)
        cache_file.truncate()
        cache_file.write(str(cache_queue))
    cache_file.close()


def PreSolve(image):
    """
        PreSolve
        pre solve the schumman data in AutoEncoder module
        we should clip the image in fact
    """
    return image[:, 80:640]


def AutoEncoder(ip,name, image):
    """
        AutoEncoder
        main logic of the AutoEncoder module
    """
    values = [
        [[], [], []],  # img
        [[], [], []],  # dt
        [[], [], []],  # d2t
        [[], [], []],  # df
        [[], [], []],  # d2f
        [[], [], []],  # fx1
        [[], [], []],  # fx2
    ]
    xlsName = name + '.xls'
    autoEncoder_xls_path = './' + FILE_TYPE.AUTO_ENCODER + '_' + xlsName

    matrix = PreSolve(image)
    model = ua.CombinedUNet()
    model.load_state_dict(torch.load('./algorithm/autoencoder/model/combinedUNet.pkl'))  # 读取参数

    # 一阶 t 偏差分
    mat = util.dImg(matrix, 'x')
    eigs = np.linalg.eig(mat)[0]
    rank = np.linalg.matrix_rank(mat)
    realDirection, imagDirection, realOver, imagOver = util.operateList(eigs, 'mul')
    values[1][0].append(str(realDirection) + '_' + str(imagDirection) + '_' + str(realOver) + '_' + str(
        imagOver) if rank == mat.__len__() else 0)
    values[1][1].append(rank)
    trace, _1, _2 = util.operateList(eigs, 'add')
    values[1][2].append(trace.real)
    # 二阶 t 偏差分
    mat = util.dImg(mat, 'x')
    eigs = np.linalg.eig(mat)[0]
    rank = np.linalg.matrix_rank(mat)
    realDirection, imagDirection, realOver, imagOver = util.operateList(eigs, 'mul')
    values[2][0].append(str(realDirection) + '_' + str(imagDirection) + '_' + str(realOver) + '_' + str(
        imagOver) if rank == mat.__len__() else 0)
    values[2][1].append(rank)
    trace, _1, _2 = util.operateList(eigs, 'add')
    values[2][2].append(trace.real)

    # 一阶 f 偏差分
    mat = util.dImg(matrix, 'y')
    eigs = np.linalg.eig(mat)[0]
    rank = np.linalg.matrix_rank(mat)
    realDirection, imagDirection, realOver, imagOver = util.operateList(eigs, 'mul')
    values[3][0].append(str(realDirection) + '_' + str(imagDirection) + '_' + str(realOver) + '_' + str(
        imagOver) if rank == mat.__len__() else 0)
    values[3][1].append(rank)
    trace, _1, _2 = util.operateList(eigs, 'add')
    values[3][2].append(trace.real)
    # 二阶 f 偏差分
    mat = util.dImg(mat, 'y')
    eigs = np.linalg.eig(mat)[0]
    rank = np.linalg.matrix_rank(mat)
    realDirection, imagDirection, realOver, imagOver = util.operateList(eigs, 'mul')
    values[4][0].append(str(realDirection) + '_' + str(imagDirection) + '_' + str(realOver) + '_' + str(
        imagOver) if rank == mat.__len__() else 0)
    values[4][1].append(rank)
    trace, _1, _2 = util.operateList(eigs, 'add')
    values[4][2].append(trace.real)

    # f(x)1
    inputData = torch.tensor(matrix).to(torch.float32)
    output, xi, dYs = model(inputData)

    fx1 = dYs[0][1].detach().numpy()
    eigs = np.linalg.eig(fx1)[0]
    rank = np.linalg.matrix_rank(fx1)
    realDirection, imagDirection, realOver, imagOver = util.operateList(eigs, 'mul')
    values[5][0].append(str(realDirection) + '_' + str(imagDirection) + '_' + str(realOver) + '_' + str(
        imagOver) if rank == fx1.__len__() else 0)
    values[5][1].append(rank)
    trace, _1, _2 = util.operateList(eigs, 'add')
    values[5][2].append(trace.real)
    # f(x)2
    fx2 = dYs[0][2].detach().numpy()
    eigs = np.linalg.eig(fx2)[0]
    rank = np.linalg.matrix_rank(fx2)
    realDirection, imagDirection, realOver, imagOver = util.operateList(eigs, 'mul')
    values[6][0].append(str(realDirection) + '_' + str(imagDirection) + '_' + str(realOver) + '_' + str(
        imagOver) if rank == fx2.__len__() else 0)
    values[6][1].append(rank)
    trace, _1, _2 = util.operateList(eigs, 'add')
    values[6][2].append(trace.real)

    # Xi
    eigs = np.linalg.eig(xi[0])[0]
    rank = np.linalg.matrix_rank(xi[0])
    realDirection, imagDirection, realOver, imagOver = util.operateList(eigs, 'mul')
    values[0][0].append(
        str(realDirection) + '_' + str(imagDirection) + '_' + str(realOver) + '_' + str(imagOver) if rank == xi[
            0].__len__() else 0)
    values[0][1].append(rank)
    trace, _1, _2 = util.operateList(eigs, 'add')
    values[0][2].append(trace.real)
    # print(values)
    # print(autoEncoder_xls_path)
    RecordValue(value=values,ip=ip, doc_name='autoEncoder.txt', _type='AutoEncoder', excel_name=autoEncoder_xls_path)

