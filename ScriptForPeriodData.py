import math
import multiprocessing
import random
import matplotlib.pyplot as plt
import time
import logging
import csv






# 保存路径
PATH = '/mnt/storage/COVID19-Huawei/'
# 总人口数量
NUM_PERSON = 7600000
# 起止时间
START_TIME = '2017-01-01 00:00:00'
END_TIME = '2017-01-01 00:01:00'
# 每5s运动范围m
STEP_LENGTH = 5

ZHENGZHOU_FENCE = (113.4441000000000059, 34.5986440000000002, 113.8599200000000025, 34.9642330000000001)

# PROCESS_NUM = multiprocessing.cpu_count()
PROCESS_NUM = 48


# ATTENTION: 用logging.basicConfig在多进程中会有bug，日志文件会被覆盖，日志不全
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%a, %d %b %Y %H:%M:%S',
#                     filename=PATH + 'Generate.log',
#                     filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# 创建一个handler，用于写入日志文件
fh = logging.FileHandler(PATH + 'Generate.log')
fh.setLevel(logging.DEBUG)
# 创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# 将定义好的输出格式添加到handler
fh.setFormatter(logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'))
ch.setFormatter(logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'))
logger.addHandler(fh)
logger.addHandler(ch)



# calculate the distance between two geo points
# distance(0, 1, 0, 2)   # 111194.92664454764 m
def distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two geo points
    :param lat1: latitude of point 1
    :param lon1: longitude of point 1
    :param lat2: latitude of point 2
    :param lon2: longitude of point 2
    :return: distance between two geo points in meter
    """
    p = 0.017453292519943295
    a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    return 12742 * math.asin(math.sqrt(a)) * 1000


# 计算纬度为latDeg degree下, 1m跨度对应于经度的多少跨度, 纬度的多少跨度
# 纵向每度固定111.19492664454764km  => 每米8.993216059188204e-06度
def calcAngleSpanPerMeter(latDeg):
    # 1m横向距离代表的经度跨度
    xSpanDegPerMeter = 1 / distance(latDeg, 0, latDeg, 1)
    # 1m纵向距离代表的纬度跨度
    ySpanDegPerMeter = 8.993216059188204e-06
    return (xSpanDegPerMeter, ySpanDegPerMeter)


# 生成以经纬度x(lon), y(lat)为中心,在纵向最大跨度为deltaY meter, 横向最大跨度为deltaX meter的矩形区域内的随机点位
# 生成的随机点位保证在fence内
# generateNextRandomPoint(113.5, 34.7, 10, 10, (113.0, 34.0, 114.0, 35.0))
def generateNextRandomPoint(x, y, deltaX, deltaY, fenceBox, digit=6):
    f_xMin, f_yMin, f_xMax, f_yMax = fenceBox
    # 判断随机点是否在fence内,不在则抛出错误
    if x < f_xMin or x > f_xMax or y < f_yMin or y > f_yMax:
        raise Exception('输入点位不在fence内')
    xSpanDegPerMeter, ySpanDegPerMeter = calcAngleSpanPerMeter(x)
    xSpanDeg = deltaX * xSpanDegPerMeter
    ySpanDeg = deltaY * ySpanDegPerMeter
    # 随机点位范围
    xMin = x - xSpanDeg
    xMax = x + xSpanDeg
    yMin = y - ySpanDeg
    yMax = y + ySpanDeg
    
    x_next = random.uniform(xMin if xMin > f_xMin else f_xMin, xMax if xMax < f_xMax else f_xMax)
    y_next = random.uniform(yMin if yMin > f_yMin else f_yMin, yMax if yMax < f_yMax else f_yMax)
    return (round(x_next, digit), round(y_next, digit))

# 生成一个在fence内的随机点位
# initialRandomPoint(ZHENGZHOU_FENCE)
def initialRandomPoint(fenceBox, digit=6):
    f_xMin, f_yMin, f_xMax, f_yMax = fenceBox
    x = random.uniform(f_xMin, f_xMax)
    y = random.uniform(f_yMin, f_yMax)
    return (round(x, digit), round(y, digit))

# 指定fence,轨迹点数,步长范围,生成一条轨迹
# points = generateTrackPoints(ZHENGZHOU_FENCE, 1000, 10)
def generateTrackPoints(fenceBox, pointNum, stepLength, digit=6):
    points = []
    currPoint = initialRandomPoint(fenceBox, digit)
    points.append(currPoint)
    for i in range(pointNum - 1):
        currPoint = generateNextRandomPoint(currPoint[0], currPoint[1], stepLength, stepLength, fenceBox, digit)
        points.append(currPoint)
    return points

def generateTracks(trackNum, fenceBox, pointNum, stepLength, digit=6):
    tracks = []
    for i in range(trackNum):
        points = generateTrackPoints(fenceBox, pointNum, stepLength, digit)
        tracks.append(points)
    return tracks

def generateTracksWithMultiProcess(trackNum, fenceBox, pointNum, stepLength, digit=6, processNum=PROCESS_NUM):
    pool = multiprocessing.Pool(processes=processNum)
    # 将trackNum分配给每个进程
    trackNumPerProcess = trackNum // processNum
    if trackNumPerProcess == 0:
        return generateTracks(trackNum, fenceBox, pointNum, stepLength, digit)
    # 创建进程池
    processes = []
    for i in range(processNum):
        if i == processNum - 1:
            processes.append(pool.apply_async(generateTracks, (trackNum % trackNumPerProcess + trackNumPerProcess, fenceBox, pointNum, stepLength, digit)))
        else:
            processes.append(pool.apply_async(generateTracks, (trackNumPerProcess, fenceBox, pointNum, stepLength, digit)))
    pool.close()
    pool.join()
    tracks = []
    for p in processes:
        tracks.extend(p.get())
    return tracks

# visualize the track
def visualizeTrack(tracks, fenceBox):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(fenceBox[0], fenceBox[2])
    ax.set_ylim(fenceBox[1], fenceBox[3])
    ax.set_aspect('equal')
    # 将tracks轨迹可视化
    for i in range(len(tracks)):
        # 选择随机颜色
        color = (random.random(), random.random(), random.random())
        track = tracks[i]
        for j in range(len(track) - 1):
            ax.plot([track[j][0], track[j + 1][0]], [track[j][1], track[j + 1][1]], color=color)
    plt.show()

#时间转为Unix时间戳, 单位秒
def time_to_unix(time_str):
    timeArray = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    unix_time = int(time.mktime(timeArray))
    return unix_time

#生成num个不重复的11位手机号
def generate_phone_number(num):
    # phone_number_list = random.sample(range(1000000000, 9999999999), num)
    phone_number_list = random.sample(range(100000000, 999999999), num)
    phone_number_list = ['1' + str(i) for i in phone_number_list]
    return phone_number_list

#从两个unix时间戳中每5秒取一个时间
def generate_time_list(start_time, end_time):
    time_list = []
    time_list.append(start_time)
    while start_time < end_time:
        start_time += 5
        time_list.append(start_time)

    return time_list

def write_tracks_to_file_worker(tracks, current_tracks_start_index, file_name, time_list, phone_number_list):
    with open(file_name, 'w') as f:
        for j in range(len(tracks)):
            track = tracks[j]
            for k in range(len(track)):
                f.write(time_list[k] + ',' + phone_number_list[j + current_tracks_start_index] + ',' + str(track[k][0]) + ',' + str(track[k][1]) + '\n')
                

# 利用多进程，将tracks写入多个文件
def write_tracks_to_files(tracks, current_tracks_start_index, current_batch_index, file_num, file_name_prefix, time_list, phone_number_list):
    process_num = file_num
    track_num_per_process = len(tracks) // process_num
    if track_num_per_process == 0:
        raise Exception('多进程写入文件要求tracks数量{}大于文件数量{}'.format(len(tracks), file_num))
    processes = []
    for i in range(process_num):
        if i == process_num - 1:
            processes.append(multiprocessing.Process(target=write_tracks_to_file_worker, args=(tracks[i * track_num_per_process:], current_tracks_start_index + i * track_num_per_process, file_name_prefix + str(current_batch_index) + '_' + str(i) + '.csv', time_list, phone_number_list)))
        else:
            processes.append(multiprocessing.Process(target=write_tracks_to_file_worker, args=(tracks[i * track_num_per_process:(i + 1) * track_num_per_process], current_tracks_start_index + i * track_num_per_process, file_name_prefix + str(current_batch_index) + '_' + str(i) + '.csv', time_list, phone_number_list)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()
   


def generate_signals_to_file(person_num, start_time, end_time, step_length, fence_box, file_name, digits=6):
    #生成时戳列表
    time_list = generate_time_list(start_time, end_time)
    # 将时间戳列表转为字符串
    time_list = [str(i) for i in time_list]
    logger.info("时间戳列表生成完毕")

    #每条轨迹的点数取决于时间戳的个数
    point_num = len(time_list)
    
    #生成手机号
    phone_number_list = generate_phone_number(person_num)
    # 将手机号列表转为字符串
    phone_number_list = [str(i) for i in phone_number_list]
    logger.info("手机号列表生成完毕")

    #分批次生成所有人的轨迹
    # ATTENTION: 可调参数
    batchSize = 5000
    # batchSize = 10000
    # batchSize = 20000

    batchNum = person_num // batchSize
        
    batches = None
    if batchNum == 0:
        batches = [person_num]
    elif person_num % batchSize != 0:
        batches = [batchSize] * batchNum
        batches.append(person_num % batchSize)
    else:
        batches = [batchSize] * batchNum
    
    logger.info("分{}批次生成轨迹，批大小为{}".format(len(batches), batchSize))
    curr_track_index = 0
    for i in range(len(batches)):
        logger.info("正在生成第{}批次的轨迹".format(i + 1))
        #生成轨迹
        tracks = generateTracksWithMultiProcess(batches[i], fence_box, point_num, step_length, digits)
        logger.info("第{}批次的轨迹生成完毕".format(i + 1))

        '''
            将轨迹写入文件，串行输出版本
        '''
        # logger.info("正在将第{}批次的轨迹写入文件".format(i + 1))
        # with open(file_name, 'a') as f:
        #     for j in range(len(tracks)):
        #         track = tracks[j]
        #         for k in range(len(track)):
        #             f.write(time_list[k] + ',' + phone_number_list[j + curr_track_index] + ',' + str(track[k][0]) + ',' + str(track[k][1]) + '\n')
        #         if (j + 1) % 100 == 0:
        #             logger.info("第{}批次的第{}条轨迹写入完毕".format(i + 1, j + 1))       
        # logger.info("第{}批次的轨迹写入文件完毕".format(i + 1))
        # curr_track_index += batches[i]


        # 利用多进程将轨迹写入多个文件
        # ATTENTION: 可调参数
        file_num_per_batch = 24
        logger.info("正在分{}个文件将第{}批次的轨迹写入文件".format(file_num_per_batch, i + 1))
        write_tracks_to_files(tracks, curr_track_index, i + 1, file_num_per_batch, file_name, time_list, phone_number_list)
        logger.info("第{}批次的轨迹写入文件完毕".format(i + 1))
        curr_track_index += batches[i]

       

    logger.info("所有轨迹写入文件完毕")





    '''
        第一版，在内存中生成完整signals再一次性导出，会内存溢出
    '''
    # logging.info("正在生成" + str(person_num) + "条轨迹")
    # tracks = generateTracksWithMultiProcess(person_num, fence_box, point_num, step_length)
    # logging.info("轨迹生成完毕")
    
    
    # #生成信号
    # signals = []
    # logging.info("正在将" + str(len(time_list) * person_num) + "个signal放入signals列表")
    # for i in range(len(time_list)):
    #     for j in range(person_num):
    #         signals.append([time_list[i], phone_number_list[j], tracks[j][i][0], tracks[j][i][1]])
    #     if i % 10 == 0:
    #         logging.info("已放入时刻：" + str(time_list[i]) + "所有人的signal")
    # logging.info("signals列表生成完毕")

    # #写入文件
    # logging.info("正在写入文件")
    # with open(file_name, 'w') as f:
    #     for signal in signals:
    #         # 经纬度保留5位小数，米级精度
    #         f.write(str(signal[0]) + ',' + str(signal[1]) + ',' + str(round(signal[2], digits)) + ',' + str(round(signal[3], digits)) + '\n')
    # logging.info("文件写入完毕")



if __name__ == '__main__':
    '''
        生成10条轨迹
    '''
    # tracks = []
    # for i in range(1):
    #     tracks.append(generateTrackPoints(ZHENGZHOU_FENCE, int(14 * 24 * 3600 / 5), 10))
    # visualizeTrack(tracks, ZHENGZHOU_FENCE)


    '''
        生成一个人的轨迹，并且将之导入qgis可视化
    '''
    # trackPoints = generateTrackPoints(ZHENGZHOU_FENCE, int(3600 / 5), 5)
    # # 写入文件
    # with open('trackPoints.txt', 'w') as f:
    #     for point in trackPoints:
    #         f.write(str(point[0]) + ',' + str(point[1]) + '\n')


    '''
        测试并行生成700条轨迹，计时
    '''

    # print(multiprocessing.cpu_count())
    # start_time = time.time()
    # tracks = generateTracksWithMultiProcess(500000, ZHENGZHOU_FENCE, int(60 / 5), 5)
    # end_time = time.time()
    # print("并行生成1000条轨迹耗时：" + str(end_time - start_time))
    # print("\n")
    # print(len(tracks))
    # print("\n")

    # start_time = time.time()
    # tracks2 = generateTracks(500000, ZHENGZHOU_FENCE, int(60 / 5), 5)
    # end_time = time.time()
    # print("生成1000条轨迹耗时：" + str(end_time - start_time))
    # print("\n")
    # print(len(tracks2))




    
    # 760w人口对应ZHENGZHOU_FENCE
    # 20w人口横轴范围缩小8倍，纵轴范围缩小4倍，则面积缩小32倍
    SQUEEZE_ZHENGZHOU_FENCE = (113.4441000000000059, 34.5986440000000002, 113.49607750000001, 34.69004125)

    logger.info('程序开始')
    # generate_signals_to_file(200000, time_to_unix('2017-01-01 00:00:00'), time_to_unix('2017-01-02 00:00:00'), STEP_LENGTH, SQUEEZE_ZHENGZHOU_FENCE, PATH + 'signals')
    generate_signals_to_file(200000, time_to_unix('2017-01-01 00:00:00'), time_to_unix('2017-01-02 00:00:00'), STEP_LENGTH, SQUEEZE_ZHENGZHOU_FENCE, PATH + 'signals_')
    logger.info('程序结束')


