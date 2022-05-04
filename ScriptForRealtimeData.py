import math
import multiprocessing
import random
import time
import logging




# 保存路径
# PATH = 'E:/COVID19-Huawei/20w1day/'
PATH = '/mnt/storage/COVID19-Huawei/20w1day/'

# 总人口数量
NUM_PERSON = 200000
# 起止时间
START_TIME = '2017-01-01 00:00:00'
END_TIME = '2017-01-02 00:00:00'
# 每5s运动范围m
STEP_LENGTH = 5

ZHENGZHOU_FENCE = (113.4441000000000059, 34.5986440000000002, 113.8599200000000025, 34.9642330000000001)
# 760w人口对应ZHENGZHOU_FENCE
# 20w人口横轴范围缩小8倍，纵轴范围缩小4倍，则面积缩小32倍
SQUEEZE_ZHENGZHOU_FENCE = (113.4441000000000059, 34.5986440000000002, 113.49607750000001, 34.69004125)
# PROCESS_NUM = multiprocessing.cpu_count()
PROCESS_NUM = 6


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



#时间转为Unix时间戳, 单位秒
def time_to_unix(time_str):
    timeArray = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    unix_time = int(time.mktime(timeArray))
    return unix_time

#生成num个不重复的11位手机号
def generate_phone_number(num):
    phone_number_list = random.sample(range(1000000000, 9999999999), num)
    phone_number_list = ['1' + str(i) for i in phone_number_list]
    return phone_number_list

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


# 生成num个在fence内的随机点位
def initialRandomPoints(fenceBox, num, digit=6):
    f_xMin, f_yMin, f_xMax, f_yMax = fenceBox
    return [[round(random.uniform(f_xMin, f_xMax), digit), round(random.uniform(f_yMin, f_yMax), digit)] for i in range(num)]

# 并行生成num个在fence内的随机点位
def initialRandomPointsParallel(fenceBox, num, digit=6, processNum=PROCESS_NUM):
    pool = multiprocessing.Pool(processes=processNum)
    ptNumPerProcess = num // processNum
    if ptNumPerProcess == 0:
        return initialRandomPoints(fenceBox, num, digit)
    # 创建进程池
    processes = []
    for i in range(processNum):
        if i == processNum - 1:
            processes.append(pool.apply_async(initialRandomPoints, (fenceBox, num % ptNumPerProcess + ptNumPerProcess, digit)))
        else:
            processes.append(pool.apply_async(initialRandomPoints, (fenceBox, ptNumPerProcess, digit)))
    pool.close()
    pool.join()
    tracks = []
    for p in processes:
        tracks.extend(p.get())
    return tracks

# 生成以经纬度x(lon), y(lat)为中心,在纵向最大跨度为deltaY meter, 横向最大跨度为deltaX meter的矩形区域内的随机点位
# 生成的随机点位保证在fence内
# generateNextRandomPoint(113.5, 34.7, 10, 10, (113.0, 34.0, 114.0, 35.0))
def generateNextRandomPoints(pts, deltaX, deltaY, fenceBox, digit=6):
    res = []
    f_xMin, f_yMin, f_xMax, f_yMax = fenceBox
    for i in range(len(pts)):
        x = pts[i][0]
        y = pts[i][1]

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
        res.append((round(x_next, digit), round(y_next, digit)))
    return res

def generateLaterAllRandomPoints(pts, startIndex, deltaX, deltaY, fenceBox, startTimeUNIX, endTimeUNIX, to_write_queue, digit=6):
    t = startTimeUNIX
    res = pts
    while t < endTimeUNIX:
        res = generateNextRandomPoints(res, deltaX, deltaY, fenceBox, digit)
        to_write_queue.put((t, res, startIndex))
        t += 5



def generateLaterAllRandomPointsParallel(pts, deltaX, deltaY, fenceBox, startTimeUNIX, endTimeUNIX, to_write_queue, digit=6, processNum=PROCESS_NUM):
    # pool = multiprocessing.Pool(processes=processNum)
    # ptNumPerProcess = len(pts) // processNum
    # if ptNumPerProcess == 0:
    #     generateLaterAllRandomPoints(pts, deltaX, deltaY, fenceBox, startTimeUNIX, endTimeUNIX, to_write_queue, digit)
    # # 创建进程池
    # processes = []
    # for i in range(processNum):
    #     if i == processNum - 1:
    #         processes.append(pool.apply_async(generateLaterAllRandomPoints, (pts[i * ptNumPerProcess:], deltaX, deltaY, fenceBox, startTimeUNIX, endTimeUNIX, to_write_queue,  digit)))
    #     else:
    #         processes.append(pool.apply_async(generateLaterAllRandomPoints, (pts[i * ptNumPerProcess:(i + 1) * ptNumPerProcess], deltaX, deltaY, fenceBox, startTimeUNIX, endTimeUNIX, to_write_queue, digit)))
    # pool.close()
    # pool.join()
    # for p in processes:
    #     p.get()

    ptNumPerProcess = len(pts) // processNum
    if ptNumPerProcess == 0:
        generateLaterAllRandomPoints(pts, deltaX, deltaY, fenceBox, startTimeUNIX, endTimeUNIX, to_write_queue, digit)
    # 创建进程池
    processes = []
    for i in range(processNum):
        if i == processNum - 1:
            processes.append(multiprocessing.Process(target=generateLaterAllRandomPoints, args=(pts[i * ptNumPerProcess:], i * ptNumPerProcess, deltaX, deltaY, fenceBox, startTimeUNIX, endTimeUNIX, to_write_queue, digit)))
        else:
            processes.append(multiprocessing.Process(target=generateLaterAllRandomPoints, args=(pts[i * ptNumPerProcess:(i + 1) * ptNumPerProcess], i * ptNumPerProcess, deltaX, deltaY, fenceBox, startTimeUNIX, endTimeUNIX, to_write_queue, digit)))
    for p in processes:
        p.start()
        logger.info("开启一个点位生成进程")
    for p in processes:
        p.join()
        logger.info("结束一个点位生成进程")

def writePointsToFile(pts, t, phoneList, fileName):
    with open(PATH + fileName + '.csv', 'a') as f:
        for i in range(len(pts)):
            f.write(str(t) + ',' + phoneList[i] + ',' + str(pts[i][0]) + ',' + str(pts[i][1]) + '\n')

def multiprocessToFileWorker(to_write_lst, phoneList, file_name_prefix):
    for i in range(len(to_write_lst)):
        writePointsToFile(to_write_lst[i][1], to_write_lst[i][0], phoneList[to_write_lst[i][2]:], file_name_prefix + '_' + str(to_write_lst[i][0]))

def writeFilesParallel(to_write_lst, phoneList, file_name_prefix, processNum=PROCESS_NUM):
    WRITE_FILE_PROCESS_NUM = processNum
    file_num_per_process = len(to_write_lst) // WRITE_FILE_PROCESS_NUM
    if file_num_per_process == 0:
        for i in range(len(to_write_lst)):
            writePointsToFile(to_write_lst[i][1], to_write_lst[i][0], phoneList[to_write_lst[i][2]:], file_name_prefix + '_' + str(to_write_lst[i][0]))
    else:
        processes = []
        for i in range(WRITE_FILE_PROCESS_NUM):
            if i == WRITE_FILE_PROCESS_NUM - 1:
                processes.append(multiprocessing.Process(target=multiprocessToFileWorker, args=(to_write_lst[i * file_num_per_process:], phoneList, file_name_prefix)))
            else:
                processes.append(multiprocessing.Process(target=multiprocessToFileWorker, args=(to_write_lst[i * file_num_per_process:(i + 1) * file_num_per_process], phoneList, file_name_prefix)))
        for p in processes:
            p.start()
        for p in processes:
            p.join()            

def consumerfunc(to_write_queue, phoneList, file_name_prefix):
    to_write = []
    # ATTENTION: 可调参数
    BATCH_SIZE = 2000
    WRITE_FILE_PROCESS_NUM = 24
    while True:
        item = to_write_queue.get()
        if item is None:
            writeFilesParallel(to_write, phoneList, file_name_prefix, WRITE_FILE_PROCESS_NUM)
            logger.info('文件导出进程退出')
            break
        
        to_write.append(item)
        
        if len(to_write) == BATCH_SIZE:
            logger.info('开始批量写入{}文件！'.format(BATCH_SIZE))
            writeFilesParallel(to_write, phoneList, file_name_prefix, WRITE_FILE_PROCESS_NUM)
            to_write = []
            logger.info('批量写入完成！')







def readInfoFromLastT(fileName):
    phoneList = []
    pts = []
    with open(PATH + fileName + '.csv', 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            raise Exception('文件为空')
        else:
            for line in lines:
                line = line.strip()
                line = line.split(',')
                phoneList.append(line[1])
                pts.append([float(line[2]), float(line[3])])
    return phoneList, pts


def generate_signals_to_file(person_num, start_time, end_time, step_length, fence_box, file_name_prefix, process_num=PROCESS_NUM, phoneList=[], pts=[]):
    # 初始化随机点位
    logger.info('初始化随机点位')
    if len(pts) == 0:
        pts = initialRandomPointsParallel(fence_box, person_num, processNum=process_num)
    logger.info('初始化{}随机点位完成'.format(person_num))
    #生成手机号
    if len(phoneList) == 0:
        phoneList = generate_phone_number(person_num)
    logger.info("手机号列表生成完毕")

    # 创建写文件消费者进程
    to_write_queue = multiprocessing.Queue()
    consumer_process = multiprocessing.Process(target=consumerfunc, args=(to_write_queue, phoneList, file_name_prefix))
    consumer_process.start()


    # 初始化时间
    start_time_unix = time_to_unix(start_time)
    logger.info('设置开始时间戳{}完成'.format(start_time_unix))
    to_write_queue.put((start_time_unix, pts, 0))
    
    end_time_unix = None
    if end_time == -1:
        end_time_unix = time_to_unix('2099-01-01 00:00:00')
        logger.info('设置结束时间为Infinity')
    else: 
        end_time_unix = time_to_unix(end_time)
        logger.info('设置结束时间为{}'.format(end_time_unix))
    
    generateLaterAllRandomPointsParallel(pts, step_length, step_length, fence_box, start_time_unix + 5, end_time_unix, to_write_queue, processNum=process_num)

    
    to_write_queue.put(None)
    logger.info('所有点位写入完毕')

if __name__ == '__main__':
    # generate_signals_to_file(200000, '2019-01-01 00:00:00', '2019-01-02 00:00:00', 5, SQUEEZE_ZHENGZHOU_FENCE, 'signals')
    logger.info(time.time())
    generate_signals_to_file(200000, '2017-01-01 00:00:00', '2017-01-02 00:00:00', 5, SQUEEZE_ZHENGZHOU_FENCE, 'signals')
    logger.info(time.time())
