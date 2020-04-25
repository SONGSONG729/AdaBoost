import numpy as np
import matplotlib.pyplot as plt


def loadSimpData():
    datMat = np.array([[1., 2.1],
                       [2., 1.1],
                       [1.3, 1.],
                       [1., 1.],
                       [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

def stumpClassify(dataMatrix, dimen, threashVal, threshIneq):
    '''
    通过对阈值比较对数据进行分类
    :param dataMatrix:
    :param dimen:
    :param threashVal:
    :param threshIneq:
    :return:
    '''
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threashVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threashVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    '''
    遍历stumpClassify()函数所有的可能输入值，并找到数据集上最佳的单层决策树
    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    '''
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0  # 用于在特征的所有可能值上进行遍历
    bestStump = {}   # 存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf  # 在一开始就初始化成无穷大，之后用于寻找可能的最小错误率
    # 在数据集的所有特征上遍历，
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        # 通过计算最小值和最大值了解应该需要多大的步长

        # 遍历已知步长的值
        for j in range(-1, int(numSteps)+1):
            # 在大于和小于之间切换不等式
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)

                # stumpClassify: 返回分类预测结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                # 若predictedVals中的值不等于labelMat中的值真正类别标签，errArr的相应位置为1
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  # 计算加权错误率
                # print("split: dim %d, thresh %.2f, thresh inequal: %s, "
                #       "the weighted error is %.3f" % \
                #       (i, threshVal, inequal, weightedError))
                # 如果当前错误率比已有错误率小，则在bestStump中保留该层决策树
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLables, numIt=40):
    '''
    基于单层决策树的AdaBoost训练过程
    :param dataArr:数据集
    :param classLables:类别标签
    :param numIt:迭代次数，需用户指定
    :return:
    '''
    weakClassArr = []
    m = np.shape(dataArr)[0]  # 数据集中数据点的数目
    # D：列向量，概率分布向量，所有元素之和为1，包含了每个数据点的权重值，一开始被初始为1/m，
    D = np.mat(np.ones((m, 1)) / m)
    # aggClassEst: 记录每个数据点的类别估计累计值
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 运行numInt次或直到训练错误率为0为止
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLables, D)  # 建立单层决策树
        # print("D: ", D.T)
        alpha = float(0.5*np.log((1.0 - error) / max(error, 1e-16)))  # 计算alpha值，
        # max(error, 1e-16)用于确保在没有错误的情况下不会发生除零溢出
        bestStump['alpha'] = alpha  # 将alpha值加入到字典中
        weakClassArr.append(bestStump)  # 将字典添加到列表中
        # print("classEst: ", classEst.T)
        # exp_on: 为下一次迭代计算D
        # exp_on = - alpha * f(x) * h_t(x)  //f(x) 是标签, h_t(x)在D_t上的预测结果
        exp_on = np.multiply(-1*alpha*np.mat(classLables).T, classEst)
        D = np.multiply(D, np.exp(exp_on))
        D = D/D.sum()
        # 错误累加计算
        aggClassEst += alpha*classEst
        # print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLables).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        # print("total error: ", errorRate, "\n")
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst

def adaClassify(datToClass, classifierArr):
    '''
    AdaBoost分类函数
    :param datToClass:
    :param classifierArr:
    :return:
    '''
    dataMatrix = np.mat(datToClass)  # 转换成numpy矩阵
    m = np.shape(dataMatrix)[0]  # 得到dataMatrix中待分类样例的个数m
    aggClassEst = np.mat(np.zeros((m, 1)))  # 构建0列向量
    # 遍历classifierArr中所有弱分类器，并基于stumpClassify()对每个分类器得到一个类别的估值
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

def loadDataSet(fileName):
    '''
    自适应数据加载函数
    :param fileName:
    :return:
    '''
    numFeat = len(open(fileName).readline().strip().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    fr.close()
    return dataMat, labelMat

def plotROC(predStrengths, classLabels):
    '''
    ROC曲线的绘制及AUC计算函数
    :param predStrengths:
    :param classLabels:
    :return:
    '''
    cur = (1.0, 1.0)
    ySUm = 0.0
    numPosClas = sum(np.array(classLabels) == 1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySUm += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySUm * xStep)


def main():
    # datMat, classLabels = loadSimpData()
    # 程序清单7-1运行过程
    # D = np.mat(np.ones((5, 1)) / 5)
    # print(buildStump(datMat, classLabels, D))


    # 程序清单7-2运行过程
    # classifierArray = adaBoostTrainDS(datMat, classLabels, 9)
    # print(classifierArray)

    # 程序清单7-3运行过程
    # datArr, labelArr = loadSimpData()
    # classifierArr = adaBoostTrainDS(datArr, labelArr, 30)
    # print("classify result: ", adaClassify([0, 0], classifierArr))
    # print("classify result: ", adaClassify([[5, 5], [0, 0]], classifierArr))

    # 程序清单7-4运行过程
    # datArr, labelArr = loadDataSet('horseColicTraining2.txt')
    # classifierArray = adaBoostTrainDS(datArr, labelArr, 10)
    # testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    # prediction10 = adaClassify(testArr, classifierArray)
    # errArr = np.mat(np.ones((len(testArr), 1)))
    # err_num = errArr[prediction10 != np.mat(testLabelArr).T].sum()
    # print("error rate: ", err_num / len(testArr))

    # 程序清单7-5运行过程
    datArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrainDS(datArr, labelArr, 50)
    plotROC(aggClassEst.T, labelArr)


if __name__ == '__main__':
    main()
