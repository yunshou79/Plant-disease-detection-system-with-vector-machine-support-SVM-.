from PIL import Image
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error


def TransFile(filename):



    im = Image.open(filename)
    im = im.convert("L")  # 设置为灰度图片
    datalist = list(im.getdata())
    return datalist  # 返回单个图片特征表


def getData(filepath):
    dataList = []
    dataLabel = []
    count = 0  # 训练个数
    for i in range(0, 20):  # 取前 10 种
        filePath = os.path.join(filepath, str(i))
        plantPath = [os.path.join(filePath, str(plant)) for plant in os.listdir(filePath)]
        for plant in plantPath:
            count += 1
            data = TransFile(plant)
            dataList.append(data)
            dataLabel.append(i)
    return dataList, dataLabel  # 返回图片特征表和标签类别


if __name__ == '__main__':
    # 获取训练数据
    trainFile = os.path.join(r'E:\机器学习课设\植物疾病数据集\PlantVillageForSVM','train')
    dataList, dataLabel = getData(trainFile)

    # 获取测试数据
    testFile = os.path.join(r'E:\机器学习课设\植物疾病数据集\PlantVillageForSVM', 'test')
    testList, testLabel = getData(testFile)

    # 定义不同的内核和参数组合
    svm_params = [
        ('linear', {}),
        ('poly', {}),
        ('sigmoid', {}),
        ('rbf', {}),
        ('rbf', {'C': 2}),
        ('rbf', {'C': 2.5}),
        ('rbf', {'C': 3}),
        ('rbf', {'C': 3.5}),
        ('rbf', {'C': 3.6}),
        ('rbf', {'C': 3.7}),
        ('rbf', {'C': 5}),
        ('rbf', {'C': 4}),
        ('rbf', {'C': 10}),
        ('rbf', {'C': 0.95}),
        ('rbf', {'C': 0.9}),
    ]

    for kernel, params in svm_params:
        # 创建 SVM 模型
        clf = SVC(kernel=kernel, **params)

        # 拟合模型
        clf.fit(dataList, dataLabel)

        # 得到验证集上的预测值
        preLabel = clf.predict(testList)

        # 评估模型
        mse = mean_squared_error(preLabel, testLabel)
        cor_nums = sum([1 for i, j in zip(testLabel, preLabel) if i == j])  # 分类正确的个数
        accuracy = cor_nums / len(testLabel) * 100
        print(f"\n内核：{kernel}，参数：{params}")
        print(f"均方误差：{mse}")
        print(f"预测准确率：{accuracy:.2f}%")
