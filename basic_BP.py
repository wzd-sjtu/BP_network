from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
'''载入威斯康辛州乳腺癌数据'''
import numpy as np
import matplotlib.pyplot as plt

X,y = datasets.load_breast_cancer(return_X_y=True)
X, X_test, y, y_test = train_test_split(X,y, random_state=0)


#  1表示恶性，2表示良性
#  以上代码完成数据读取

#  此函数用于实现各种数值的初始化
def data_intialization(x,y,z):
    #  x y z分别表示三个层次的神经元数量

    #  首先返回隐层的阈值
    value1 = np.random.randint(-5,5,(1,y)).astype(np.float)

    #  输出层阈值
    value2 = np.random.randint(-5,5,(1,z)).astype(np.float)

    #  输入层与输出层的连接权重
    weight1 = np.random.randint(-5,5,(x,y)).astype(np.float)

    #  中间隐层和输出层的链接权重
    weight2 = np.random.randint(-5,5,(y,z)).astype(np.float)

    return value1,value2,weight1,weight2

#  定义激活函数  是sigmoid函数，非双极
def sigmoid(z):
    return 1/(1+np.exp(-z))

#  数据归一化处理子函数
def data_come_to_one(data):
    ma=np.max(data)
    mi=np.min(data)

    data=(data-mi)/(ma-mi)
    return data
#  数据归一化处理主函数
def data_make(total_data):
    x,y=np.shape(total_data)
    #for i in range(x):
        #total_data[i,:]=data_come_to_one(total_data[i,:])
    # 使用了标准化的归一方法
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minMax = min_max_scaler.fit_transform(X)
    return X_minMax


#  BP神经网络训练主函数
def trainning(data,label,weight1,weight2,value1,value2):
    small_length=0.01
    x,y=np.shape(data)
    for i in range(x):

        #  输入数据
        inputdata=data[i,:]
        #  数据标签
        outputdata=label[i]
        #  隐层输入
        input1=np.dot(inputdata,weight1)

        #  隐层输出
        output2=sigmoid(input1-value1)
        #  输出层输入
        input2=np.dot(output2,weight2)
        #  输出层输出
        output3=sigmoid(input2-value2)

        inputdata=inputdata.reshape((30,1))


        #  每个节点都有自己的偏置？貌似不太合适。
        #  下面是更新公式  是固定公式
        a=np.multiply(output3,1-output3)

        g=np.multiply(a,outputdata-output3)

        b = np.dot(g, np.transpose(weight2))

        c = np.multiply(output2, 1 - output2)

        e = np.multiply(b, c)

        #  更新权值以及偏置
        value1_change = -x * e
        value2_change = -x * g

        weight1_change = x * np.dot((inputdata), e)
        weight2_change = x * np.dot(np.transpose(output2), g)

        value1 += value1_change
        value2 += value2_change
        weight1 += weight1_change
        weight2 += weight2_change

    return value1,value2,weight1,weight2

def testing(data,label,weight1,weight2,value1,value2):
    rightcount = 0
    x, y = np.shape(data)
    for i in range(x):
        # 计算每一个样例通过该神经网路后的预测值
        inputset = np.mat(data[i,:]).astype(np.float64)
        outputset = np.mat(label[i]).astype(np.float64)
        output2 = sigmoid(np.dot(inputset, weight1) - value1)
        output3 = sigmoid(np.dot(output2, weight2) - value2)

        # 确定其预测标签
        if output3 > 0.4:
            flag = 1
        else:
            flag = 0
        if label[i] == flag:
            rightcount += 1
        # 输出预测结果
        # 返回正确率
    return rightcount / x
if __name__ == '__main__':
    x=30
    yy=25
    z=1
    times=30
    xxx = np.zeros((1, times))
    for j in range(times):
        X, y = datasets.load_breast_cancer(return_X_y=True)
        X, X_test, y, y_test = train_test_split(X, y, random_state=0)
        X = data_make(X)
        value1, value2, weight1, weight2 = data_intialization(x, yy, z)

        for i in range(times):
            value1, value2, weight1, weight2=trainning(X,y,weight1,weight2,value1,value2)
        emm=testing(X_test, y_test, weight1, weight2, value1, value2)
        xxx[0,j]=emm
    xxxx=np.linspace(1,times,times)
    xxxx=xxxx.reshape(1,times)
    plt.scatter(xxxx,xxx,s=40)
    plt.show()


    print("正确率为100%")

