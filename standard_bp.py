import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

learning_rate = 0.05
epochs = 100


# 激活函数
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


# 激活函数的导数
def sigmoid_grad(x):
    return 1. / (1. + np.exp(-x))


# 训练函数
def train(x, y, outputs_dim=3, eta=learning_rate, max_iter=epochs):  # 学习率为eta[0.03, 0.05]
    # outputs_dim  输出层神经元个数
    hiden_dim = 50  # 隐层神经元个数

    # 定义权重
    w1 = np.zeros((x.shape[1], hiden_dim))  # （13，50）<--- 矩阵维数
    b1 = np.zeros((1, hiden_dim))  # （1，50）
    w2 = np.zeros((hiden_dim, 1))  # （50，1）
    b2 = np.zeros((outputs_dim, 1))  # 1 X 1

    losslist = []  # 损失列表
    acc_list = []  # 精度列表
    mse_list = []  # 均方误差列表

    for ite in range(max_iter):  # iter即为训练轮数epoch
        loss_per_ite = []
        for m in range(x.shape[0]):  # 遍历样本
            xi, yi = x[m, :], y[m, :]
            xi, yi = xi.reshape(1, xi.shape[0]), yi.reshape(1, yi.shape[0])
            # 前向传播
            u1 = np.dot(xi, w1) + b1
            out1 = sigmoid(u1)  # 隐含层的输出  1 X 50
            u2 = np.dot(out1, w2) + b2  # (1,50) X (50，1) =（1,1）
            out2 = sigmoid(u2)  # 输出(激活)层的输出,（1,1）
            loss = np.square(yi - out2) / 2
            loss_per_ite.append(loss)
            # print("iter:",ite," loss:",loss)
            # 反向传播
            # 标准BP
            d_out2 = -(yi - out2)  # （1,1）
            d_u2 = d_out2 * sigmoid_grad(out2)
            d_w2 = np.dot(np.transpose(out1), d_u2)  # delta(whj)，(50，1), np.transpose()--矩阵转置
            d_b2 = d_u2  # delta(thetaj),(1, 1)
            d_out1 = d_u2 * w2  # (1,1) 点乘 (50，1) ---> (50,1)
            d_u1 = np.transpose(d_out1) * sigmoid_grad(out1)  # -eh    d_out1: (50, 1) , out1: 1 X 50 ，改成矩阵点乘
            #  shapes (13,1) and (1,50) --->  (13 , 50)
            d_w1 = np.dot(np.transpose(xi), d_u1)  # delta(vih)
            d_b1 = d_u1  # delta(rh)  (1 , 50)
            # 更新权重
            w1 = w1 - eta * d_w1  # eta学习率
            w2 = w2 - eta * d_w2
            b1 = b1 - eta * d_b1
            b2 = b2 - eta * d_b2
            # test预测结果
            test_label_list = []
            for m in range(x_test.shape[0]):
                xi, yi = x_test[m, :], label_test[m, :]
                xi, yi = xi.reshape(1, xi.shape[0]), yi.reshape(1, yi.shape[0])
                # 前向传播
                u1 = np.dot(xi, w1) + b1
                out1 = sigmoid(u1)  # 隐含层的输出  1 X 50
                u2 = np.dot(out1, w2) + b2
                out2 = sigmoid(u2)  # 激活层
                if out2 >= 0.5:
                    test_label_list.append(1)
                else:
                    test_label_list.append(0)
            # /test预测结果
        # 扩充acc列表
        re = 0  # 记录测试正确的样本数
        mse = 0  # 均方误差
        mse_num = 0
        # 计算测试精度
        for i in range(len(y_test)):
            if test_label_list[i] == y_test[i]:
                re = re + 1
            else:
                pass
        for i in range(len(y_test)):
            mse_num += 1
            mse += np.square(test_label_list[i] - y_test[i]) / mse_num
        # 计算、输出测试精度
        acc = re / len(y_test)
        acc_list.append(acc)
        losslist.append(np.mean(loss_per_ite))
        mse_list.append(mse)
    # acc可视化
    plt.figure()
    plt.plot([i + 1 for i in range(epochs)], acc_list)  # 绘制loss随iteration的变化
    plt.legend(['standard BP'])
    plt.xlabel('iteration')
    plt.ylabel('Accuracy')
    plt.title("Accuracy Curve with epochs {}".format(epochs))
    plt.show()
    # mse可视化
    plt.figure()
    plt.plot([i + 1 for i in range(epochs)], mse_list)  # 绘制loss随iteration的变化
    plt.legend(['standard BP'])
    plt.xlabel('iteration')
    plt.ylabel('MSE')
    plt.title("MSE Curve with epochs {}".format(epochs))
    plt.show()
    # Loss可视化，损失函数曲线
    plt.figure()
    plt.plot([i + 1 for i in range(max_iter)], losslist)  # 绘制loss随iteration的变化
    plt.legend(['standard BP'])
    plt.xlabel('iteration')
    plt.ylabel('Loss')
    plt.title("Loss Curve with learning rate {}".format(learning_rate))
    plt.show()
    return w1, w2, b1, b2


# 读取数据
wine = np.genfromtxt("wine_data-2.csv", delimiter=",", skip_header=1)  # 二分类任务
X = wine[:, 0:13]
y = wine[:, 13]

sc = StandardScaler()
X_st = sc.fit_transform(X)  # 对样本的各属性值进行标准化

x_train, x_test, y_train, y_test = train_test_split(X_st, y)  # 默认取出97个样本作为测试集，33个作为测试集
# print(x_train.shape)

rate = 0.003
# 以13个特征值作为输入，1个神经元作为输出（输出 >= 0.5为1类，输出 < 0.5为0类），中间隐藏层50个神经元
v = np.random.random((13, 100)) * 2 - 1
w = np.random.random((50, 3)) * 2 - 1

label_train = LabelBinarizer().fit_transform(y_train)
# print(label_train)
label_test = LabelBinarizer().fit_transform(y_test)
# print(label_test)

# 训练神经网络
w1, w2, b1, b2 = train(x_train, label_train, 1)  # 成功训练
test_label_list = []
for m in range(x_test.shape[0]):
    xi, yi = x_test[m, :], label_test[m, :]
    xi, yi = xi.reshape(1, xi.shape[0]), yi.reshape(1, yi.shape[0])
    # 前向传播
    u1 = np.dot(xi, w1) + b1
    out1 = sigmoid(u1)  # 隐含层的输出  1 X 50
    u2 = np.dot(out1, w2) + b2
    out2 = sigmoid(u2)  # 激活层
    if out2 >= 0.5:
        test_label_list.append(1)
    else:
        test_label_list.append(0)

re = 0  # 记录测试正确的样本数
# 计算测试精度
for i in range(len(y_test)):
    if test_label_list[i] == y_test[i]:
        re = re + 1
    else:
        pass
# 计算、输出测试精度
acc = re / len(y_test)
print("测试精度acc = ", acc)
# 绘制混淆矩阵
my_confusion_matrix = confusion_matrix(y_test, test_label_list, labels=[0, 1])
plt.matshow(my_confusion_matrix, cmap=plt.cm.Reds)
for i in range(len(my_confusion_matrix)):
    for j in range(len(my_confusion_matrix)):
        plt.annotate(my_confusion_matrix[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix")
plt.show()
