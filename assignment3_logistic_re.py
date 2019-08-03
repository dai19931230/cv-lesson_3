## Logistic Regression

# do Logistic Regression in Python mode

###############################
import numpy as np
import random
import matplotlib.pyplot as ply


def sigmoid_function(x):

    result = 1 / (1 + np.exp(-x))
    return result


def inference(w, b, x):  # inference, test, predict, same thing. Run model after training


    pred_y = sigmoid_function((np.dot(w.T, x) + b).reshape(-1, x.shape[1]))
    # print(pred_y)
    #     print(pred_y.shape)
    return pred_y


def eval_loss(w, b, x_train, gt_y_train):
    # avg_loss = 0.0
    # print('len_y:{0}'.format(gt_y_train.shape[1]))
    h = inference(w, b, x_train)
    # print(h)
    # print(gt_y_train)
    avg_loss = -np.sum(gt_y_train * np.log(h) + (1 - gt_y_train) * np.log(1 - h)) / gt_y_train.shape[1]

    return avg_loss


def gradient(pred_y, gt_y, x):
    # print(gt_y.shape)
    diff = pred_y - gt_y #(1, 50)
    # print(x.shape)

    dw = np.sum(diff * x, 1).reshape(x.shape[0], -1)
    # print(dw)
    db = np.sum(diff)
    # print(db)
    return dw, db


def cal_step_gradient(batch_x_train, batch_gt_y_train, w, b, lr):
    # avg_dw, avg_db = 0, 0
    batch_size = batch_x_train.shape[1]
    # print(bat)
    # print(batch_x_train.shape)
    pred_y = inference(w, b, batch_x_train)

    dw, db = gradient(pred_y, batch_gt_y_train, batch_x_train)

    avg_dw = dw / batch_size
    avg_db = db / batch_size
    # print(avg_dw)
    w -= lr * avg_dw
    b -= lr * avg_db

    return w, b
    # for i in range(batch_size):
    #     pred_y = inference(w, b, batch_x_list[i])	# get label data
    #     dw, db = gradient(pred_y, batch_gt_y_list[i], batch_x_list[i])
    #     avg_dw += dw
    #     avg_db += db
    # avg_dw /= batch_size
    # avg_db /= batch_size
    # w -= lr * avg_dw
    # b -= lr * avg_db
    # return w, b


def train(x_train, gt_y_train, batch_size, lr, max_iter):
    w = np.zeros((x_train.shape[0], 1))
    b = np.zeros((1, 1))
    # print(w.shape)
    # print(w)
    # num_samples = x_train.shape[1]

    for i in range(max_iter):
        batch_idxs = np.random.choice(x_train.shape[1], batch_size)
        # print(batch_idxs)
        batch_x = x_train[:, batch_idxs].reshape(-1, batch_size)
        batch_y = gt_y_train[:, batch_idxs].reshape(-1, batch_size)
        # print(batch_y)

        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss is {0}'.format(eval_loss(w, b, x_train, gt_y_train)))
    return w, b


def gen_sample_data():
    num_samples = 50
    x_list_x = []
    x_list_y = []
    y_list_x = []
    y_list_y = []
    for i in range(num_samples):
        point1_x = random.randint(0, 10) + random.random()
        point1_y = random.randint(0, 10) + random.random()  # for noise random.random[0, 1)
        point2_x = random.randint(12, 25) + random.random()
        point2_y = random.randint(12, 25) + random.random()

        x_list_x.append(point1_x)
        x_list_y.append(point1_y)
        y_list_x.append(point2_x)
        y_list_y.append(point2_y)

    class_point1 = [0 for i in range(num_samples)]
    class_point2 = [1 for i in range(num_samples)]
    x_list = x_list_x+y_list_x
    y_list = x_list_y+y_list_y
    class_point = class_point1+class_point2
    return x_list, y_list, class_point


def run():
    x_list, y_list, class_point = gen_sample_data()
    print(len(x_list))
    # print(class_point[0:len(x1_train)])
    fig, (ax, ax2) = ply.subplots(1, 2)
    p1 = ax.scatter(x_list[0:int(len(x_list)/2)], y_list[0:int(len(y_list)/2)],c='r', marker='*')
    p2 = ax.scatter(x_list[int(len(x_list) / 2):len(x_list)], y_list[int(len(y_list) / 2):len(x_list)],c='b', marker='o')
    ax.set_xlabel('X')  # 设置X轴标签
    ax.set_ylabel('Y')  # 设置Y轴标签
    ax.legend([p1, p2], ['class 0', 'class 1'])
    ax.set_title("origin")
    # ply.show()

    x_list_array = np.array(x_list).reshape(-1, len(x_list))
    y_list_array = np.array(y_list).reshape(-1, len(y_list))
    # print(x_list_array.shape)
    x_train = np.r_[x_list_array, y_list_array] # (2,100)
    # print(x_train.shape)
    y_train = np.array(class_point).reshape(-1, len(class_point))
    # print(y_train.shape) #(1,100)

    lr = 0.001
    max_iter = 10000
    #     print(x_train.shape)
    w, b = train(x_train, y_train, 50, lr, max_iter)
    pred_y = inference(w, b, x_train)


    pred_y[pred_y <= 0.5] = 0
    pred_y[pred_y > 0.5] = 1
    print(pred_y.shape)
    # y_prid = []
    # for i in range(len(x_list)):
    #     y = inference(w, b, x_list[i])
    #     y_prid.append(y)

    # y_prid = w * x_list + b
    # fig, ax2 = ply.subplots(1, 2)
    # print(x_train)
    # print(pred_y)
    # print(y_train)
    print(x_train.shape)

    p3 = ax2.scatter(x_list_array[pred_y <= 0.5], y_list_array[pred_y <= 0.5], c='r', marker='*')
    p4 = ax2.scatter(x_list_array[pred_y > 0.5], y_list_array[pred_y > 0.5], c='b', marker='o')
    ax2.set_xlabel('X')  # 设置X轴标签
    ax2.set_ylabel('Y')  # 设置Y轴标签
    ax2.legend([p3, p4], ['class 0', 'class 1'])
    ax2.set_title("pridection")
    ply.show()


if __name__ == '__main__':
    run()
