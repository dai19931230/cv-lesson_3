## Linear Regression

# Reorganize Linear Regression in Python mode

###############################
import numpy as np
import random
import matplotlib.pyplot as ply


def inference(w, b, x):  # inference, test, predict, same thing. Run model after training
    #     print(w.shape)

    #     print(x.shape)

    pred_y = (np.dot(w.T, x) + b).reshape(-1, x.shape[1])
    # print(pred_y)
    #     print(pred_y.shape)
    return pred_y


def eval_loss(w, b, x_train, gt_y_train):
    # avg_loss = 0.0
    # print('len_y:{0}'.format(gt_y_train.shape[1]))
    avg_loss = 0.5 * np.sum((np.dot(w.T, x_train) + b - gt_y_train) ** 2) / gt_y_train.shape[1]

    return avg_loss


def gradient(pred_y, gt_y, x):
    # print(gt_y.shape)
    diff = pred_y - gt_y #(1, 50)
    # print(x.shape)

    dw = np.sum(diff * x, 1)
    # print(dw)
    db = np.sum(diff)

    return dw, db


def cal_step_gradient(batch_x_train, batch_gt_y_train, w, b, lr):
    # avg_dw, avg_db = 0, 0
    batch_size = batch_x_train.shape[1]
    # print(bat)

    pred_y = inference(w, b, batch_x_train)

    dw, db = gradient(pred_y, batch_gt_y_train, batch_x_train)

    avg_dw = dw / batch_size
    avg_db = db / batch_size
    print(avg_dw)
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
    b = np.zeros((x_train.shape[0], 1))
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
    w = random.randint(0, 10) + random.random()  # for noise random.random[0, 1)
    b = random.randint(0, 5) + random.random()
    print('w_normal:{0}, b_normal:{1}'.format(w,b))
    num_samples = 100
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = random.randint(0, 100) * random.random()
        y = w * x + b + random.random() * random.randint(-1, 1)
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list, w, b


def run():
    x_list, y_list, w, b = gen_sample_data()

    x_train = (np.array(x_list)).reshape(-1, 100)
    y_train = (np.array(y_list)).reshape(-1, 100)
    fig, (ax, ax2) = ply.subplots(1, 2)
    ax.plot(x_train, y_train, 'ro')
    # ply.show()

    lr = 0.001
    max_iter = 10000
    #     print(x_train.shape)
    w, b = train(x_train, y_train, 50, lr, max_iter)
    pred_y = inference(w, b, x_train)

    # y_prid = []
    # for i in range(len(x_list)):
    #     y = inference(w, b, x_list[i])
    #     y_prid.append(y)

    # y_prid = w * x_list + b
    # fig, ax2 = ply.subplots(1, 2)
    # print(x_train)
    # print(pred_y)
    # print(y_train)
    ax2.plot(x_train, pred_y, 'y*')
    ply.show()


if __name__ == '__main__':
    run()
