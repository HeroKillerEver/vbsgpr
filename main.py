import tensorflow as tf
import data_loader
from model import VBSGPR
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser(description='A tensroflow implementation for VBSGPR', epilog='#' * 75)

parser.add_argument('--log_beta', default=0., type=float, help='initial value for log beta. Default: 0.')
parser.add_argument('--log_sf2', default=0., type=float, help='initial value for log sf2. Default: 0.')
parser.add_argument('--log_theta', default=0., type=float, help='initial value for log theta. Default: 0.')

parser.add_argument('--sgp', default=pic, type=str, help='sgpr: dtc, fitc, pic. Default: pic')
parser.add_argument('--epochs', default=50, type=int, help='num of epochs. Default: 50')
parser.add_argument('--clusters', default=1000, type=int, help='number of clusters. Default: 1000')
parser.add_argument('--num_inducing', type=int, default=100, help='number of inducing points. Default: 100')

config = parser.parse_args()


def main():
    tf.set_random_seed(2018)
    
    dtrain, dtest, z, y_std = data_loader.load('airplane.csv', n_clusters=config.clusters, n_induce=config.num_inducing, sgp=config.sgp)
    N, _ = dtrain.shape
    model = VBSGPR(N, config.log_beta, config.log_sf2, config.log_theta, z, whiten=True)

    clusters = [i for i in range(config.clusters)]

    lb = model.lower_bound()
    fmu, fcov = model.predict_f()
    gp_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vbsgpr')
    gp_opt = tf.train.AdamOptimizer(0.01, beta1=0.9, name='gp_opt') # Best: 40.459
    # gp_opt = tf.train.MomentumOptimizer(0.01, momentum=0.9, use_nesterov=False)
    gp_train_op = gp_opt.minimize(-lb, var_list=gp_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(config.epochs):
            random.shuffle(clusters)
            for i, cluster in enumerate(clusters):
                data_batch = dtrain[np.where(dtrain[:, -1] == cluster)]
                X, y = data_batch[:, :-2], data_batch[:, -2:-1]
                _, lb_ = sess.run([gp_train_op, lb], {model.x: X, model.y: y, model.batch: y.shape[0]})
                if i % 100 == 0: 
                    print ('Epoch: [{}], The {}-th Cluster: [{}], Lower Bound: [{}]'.format(
                            epoch, i, cluster, lb_))
            X_test, y_test = dtest[:, :-2], dtest[:, -2:-1]
            f_test, _ = sess.run([fmu, fcov], {model.x: X_test})
            rmse = np.sqrt(np.mean(y_std**2 * ((y_test - f_test))**2))
            print ('Epoch {} test RMSE: {}'.format(epoch, rmse))


if __name__ == '__main__':
    main()



