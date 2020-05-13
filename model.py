import tensorflow as tf
import tensorflow.contrib.slim as slim 
import numpy as np

class Vbsgpr(object):
    """docstring for Vbac"""
    def __init__(self, sgpr, z): 
        """ 
        """
        super(Vbsgpr, self).__init__()
        self.sgpr = sgpr
        self.z = z.astype('float32')
        self.num_induce = self.z.shape[0]
        self.num_dim = self.z.shape[1]
        self.lr = lr
        self.epsilon = 1e-4

    def eKzx(self, z, x, lmean, lcov, sf):
        """
         Known as Omega: <K_{z, x}>_{q(theta)}.
        :param z: MxD inducing inputs
        :param x: NxD data inputs
        :param lmean: length-scale mean (D,)
        :param lcov: length-scale cov (D, )
        :param sf: signal variance (,) scalar
        :return: MxN
        """
        alpha = x * lmean  # N x D
        beta = x ** 2 * lcov + 1 # N x D
        vec = (tf.expand_dims(alpha, 1) - tf.expand_dims(z, 0))**2 \
        	  / tf.expand_dims(beta, 1) # N x 1 x D
        deno = tf.reduce_prod(beta**(-0.5), 1, keepdims=True) # N x 1
        res = sf * deno * tf.exp( -0.5 * tf.reduce_sum(vec, 2)) # N x M
        return tf.transpose(res) 

    def eKxx(self, x, lmean, lcov, sf):
        """
         Known as Gamma: <K_{x, x}>_{q(theta)}.
        :param x: NxD data inputs
        :param lmean: length-scale mean (D,)
        :param lcov: length-scale cov (D, )
        :param sf: signal variance (,) scalar
        :return: N x N
        """
        alpha = lmean * (tf.expand_dims(x, 1) - tf.expand_dims(x, 0))**2 # N x N x D
        beta = lcov * (tf.expand_dims(x, 1) - tf.expand_dims(x, 0))**2 + 1 # N x N x D
        deno = tf.reduce_prod(beta**(-0.5), 2, keepdims=False) # N x N 
        nume = tf.exp( -0.5 * tf.reduce_sum(alpha/beta, 2) ) # N x N 
        res = sf**2 * deno * nume
        return res


    def eKzxxz(self, z, x, lmean, lcov, sf, invC):
        """
        Known as Psi: <K_{z, x} * invC * K_{x, z}>_{q(theta)}.
        :param z: MxD inducing inputs
        :param x: NxD data inputs
        :param lmean: length-scale mean (D,)
        :param lcov: length-scale cov (D, )
        :param sf: signal variance (,) scalar
        :param invC: N x N
        :return: M x M
        """
        alpha = x * lmean # N x D
        xx = lcov * (tf.expand_dims(x**2, 1) + tf.expand_dims(x**2, 0)) + 1 # N x N x D
        zx = tf.expand_dims(x, 1) * tf.expand_dims(z, 0) # N x M x D
        r = tf.expand_dims(alpha, 1) - tf.expand_dims(z, 0) # N x M x D
        S = lcov * (tf.expand_dims(tf.expand_dims(zx, 0), 2) - tf.expand_dims(tf.expand_dims(zx, 1), 3))**2 # N x N x M x M x D
        T = (tf.expand_dims(tf.expand_dims(r, 0), 2))**2 + (tf.expand_dims(tf.expand_dims(r, 1), 1))**2 # N x N x M x M x D
        nume = tf.exp( -0.5 * tf.reduce_sum((S + T) / tf.expand_dims(tf.expand_dims(xx, 2), 2), 4)) # N x N x M x M 
        deno = tf.reduce_prod(tf.expand_dims(tf.expand_dims(xx, 2), 2)**(-0.5), 4) # N x N x 1 x 1
        res = sf**2 * tf.reduce_sum(tf.expand_dims(tf.expand_dims(invC, 2), 2) * nume * deno, axis=[0, 1]) # M x M 
        return res


    def K(self, sf2, x1, x2, ls):
        """
        :params: x1: N x D
        :params: x2: M x D

        return (N, M)
        """
        alpha = x1[:, None, :] / ls # N x 1 x D
        beta = x2[None, :, :] / ls # 1 x M x D
        return sf2 * tf.exp(tf.reduce_sum(-0.5 * (alpha - beta)**2, axis=2))


    def pic(self, ls):

        K_x_testz = self.K(self.sf2, self.x_test, self.z, ls) # (num_test, num_induce)
        K_x_testx = self.K(self.sf2, self.x_test, self.x, ls) # (num_test, num_train)

        K_zx = self.K(self.sf2, self.z, self.x, ls) # (num_induce, num_train)
        K_xx = self.K(self.sf2, self.x, self.x, ls) + self.C_xx # (num_train, num_train)

        K_x_test_zx = tf.concat([K_x_testz, K_x_testx], axis=1) # (num_test, num_induce + num_train)

        K_tmp1 = tf.concat([self.K_uu, K_zx], axis=1) # (num_induce, num_induce + num_train)
        K_tmp2 = tf.concat([tf.transpose(K_zx), K_xx], axis=1) # (num_train, num_induce + num_train)

        K = tf.concat([K_tmp1, K_tmp2], axis=0)

        Y = tf.concat([self.qU_mean, self.y], axis=0)

        mu_pic = K_x_test_zx @ tf.matrix_inverse(K) @ Y[:, None]

        sigma_pic = K_x_test_zx @ tf.matrix_inverse(K) @ tf.transpose(K_x_test_zx)

        return mu_pic[:, 0], sigma_pic # (num_test, ), (num_test, num_test)




    def build_model(self):

        self.x = tf.placeholder(tf.float32, [None, self.num_dim], name='x')
        self.x_test = tf.placeholder(tf.float32, [None, self.num_dim], name='x_test')
        self.y = tf.placeholder(tf.float32, [None], name='y')
        self.y_test = tf.placeholder(tf.float32, [None], name='y_test')
        # self.N = tf.placeholder(tf.int32, shape=(), name='N')
        # self.invC = tf.placeholder(tf.float32, [None, None], name="invC")


        self.lmean = tf.Variable(tf.random_uniform(shape=(self.num_dim, ), minval=0.1, maxval=1), name='lmean')
        self.lcov = tf.Variable(tf.random_uniform(shape=(self.num_dim, ), minval=0.1, maxval=1), name='lcov' )
        self.ls = tf.Variable(tf.random_uniform(shape=(self.num_dim, ), minval=0.1, maxval=1), name='ls')
        self.ls_const = tf.Variable(tf.ones_like(self.ls), trainable=False)
        self.sf = tf.Variable(tf.random_uniform(shape=(), minval=0.1, maxval=1), name='sf')
        self.sn2 = tf.Variable(tf.random_uniform(shape=(), minval=0.1, maxval=1), name='sn2')
        self.qU_mean = tf.Variable(tf.random_normal(shape=[self.num_induce, ]), name='qU_mean')
        # tmp = np.random.randn(self.num_induce, self.num_induce)
        # tmp = 0.5 * (tmp + tmp.T)
        # tmp = tmp + self.epsilon * np.eye(self.num_induce, dtype="float64")
        self.qU_cov = tf.Variable(tf.eye(self.num_induce), name='qU_cov')

        self.sf2 = self.sf**2
        self.K_epi_uu = self.K(self.sf2, self.z, self.z, self.ls)

        self.K_xx = self.K(self.sf2, self.x, self.x, self.ls)
        self.K_xz = self.K(self.sf2, self.x, self.z, self.ls)

        self.invK_epi_uu = tf.matrix_inverse(self.K_epi_uu + self.epsilon * tf.eye(self.num_induce))

        self.tmp = self.K_xx - self.K_xz @ self.invK_epi_uu @ tf.transpose(self.K_xz) 

        if self.sgpr == 'dtc':
            self.C_xx = self.sn2 * tf.diag(tf.ones_like(self.y))
        elif self.sgpr == 'fitc':
            self.C_xx = tf.diag(tf.diag_part(self.tmp)) + self.sn2 * tf.diag(tf.ones_like(self.y))
        elif self.sgpr == 'pitc' or self.sgpr == 'pic':
            self.C_xx = self.tmp + self.sn2 * tf.diag(tf.ones_like(self.y))   
        else:
            raise ValueError('dtc, fitc, pitc, pic')


        # self.invC = tf.matrix_inverse(self.C_xx + self.epsilon * tf.diag(tf.ones_like(self.y)))
        self.invC = self.C_xx + self.epsilon * tf.diag(tf.ones_like(self.y))






        self.K_uu = self.K(self.sf**2, self.z, self.z, self.ls_const)
        
        self.invKmm = tf.matrix_inverse(self.K_uu + self.epsilon * tf.eye(self.num_induce))


        self.EK_xx = self.eKxx(self.x, self.lmean, tf.abs(self.lcov), self.sf)
        self.EK_zx = self.eKzx(self.z, self.x, self.lmean, tf.abs(self.lcov), self.sf)



        self.EK_zxxz = self.eKzxxz(self.z, self.x, self.lmean, tf.abs(self.lcov), self.sf, self.invC)

 
 

        # grad_cov_summary = tf.summary.histogram("grad_cov", self.grad_cov)


        # MI_summary = tf.summary.scalar("MI", self.MI)

        self.A = self.invKmm @ self.EK_zxxz @ self.invKmm + self.invKmm
        self.B = self.invKmm @ self.qU_cov @ self.invKmm - self.invKmm

        self.EK_xx_test = self.eKxx(self.x_test, self.lmean, tf.abs(self.lcov), self.sf)
        self.EK_zx_test = self.eKzx(self.z, self.x_test, self.lmean, tf.abs(self.lcov), self.sf)
        self.EK_xzzx_test = self.eKzxxz(self.x_test, self.z, self.lmean, tf.abs(self.lcov), self.sf, self.B)


        self.lb = tf.reduce_sum(self.qU_mean[None, :] @ self.invKmm @ self.EK_zx @ self.invC @ self.y[:, None]) \
                  - tf.reduce_sum(0.5 * self.qU_mean[None, :] @ self.A @ self.qU_mean[:, None]) \
                  - 0.5 * tf.trace(self.qU_cov @ self.A) \
                  - 0.5 * tf.trace(self.invC @ self.EK_xx) \
                  + 0.5 * tf.trace(self.invKmm @ self.EK_zxxz) \
                  + 0.5 * tf.log(tf.matrix_determinant(self.qU_cov)) \
                  - 0.5 * tf.reduce_sum(self.lmean  * self.lmean) \
                  - 0.5 * tf.reduce_sum(self.lcov) \
                  + 0.5 * tf.reduce_sum(tf.log(1 + tf.abs(self.lcov)))

        dist = tf.distributions.Normal(loc=self.lmean, scale=self.lcov)
        ls_samples = dist.sample(sample_shape=(20,))

        if self.sgpr == 'pic':
            self.mu_samples, self.sigma_samples = tf.map_fn(lambda x: self.pic(x), elems=ls_samples, dtype=(tf.float32, tf.float32)) # (n_samples, num_test), (n_samples, num_test, num_test)
            self.mu = tf.redue_mean(self.mu_samples, axis=0)
            self.sigma = tf.redue_mean(self.sigma_samples, axis=0)

        else:
            self.mu_tmp = tf.transpose(self.EK_zx_test) @ self.invKmm @ self.qU_mean[:, None]
            self.mu = tf.squeeze(self.mu_tmp)
            self.sigma = self.EK_xx_test + self.EK_xzzx_test

        self.l2_loss = tf.reduce_sum((self.mu - self.y_test)**2) 





        # self.lb = tf.reduce_sum(self.N * self.beta * self.qU_mean @ self.invKmm @ (self.EKzx+self.Kfzx) @ self.invC @ self.invH @ tf.expand_dims(self.y, axis=1))  \
        #           - 0.5 * tf.reduce_sum(self.qU_mean @ self.invKmm @ (self.EKzxxz+self.Kfzxxz) @ self.invKmm @ tf.transpose(self.qU_mean)) \
        #           - 0.5 * tf.trace(  self.qU_cov @ self.invKmm @ (self.EKzxxz+self.Kfzxxz) @ self.invKmm ) \
        #           - 0.5 * tf.trace( self.invC @ (self.EKxx+self.Kfxx) ) + 0.5 * tf.trace(self.invKmm @ (self.EKzxxz+self.Kfzxxz)) \
        #           - 0.5 * tf.reduce_sum(self.qU_mean @ self.invKmm @ tf.transpose(self.qU_mean)) - 0.5 * tf.trace(self.qU_cov @ self.invKmm) \
        #           + 0.5 * tf.log( tf.matrix_determinant( self.qU_cov)) - 0.5 * tf.reduce_sum(self.lmean  * self.lmean) \
        #           - 0.5 * tf.reduce_sum(self.lcov) + 0.5 * tf.reduce_sum(tf.log(1 + tf.abs(self.lcov))) 

        lb_summary = tf.summary.scalar("lower_bound", self.lb)




        gp_vars = tf.trainable_variables()
        self.opt = tf.train.AdamOptimizer(self.lr, name='opt')
        self.train_op = self.opt.minimize(-self.lb, var_list=gp_vars)


        """
        summary op
        """

        # self.summary = tf.summary.merge([grad_cov_summary, MI_summary, lb_summary])

        for var in gp_vars:
            tf.summary.histogram(var.op.name, var)

        self.summary = tf.summary.merge_all()
class VBSGPR(object):
    """
    VBSGPR of DTC, FITC, PITC
    """
    def __init__(self, num_data, log_beta, log_sf2, log_theta, z, qmu=None, qlogdev=None, sgp='dtc', kernel='SE', whiten=False):
        self.kernel = kernel
        self.num_data = num_data
        log_theta = np.array(log_theta)
        self.m, dim = z.shape
        self.sgp = sgp
        with tf.variable_scope('vbsgpr', reuse=tf.AUTO_REUSE):
            self.log_beta = tf.get_variable('log_beta', shape=(), dtype=tf.float32, initializer=tf.constant_initializer(log_beta))
            self.log_sf2 = tf.get_variable('log_sf2', shape=(), dtype=tf.float32, initializer=tf.constant_initializer(log_sf2))
            self.log_theta = tf.get_variable('log_theta', shape=log_theta.shape, dtype=tf.float32, initializer=tf.constant_initializer(log_theta))
            self.z = tf.get_variable('z', shape=(self.m, dim), dtype=tf.float32, initializer=tf.constant_initializer(z))
            if qmu is None:
                self.qmu = tf.get_variable('qmu', shape=(self.m, 1), dtype=tf.float32, initializer=tf.zeros_initializer())
            else:
                self.qmu = tf.get_variable('qmu', shape=(self.m, 1), dtype=tf.float32, initializer=tf.constant_initializer(qmu))
            if qlogdev is None:
                self.qlogdev = tf.get_variable('qlogdev', shape=(self.m,), dtype=tf.float32, initializer=tf.zeros_initializer())
            else:
                self.qlogdev = tf.get_variable('qlogdev', shape=(self.m,), dtype=tf.float32, initializer=tf.constant_initializer(qlogdev))

        self.x = tf.placeholder(tf.float32, shape=(None, dim), name='x')
        self.y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
        self.batch = tf.placeholder(tf.float32, shape=(), name='batch')
        self.whiten = whiten
    


    def square_dist(self, X1, X2):
        X1 = X1 / tf.exp(self.log_theta)
        X1s = tf.reduce_sum(tf.square(X1), 1)
        X2 = X2 / tf.exp(self.log_theta)
        X2s = tf.reduce_sum(tf.square(X2), 1)
        return -2 * tf.matmul(X1, X2, transpose_b=True) + tf.reshape(X1s, (-1, 1)) + tf.reshape(X2s, (1, -1))
    
    def euclid_dist(self, X1, X2):
        r2 = self.square_dist(X1, X2)
        return tf.sqrt(r2 + 1e-12)

    def cov(self, x1, x2):
        if self.kernel == 'SE':
            K = tf.exp(self.log_sf2) * tf.exp((-self.square_dist(x1, x2) / 2))
        elif self.kernel == 'Matern12':
            r2 = self.euclid_dist(x1, x2) 
            K = tf.exp(self.log_sf2) * tf.exp(-r2)
        elif self.kernel == 'Matern32':
            r2 = self.euclid_dist(x1, x2) 
            K = tf.exp(self.log_sf2) * (1 + np.sqrt(3) * r2) * tf.exp(-np.sqrt(3) * r2)
        elif self.kernel == 'Matern52':
            r2 = self.euclid_dist(x1, x2) 
            K = tf.exp(self.log_sf2) * (1 + np.sqrt(5.) * r2 + 5. / 3. * tf.square(r2)) * tf.exp(-np.sqrt(5.) * r2)
        return K

    def lower_bound(self,):
        Kzz = self.cov(self.z, self.z) + tf.eye(self.m, dtype=tf.float32) * 1e-6
        self.L = tf.cholesky(Kzz)
        fmu, fcov = self.predict_f() # (None, 1), (None, 1)
        var_exps = self.variational_expectations(fmu, fcov, self.y)
        scale = self.num_data / self.batch
        return tf.reduce_sum(var_exps) * scale - self.KL()

    def variational_expectations(self, fmu, fcov, y):
        if self.sgp == 'dtc':
            return -0.5 * np.log(2 * np.pi) + 0.5 * self.log_beta \
                   -0.5 * tf.exp(self.log_beta) * (tf.square(y - fmu) + fcov)

        elif self.sgp == 'fitc':
            C_diag = tf.matrix_diag_part(self.B) + 1. / tf.exp(self.log_beta) # (batch, )
            return -0.5 * np.log(2 * np.pi) - 0.5 * tf.reduce_sum(tf.log(C_diag)) \
                   -tf.reduce_sum(0.5 * 1. / C_diag[:, None] * ((y - fmu)**2 + fcov))

        elif self.sgp == 'pic':
            C = self.B + tf.eye(self.batch, dtype=tf.float32) * 1. / tf.exp(self.log_beta)
            Lc = tf.cholesky(C)
            Lsigma = tf.cholesky(fcov)
            A = tf.matrix_triangular_solve(Lc, Lsigma) # Lc^-1 @ Lsigma
            B = tf.matrix_rtiangular_solve(Lc, (y - fmu)) # Lc^-1 @ (y - fmu)
            return -0.5 * np.log(2 * np.pi) - tf.reduce_sum(tf.log(tf.martix_diag_part(Lc))) \
                   -0.5 * tf.trace(tf.matmul(A, A, transpose_b=True)) - 0.5 * tf.reduce_sum(tf.matmul(B, B, transpose_a=True))
        
        else:
            raise ValueError('sgp can only be dtc, fitc, pic')



    def KL(self):
        """
        L: (m, m) lower triangular
        sigma: (M, ) which is not sigma^2
        """
        if not self.whiten:
            sigma = tf.exp(self.qlogdev)
            B = tf.matrix_triangular_solve(self.L, self.qmu, lower=True)
            C = tf.matrix_triangular_solve(self.L, tf.diag(sigma), lower=True)
            kl = tf.reduce_sum(0.5 * tf.matmul(B, B, transpose_a=True))
            kl += tf.reduce_sum(0.5 * tf.matrix_diag_part(tf.matmul(C, C, transpose_a=True)))
            kl += tf.reduce_sum(tf.log(tf.matrix_diag_part(self.L)))
            kl -= tf.reduce_sum(self.qlogdev)
            kl -= 0.5 * self.m
        
        else:
            kl = 0.5 * tf.reduce_sum(tf.exp(2 * self.qlogdev))
            kl += 0.5 * tf.reduce_sum(self.qmu**2)
            kl -= 0.5 * self.m
            kl -= tf.reduce_sum(self.qlogdev)

        return kl

    def KL2(self):
        sigma2 = tf.diag(tf.exp(2 * self.qlogdev))
        invKuu = tf.linalg.inv(tf.matmul(self.L, self.L, transpose_b=True))
        return 0.5 * (tf.trace(invKuu @ sigma2) + tf.reduce_sum(tf.transpose(self.qmu) @ invKuu @ self.qmu) - self.m 
                      + tf.log(tf.linalg.det(tf.matmul(self.L, self.L, transpose_b=True))) 
                      - tf.log(tf.linalg.det(sigma2)))





    def predict_f(self,):

        Kzx = self.cov(self.z, self.x) # (m, None)
        Kxx = self.cov(self.x, self.x) # (None, None)
        
        Ls = tf.diag(tf.exp(self.qlogdev)) 
        A = tf.matrix_triangular_solve(self.L, Kzx, lower=True) # L^-1 @ Kzx

        self.B = Kxx - tf.matmul(A, A, transpose_a=True) # Kxx - Kxz @ Kzz^-1 @ Kzx
        if not self.whiten: 
            A = tf.matrix_triangular_solve(tf.transpose(self.L), A, lower=False) # L^-T @ L^-1 @ Kzx

        fmu = tf.matmul(A, self.qmu, transpose_a=True) # Kzx @ L^-T @ qmu  ////  Kzx @ L^-T @ L^-1 @ qmu

        # B = tf.matrix_triangular_solve(self.L, self.qmu, lower=True) # L^-1 @ qmu
        # C = tf.matrix_triangular_solve(self.L, Ls, lower=True) # L^-1 @ Ls
        # D = tf.matmul(A, C, transpose_a=True) # Kxz @ Kzz^-1 @ qcov @ Kzz^-1 @ Kzx

        C = tf.matmul(A, Ls, transpose_a=True) # Kxz @ L^-T @ Ls  ////  Kzx @ L^-T @ L^-1 @ Ls        
        fcov = self.B + tf.matmul(C, C, transpose_b=True) # Kxx - Kxz @ Kzz^-1 @ Kzx +  (Kxz @ Kzz^-1 @ qcov @ Kzz^-1 @ Kzx  ////  Kxz @ L^-T @ qcov @ L^-1 @ Kzx)
        
        if self.sgp == 'dtc' or self.sgp == 'fitc':
            fcov = tf.diag_part(fcov)[:, None]
        elif self.sgp == 'pic':
            pass
        else:
            raise ValueError('sgp can only be dtc, fitc, pic')
        return fmu, fcov
        
