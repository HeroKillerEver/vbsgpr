import tensorflow as tf
import numpy as np



class Solver(object):
	"""docstring for Solver"""
	def __init__(self, model, data_train, data_test, batch=1, epoch=10, log_dir = 'logs'):
		super(Solver, self).__init__()
		self.model = model
		self.data_train = data_train
		self.data_test = data_test
		self.batch = batch
		self.epoch = epoch
		self.n_blocks = data_train.__len__()
		self.n_test = sum([self.data_test[i].shape[0] for i in range(1, 1001)])
		self.step = int(self.n_blocks / self.batch) 
		self.log_dir = log_dir
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth=True


	def train(self):

		vbsgpr = self.model
		vbsgpr.build_model()

		if tf.gfile.Exists(self.log_dir):
			tf.gfile.DeleteRecursively(self.log_dir)
		tf.gfile.MakeDirs(self.log_dir)

		with tf.Session(config=self.config) as sess:
			sess.run(tf.global_variables_initializer())
			summary_writer = tf.summary.FileWriter(self.log_dir, graph=sess.graph)
			saver = tf.train.Saver()

			print ("Start Training...!")


			for epoch in range(self.epoch):
				idx = np.random.permutation(np.arange(1, self.n_blocks + 1))
				for step in range(self.step):
					train_tmp = [ self.data_train[x] for x in idx[step*self.batch: (step+1)*self.batch] ]
					train_batch = np.vstack(train_tmp)
					x, y = train_batch[:, 0:-1], train_batch[:, -1]
					feed_dict = {vbsgpr.x: x[0:100, :], vbsgpr.y: y[0:100]}

					lb, lcov, _ = sess.run([vbsgpr.lb, vbsgpr.lcov, vbsgpr.train_op], feed_dict)

					print ('epoch: [{0}/{1}], step: [{2}/{3}], lb: [{4:.4f}]'.format(epoch, self.epoch, step, self.step, lb))

					if step % self.epoch == 0:
						print ('starting evaluation...!')
						rmse = 0
						for block in range(1, self.n_blocks+1):
							x_test, y_test = self.data_test[block][:, 0:-1], self.data_test[block][:, -1]
							feed_dict = {vbsgpr.x_test: x_test, vbsgpr.y_test: y_test}
							l2_loss = sess.run(vbsgpr.l2_loss, feed_dict)
							rmse += l2_loss

						rmse = np.sqrt(rmse / self.n_test)
						print ('rmse: {0}'.format(rmse))



					# summary_writer.add_summary(summary)








