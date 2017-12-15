# EECS484LinearDropOut
A simple exploration project to the drop-out mechanism in neural network 

##Keep_drop modification
User can change the keep_drop value in TensorFlow.py at line 49 to change the keep rate(the rate of keeping the neural unit output, 1 will means the drop out layer will not drop any unit output)
   *sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,keep_prob: 1})*
##Iteration modification
User can also change the number of training iteration at line 47. 
   *  for _ in range(20000): ...*

##Weights dimension modification
User can also change the input output layers' weight dimension by editing the input layer and output layer's matrix as long as the input and output sides are 784 to 10 

*For example : *

  Input layer
  W2 = tf.Variable(tf.random_uniform([784, 392], minval=-1, maxval=1, dtype=tf.float32))
  b2 = tf.Variable(tf.random_uniform([392], minval=-1, maxval=1, dtype=tf.float32))
  y2 = tf.nn.sigmoid(tf.matmul(x, W2) + b2)

  Drop out layers
  keep_prob = tf.placeholder(tf.float32)
  y_drop = tf.nn.dropout(y2, keep_prob)

  Output Layer
  W1 = tf.Variable(tf.zeros([392, 10]))
  b1 = tf.Variable(tf.zeros([10]))
  y1 = tf.nn.softmax(tf.matmul(y_drop, W1) + b1)
  
  
*Can be changed to *

  Input layer
  W2 = tf.Variable(tf.random_uniform([784, 500], minval=-1, maxval=1, dtype=tf.float32))
  b2 = tf.Variable(tf.random_uniform([500], minval=-1, maxval=1, dtype=tf.float32))
  y2 = tf.nn.sigmoid(tf.matmul(x, W2) + b2)

  Drop out layers
  keep_prob = tf.placeholder(tf.float32)
  y_drop = tf.nn.dropout(y2, keep_prob)

  Output Layer
  W1 = tf.Variable(tf.zeros([500, 10]))
  b1 = tf.Variable(tf.zeros([10]))
  y1 = tf.nn.softmax(tf.matmul(y_drop, W1) + b1)
  
