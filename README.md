# EECS484LinearDropOut
A simple exploration project to the drop-out mechanism in neural network 

## Keep_drop modification
User can change the keep_drop value in TensorFlow.py at line 49 to change the keep rate(the rate of keeping the neural unit output, 1 will means the drop out layer will not drop any unit output).<br />
   &nbsp;&nbsp;&nbsp; <span style="color:blue">*sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,keep_prob: 1})*</span>
## Iteration modification
User can also change the number of training iteration at line 47.<br />
   &nbsp;&nbsp;&nbsp <span style="color:blue">*for _ in range(20000): ...*</span>

## Weights dimension modification
User can also change the input output layers' weight dimension by editing the input layer and output layer's matrix as long as the input and output sides are 784 to 10.<br />

*For example:* <br />

   <span style="color:blue"> &nbsp;&nbsp;&nbsp; Input layer<br />
   &nbsp;&nbsp;&nbsp; W2 = tf.Variable(tf.random_uniform([784, 392], minval=-1, maxval=1, dtype=tf.float32))<br />
  &nbsp;&nbsp;&nbsp; b2 = tf.Variable(tf.random_uniform([392], minval=-1, maxval=1, dtype=tf.float32))<br />
   &nbsp;&nbsp;&nbsp; y2 = tf.nn.sigmoid(tf.matmul(x, W2) + b2)<br /> </span>

  Drop out layers<br />
  <span style="color:blue"> &nbsp;&nbsp;&nbsp; keep_prob = tf.placeholder(tf.float32)<br />
  &nbsp;&nbsp;&nbsp;  y_drop = tf.nn.dropout(y2, keep_prob)<br /> </span>

  Output Layer<br />
   <span style="color:blue"> &nbsp;&nbsp;&nbsp; W1 = tf.Variable(tf.zeros([392, 10]))<br />
   &nbsp;&nbsp;&nbsp; b1 = tf.Variable(tf.zeros([10]))<br />
   &nbsp;&nbsp;&nbsp; y1 = tf.nn.softmax(tf.matmul(y_drop, W1) + b1)<br /></span>
  
  
*Can be changed to* <br />

  Input layer<br />
   <span style="color:blue">&nbsp;&nbsp;&nbsp; W2 = tf.Variable(tf.random_uniform([784, 500], minval=-1, maxval=1, dtype=tf.float32))<br />
   &nbsp;&nbsp;&nbsp; b2 = tf.Variable(tf.random_uniform([500], minval=-1, maxval=1, dtype=tf.float32))<br />
   &nbsp;&nbsp;&nbsp; y2 = tf.nn.sigmoid(tf.matmul(x, W2) + b2)<br /></span>

  Drop out layers<br />
  <span style="color:blue"> &nbsp;&nbsp;&nbsp; keep_prob = tf.placeholder(tf.float32)<br />
   &nbsp;&nbsp;&nbsp; y_drop = tf.nn.dropout(y2, keep_prob)<br /></span>

  Output Layer<br />
  <span style="color:blue"> &nbsp;&nbsp;&nbsp; W1 = tf.Variable(tf.zeros([500, 10]))<br />
   &nbsp;&nbsp;&nbsp; b1 = tf.Variable(tf.zeros([10]))<br />
   &nbsp;&nbsp;&nbsp; y1 = tf.nn.softmax(tf.matmul(y_drop, W1) + b1)<br /></span>
