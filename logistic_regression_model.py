import os
import sys
import numpy
import pandas
import tensorflow as tf
from logistic_regression_input import *
from mlxtend.preprocessing import one_hot

tf.app.flags.DEFINE_integer('training_iteration', 100000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 16, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/home/deka/Desktop/test_tensorflow_serving/log_test_serving_model4', 'Working directory.')
FLAGS = tf.app.flags.FLAGS


def main(_):
    if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    	print('Usage: mnist_export.py [--training_iteration=x] '
    	  '[--model_version=y] export_dir')
    	sys.exit(-1)
    if FLAGS.training_iteration <= 0:
    	print('Please specify a positive value for training iteration.')
    	sys.exit(-1)
    if FLAGS.model_version <= 0:
    	print('Please specify a positive value for version number.')
    	sys.exit(-1)

    # Info about our data we run it from logistic_regression_input, logistic_regression_input.info_data()
    info_data()
    # Train model
    print('Training model...')
    # Use InteractiveSession
    sess = tf.InteractiveSession()
    # Import data
    train_X, test_X, train_Y, test_Y = input_data()
    # Hyperparameters
    learning_rate = 1e-7
    batch_size = 100
    display_step = 1
    ##################### Construct Model #####################
    X = tf.placeholder(dtype=tf.float32, shape=[None,3], name="first_placeholder")
    # This placeholder will feed our model. shape=[None,3], because we have 3 columns (N,EGT,WF)
    Y = tf.placeholder(dtype=tf.float32, shape=[None,3], name="second_placeholder")
    # This placeholder will be our trainer, shape=[None,3], because we have 3 classes (Status:0,1,2)
    # Set model weights
    W = tf.Variable(tf.zeros([3, 3]),name="weights")
    b = tf.Variable(tf.zeros([3]),name="biases")
    # Construct prediction
    pred = tf.nn.softmax(tf.matmul(X, W) + b)
    prediction_nn = tf.argmax((pred),1,name="prediction") # Prediction
    #predict = tf.argmax(pred, 1,name="prediction") # Predcition
    # Minimize error using cross entropy    #print(predicted_array)

    cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred), reduction_indices=1),name="cost_function")
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # Initializ all variables
    init = tf.global_variables_initializer()
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(FLAGS.training_iteration):
        avg_cost = 0.
        total_batch = int(len(train_X)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            if(i == total_batch):
                break
            batch_xs = train_X[batch_size*i:batch_size*(i+1),:]
            batch_ys = train_Y[batch_size*i:batch_size*(i+1),:]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs,
                                                          Y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")
    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: test_X, Y: test_Y}))
    # Test model
    test_predicition1 = sess.run(prediction_nn,feed_dict={X: test_X})
    test_pred_arr1 = numpy.array(test_predicition1)
    # Graphic to test model
    print(test_predicition1)
    plt.plot(test_pred_arr1,color='green') # Predicted line
    plt.ylabel('Status')
    plt.plot(test_Y,color='blue') # Test line
    plt.show()
    # Path to save model
    export_path_base = sys.argv[-1]
    export_path = os.path.join(
    tf.compat.as_bytes(export_path_base),
    tf.compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    # Build the signature_def_map.
    regression_inputs = tf.saved_model.utils.build_tensor_info(X) # Save first_placeholder to take prediction
    regression_outputs_prediction = tf.saved_model.utils.build_tensor_info(prediction_nn) # Save predcition function
    regression_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
    inputs={
        tf.saved_model.signature_constants.CLASSIFY_INPUTS:regression_inputs
    },
    outputs={
        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
            regression_outputs_prediction
    },
    method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
    ))
    tensor_info_x = tf.saved_model.utils.build_tensor_info(X) # Save first_placeholder to take prediction
    tensor_info_y = tf.saved_model.utils.build_tensor_info(prediction_nn) # Save prediction function
    prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
    inputs={'input_value':tensor_info_x},
    outputs={'output_value':tensor_info_y},
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        'predicted_value':
            prediction_signature,
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            regression_signature,
    },
    legacy_init_op = legacy_init_op)

    builder.save()
    writer = tf.summary.FileWriter("/home/deka/Desktop/test_tensorflow_serving/log_test_serving_model2") # Path to save tensorboard file
    writer.add_graph(sess.graph) # Save tensorboard
    merged_summary = tf.summary.merge_all()
    print("Done exporting!")

if __name__ == '__main__':
    tf.app.run()
