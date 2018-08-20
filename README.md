<h1> Logistic Regression, Tensorflow Serving </h1>

<h2> Script to visualization data and feed model. </h2>

    import numpy
    import pandas
    import matplotlib.pyplot as plt
    from mlxtend.preprocessing import one_hot

<p> Import packages for array, dataframe, data vizualization and encode data to one hot </p>

    def import_data():
        dataframe = pandas.read_csv('/home/deka/Desktop/datasets/Dataset1.csv')
        return dataframe
        
<p> Import dataset </p>

    def info_data():
        data = import_data()
        print(data.head(10),"\n") # print first 10 raws
        print(data.info(),"\n") # print info about dataframe
        print(data.shape,"\n") # print dataframe shape
        print(data.describe(),"\n") # print info about values
        print(data.corr(),"\n") # dataframe correlation

        # Convert object type to float type
        data.N=pandas.to_numeric(pandas.Series(data.N), errors='coerce');
        data.EGT=pandas.to_numeric(pandas.Series(data.EGT), errors='coerce');
        data.WF=pandas.to_numeric(pandas.Series(data.WF), errors='coerce');
        data.dropna(inplace=True) # Drop all NaN values

        # Graphics
        plt.plot(data['N'])
        plt.ylabel('Parameter N')
        plt.show()
        plt.plot(data['EGT'],color='red')
        plt.ylabel('Parameter EGT')
        plt.show()
        plt.plot(data['WF'],color='green')
        plt.ylabel('Parameter WF')
        plt.show()
     
<p> Data vizualization, information about data and graphics </p> 

    def input_data():
        df = import_data() # import dataframe
        divide = 3584 # value for divide data on train and test
        df.drop('dateandtime',inplace=True,axis=1)
        # Convert object type to float type
        df.N=pandas.to_numeric(pandas.Series(df.N), errors='coerce');
        df.EGT=pandas.to_numeric(pandas.Series(df.EGT), errors='coerce');
        df.WF=pandas.to_numeric(pandas.Series(df.WF), errors='coerce');
        df.dropna(inplace=True)
        # Divide data on train and test without shuffle
        train_X = numpy.array(df.values[:divide,0:3])
        train_Y_p = numpy.array(df.values[:divide,3:])
        test_X = numpy.array(df.values[divide:,0:3])
        test_Y_p = numpy.array(df.values[divide:,3:])
        # Peapare one hot encode
        train_Y_p = train_Y_p.astype('int') # Convert to int type (train data)
        test_Y_p = test_Y_p.astype('int') # Convert to int type (test data)
        trf = train_Y_p.ravel() # Need be dimension 1, to encode in one hot
        tref = test_Y_p.ravel() # Need be dimension 1, to encode in one hot
        traf_Y = one_hot(trf, num_labels=3) # num_labels need be same as your classes (0,1,2)
        tres_Y = one_hot(tref, num_labels=3) # num_labels need be same as your classes (0,1,2)
        train_Y_en = numpy.array(traf_Y) # one hot numpy array (train data)
        test_Y_en = numpy.array(tres_Y) # one hot numpy array (test data)
        
        return train_X, test_X, train_Y_en, test_Y_en
    
        # One hot is:
        #    ___________________
        #    |_cat_|_dog_|_rat_|
        #    |__1__|__0__|__0__|
        #    |__0__|__1__|__0__|
        #    |__0__|__0__|__1__|

        
 <p> Divide data on train and test, encode data to one hot </p>
 
 <h1> Script to train model and save it. </h1>
 
 <p> Run in like this : python logistic_regression_model.py /home/deka/Desktop/test_tensorflow_serving/log_test_serving_model3/. Need give path to save model. </p>
 
         import os
        import sys
        import numpy
        import pandas
        import tensorflow as tf
        from logistic_regression_input import *
        from mlxtend.preprocessing import one_hot
        
<p> Import packages </p>

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
         
![Parameter_N](https://user-images.githubusercontent.com/37526996/44318401-7eb98e80-a457-11e8-888e-4541da3cfb69.png)
![Parameter_EGT](https://user-images.githubusercontent.com/37526996/44318404-80835200-a457-11e8-80c3-f26dd36112d6.png)
![Parameter_WF](https://user-images.githubusercontent.com/37526996/44318407-837e4280-a457-11e8-8ae3-8b87bf49d106.png)
            
<p> Graphics, we call from logistic_regression_input.py script. </p>
            
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
            
<p> Create logistic regression model, create batch </p>
            
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
                    
 <p> Train model </p>

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
            
<p> Test model. Predict vizualization. </p>

![Predicted classes](https://user-images.githubusercontent.com/37526996/44318293-f6d38480-a456-11e8-93bc-af4f03622fad.png)
            
            # Path to save model
            export_path_base = sys.argv[-1]
            export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(FLAGS.model_version)))
            print('Exporting trained model to', export_path)
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)
            
<p> Saving model path, version, name. Saving model builder. </p>
            
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
            
<p> Creating signature map, and save model. </p>
            
            writer = tf.summary.FileWriter("/home/deka/Desktop/test_tensorflow_serving/log_test_serving_model2") # Path to save tensorboard file
            writer.add_graph(sess.graph) # Save tensorboard
            merged_summary = tf.summary.merge_all()
            print("Done exporting!")
           
<p> Save tensorboard file </p>

        if __name__ == '__main__':
            tf.app.run()
                        
<h2>  Run server. </h2>

<p> Run server like this: ```tensorflow_model_server --port=6660 --model_name=deka --model_base_path=/home/deka/Desktop/log_test_serving/test_serving_model3/```.  </p>

<p> port="need_give_port", model_name="give_own_name_for_your_model", model_base_path="give_path_to_your_model". </p>

<p> If you run successfully, you can see this, in your command </p>

            2018-08-20 08:48:34.942003: I tensorflow_serving/model_servers/main.cc:157] Building single TensorFlow model file config:  model_name: deka model_base_path: /home/deka/Desktop/test_tensorflow_serving/log_test_serving_model3/
            2018-08-20 08:48:34.942212: I tensorflow_serving/model_servers/server_core.cc:462] Adding/updating models.
            2018-08-20 08:48:34.942234: I tensorflow_serving/model_servers/server_core.cc:517]  (Re-)adding model: deka
            2018-08-20 08:48:35.043158: I tensorflow_serving/core/basic_manager.cc:739] Successfully reserved resources to load servable {name: deka version: 16}
            2018-08-20 08:48:35.043258: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: deka version: 16}
            2018-08-20 08:48:35.043304: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: deka version: 16}
            2018-08-20 08:48:35.043400: I external/org_tensorflow/tensorflow/contrib/session_bundle/bundle_shim.cc:360] Attempting to load native SavedModelBundle in bundle-shim from: /home/deka/Desktop/test_tensorflow_serving/log_test_serving_model3/16
            2018-08-20 08:48:35.043485: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: /home/deka/Desktop/test_tensorflow_serving/log_test_serving_model3/16
            2018-08-20 08:48:35.045907: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
            2018-08-20 08:48:35.072871: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:113] Restoring SavedModel bundle.
            2018-08-20 08:48:35.075987: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:148] Running LegacyInitOp on SavedModel bundle.
            2018-08-20 08:48:35.078524: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:233] SavedModel load for tags { serve }; Status: success. Took 35030 microseconds.
            2018-08-20 08:48:35.078589: I tensorflow_serving/servables/tensorflow/saved_model_warmup.cc:83] No warmup data file found at /home/deka/Desktop/test_tensorflow_serving/log_test_serving_model3/16/assets.extra/tf_serving_warmup_requests
            2018-08-20 08:48:35.078727: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: deka version: 16}
            2018-08-20 08:48:35.080675: I tensorflow_serving/model_servers/main.cc:327] Running ModelServer at 0.0.0.0:6660 ...

 
 <p> Test server. </p>
 
 <p> Run this to test server: deka@grave:~/Desktop/logistic_regression_scripts$ python test_server_ex.py </p>
 
        from grpc.beta import implementations
        import tensorflow as tf
        import numpy
        import pandas
        from logistic_regression_input import *

        from tensorflow.core.framework import types_pb2
        from tensorflow.python.platform import flags
        from tensorflow_serving.apis import predict_pb2
        from tensorflow_serving.apis import prediction_service_pb2


        tf.app.flags.DEFINE_string('server', 'localhost:6660',
                                   'inception_inference service host:port')
        FLAGS = tf.app.flags.FLAGS


        def main(_):
            x_tr,x_te,y_tr,y_te = input_data()
            x_test_arr = numpy.float32(x_te)
            #feed_value2 = numpy.asarray([90.234,1352.642,5978.735])
            # Prepare request
            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'deka'
            request.inputs['inputs'].dtype = types_pb2.DT_FLOAT
            #request.inputs['inputs'].float_val.append(feed_value2)
            request.inputs['inputs'].CopyFrom(
                tf.contrib.util.make_tensor_proto(x_test_arr))
            request.output_filter.append('classes')
            # Send request
            host, port = FLAGS.server.split(':')
            channel = implementations.insecure_channel(host, int(port))
            stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
            prediction = stub.Predict(request, 5.0)  # 5 secs timeout
            predd = numpy.asarray(prediction)
            floats = prediction.outputs['classes'].int64_val
            pred_array = numpy.asarray(floats)
            df = pandas.DataFrame({"predicted_value":pred_array})
            print(df)
            df.to_csv('/home/deka/Desktop/prediction_log.csv') # Save prediction as dataset


            if __name__ == '__main__':
                tf.app.run()

 
 <p> Predicted value: </p>
 
                 predicted_value
            0                   1
            1                   1
            2                   1
            3                   1
            4                   1
            5                   1
            6                   1
            7                   0
            8                   0
            9                   0
            10                  0              
            0                   1
            1                   1
            2                   1
            3                   1
            4                   1
            5                   1
            6                   1
            7                   0
            8                   0
            9                   0
            10                  0
            11                  0
            12                  0
            13                  0
            14                  0
            15                  0
            16                  0
            17                  0
            18                  0
            19                  0
            20                  0
            21                  0
            22                  0
            23                  0
            24                  0
            25                  0
            26                  0
            27                  0
            28                  0
            29                  0
            ...               ...
            1136                1
            1137                1
            1138                1
            1139                1
            1140                1
            1141                1
            1142                1
            1143                1
            1144                1
            1145                1
            1146                1
            1147                1
            1148                1
            1149                1
            1150                1
            1151                1
            1152                1
            1153                1
            1154                1
            1155                1
            1156                1
            1157                1
            1158                1
            1159                1
            1160                1
            1161                1
            1162                1
            1163                1
            1164                1
            1165                1

            [1166 rows x 1 columns]

            11                  0
            12                  0
            13                  0
            14                  0
            15                  0
            16                  0
            17                  0
            18                  0
            19                  0
            20                  0
            21                  0
            22                  0
            23                  0
            24                  0
            25                  0
            26                  0
            27                  0
            28                  0
            29                  0
            ...               ...
            1136                1
            1137                1
            1138                1
            1139                1
            1140                1
            1141                1
            1142                1
            1143                1
            1144                1
            1145                1
            1146                1
            1147                1
            1148                1
            1149                1
            1150                1
            1151                1
            1152                1
            1153                1
            1154                1
            1155                1
            1156                1
            1157                1
            1158                1
            1159                1
            1160                1
            1161                1
            1162                1
            1163                1
            1164                1
            1165                1

            [1166 rows x 1 columns]

