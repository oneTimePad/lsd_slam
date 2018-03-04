import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from depth_pkg.msg import Depth
import depth_predictor.models as models
import rospy
import time
def predict(model_data_path, pub, rate):


    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1



    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess)
        num = 1
        next_image = os.path.join("/home/lie/Desktop/pi_images","image%d"%num+".jpg")
        while not rospy.is_shutdown():
            if os.path.exists(next_image) :
                time.sleep(.2)
                # Read image1
                img = Image.open(next_image)
                img = img.resize([width,height], Image.ANTIALIAS)
                img = np.array(img).astype('float32')
                img = np.expand_dims(np.asarray(img), axis = 0)
                num+=1
                next_image = os.path.join("/home/lie/Desktop/pi_images","image%d"%num+".jpg")
                # Evalute the network for the given image
                pred = sess.run(net.get_output(), feed_dict={input_node: img})
                pred = pred[0,:,:,0]
                print(pred.shape)
                #pred+=0.5
                depth = pred.flatten()
                depth /= np.amax(depth)
                depth = depth*1+0.5
                print(depth[0])
                print("done %f %f" % (max(list(depth)), min(list(depth))))
                msg = Depth()
                msg.depth = list(depth)
                pub.publish(msg)

                # Plot result
                #fig = plt.figure()
                #ii = plt.imshow(pred, interpolation='nearest')
                #fig.colorbar(ii)
                #plt.show()
                while True:
                #pub.publish(msg)
                   rate.sleep()
            time.sleep(.5)
            rate.sleep()



        # Plot result
        #fig = plt.figure()
        #ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        #fig.colorbar(ii)
        #plt.show()



def main():
    # Parse arguments
    #parser = argparse.ArgumentParser()
    #parser.add_argument('model_path', help='Converted parameters for the model')
    #parser.add_argument('image_paths', help='Directory of images to predict')
    #args = parser.parse_args()
    MODEL_PATH= "/home/lie/depth_prediction/NYU_FCRN.ckpt"
    pub = rospy.Publisher('depth', Depth, queue_size=2)
    rospy.init_node('depth_predictor', anonymous=True)
    rate = rospy.Rate(40000)

    # Predict the image
    pred = predict(MODEL_PATH, pub, rate)

    os._exit(0)
