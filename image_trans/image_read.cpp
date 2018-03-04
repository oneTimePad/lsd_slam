#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <string.h>
#include <cstdlib>

int main(int argc, char** argv)
{

  ros::init(argc, argv, "image_publisher");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise("images", 1);


  ros::Rate loop_rate(5);

  char path[256];
  int i =1;
  sprintf(path,"/home/lie/Desktop/pi_images/image%d.jpg",i);
  while (nh.ok()) {
    cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    if(image.data){
       //printf("Sending");
       sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();
       pub.publish(msg);
       ros::spinOnce();
       i++;
       sprintf(path,"/home/lie/Desktop/pi_images/image%d.jpg",i);
    }
    loop_rate.sleep();
  }
}
