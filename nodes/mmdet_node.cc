#include <iostream>
#include "mmdet_ros.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "mmdet_pp");
  MMdetPPROS app;
  app.createROSPubSub();
  ros::spin();

  return 0;
}