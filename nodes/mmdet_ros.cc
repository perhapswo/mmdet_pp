// headers in STL
#include <chrono>
#include <cmath>

// headers in PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_ros/transforms.h>

// headers in ROS
#include <tf/transform_datatypes.h>
#include <ros/package.h>

// headers in local files
#include "autoware_msgs/DetectedObjectArray.h"
#include "autoware_perception_msgs/DynamicObjectWithFeatureArray.h"
#include "mmdet_ros.h"

void load_anchors(float* &anchor_data, string file_name)
{
  ifstream InFile;
  InFile.open(file_name.data());
  assert(InFile.is_open());

  vector<float> temp_points;
  string c;

  while (!InFile.eof())
  {
      InFile >> c;

      temp_points.push_back(atof(c.c_str()));
  }
  anchor_data = new float[temp_points.size()];
  for (int i = 0 ; i < temp_points.size() ; ++i) {
      anchor_data[i] = temp_points[i];
  }
  InFile.close();
  return;
}

MMdetPPROS::MMdetPPROS() 
    : DB_CONF_("/home/jz/yxb/lidar_point_pillars_ws/src/mmdet_pp/bootstrap.yaml"), NUM_POINT_FEATURE_(4), OUTPUT_NUM_BOX_FEATURE_(7), TRAINED_SENSOR_HEIGHT_(PP_SENSOR_HEIGHT), NORMALIZING_INTENSITY_VALUE_(255.0f), BASELINK_FRAME_("base_footprint")
{
    //ros related param
    private_nh_.param<bool>("baselink_support", baselink_support_, false);

    YAML::Node config = YAML::LoadFile(DB_CONF_);
    if(config["UseOnnx"].as<bool>()) {
        pfe_file_ = config["PfeOnnx"].as<std::string>();
        backbone_file_ = config["BackboneOnnx"].as<std::string>();
    }else {
        pfe_file_ = config["PfeTrt"].as<std::string>();
        backbone_file_ = config["BackboneTrt"].as<std::string>();
    }

    pp_config_ = config["ModelConfig"].as<std::string>();
    point_pillars_ptr_.reset(new PointPillars(
    config["ScoreThreshold"].as<float>(),
    config["NmsOverlapThreshold"].as<float>(),
    config["UseOnnx"].as<bool>(),
    pfe_file_,
    backbone_file_,
    pp_config_));
    std::string anchor_file = config["AnchorFile"].as<std::string>();
    load_anchors(anchor_data_, anchor_file);
    ROS_INFO("finish init multihead_point_pillars...");
}

void MMdetPPROS::createROSPubSub()
{
  sub_points_ = nh_.subscribe<sensor_msgs::PointCloud2>("/points_raw", 1, &MMdetPPROS::pointsCallback, this);
  pub_objects_ = nh_.advertise<autoware_msgs::DetectedObjectArray>("/detection/lidar_detector/objects", 1);
  pub_feature_objects_ = nh_.advertise<autoware_perception_msgs::DynamicObjectWithFeatureArray>("/detection/lidar_detector/feature_objects", 3);
}

geometry_msgs::Pose MMdetPPROS::getTransformedPose(const geometry_msgs::Pose &in_pose, const tf::Transform &tf)
{
  tf::Transform transform;
  geometry_msgs::PoseStamped out_pose;
  transform.setOrigin(tf::Vector3(in_pose.position.x, in_pose.position.y, in_pose.position.z));
  transform.setRotation(
      tf::Quaternion(in_pose.orientation.x, in_pose.orientation.y, in_pose.orientation.z, in_pose.orientation.w));
  geometry_msgs::PoseStamped pose_out;
  tf::poseTFToMsg(tf * transform, out_pose.pose);
  return out_pose.pose;
}

void MMdetPPROS::pubDetectedObject(const std::vector<float> &detections, const std_msgs::Header &in_header)
{
  autoware_msgs::DetectedObjectArray objects;
  autoware_perception_msgs::DynamicObjectWithFeatureArray feature_objects;
  objects.header = in_header;
  feature_objects.header = in_header;
  int num_objects = detections.size() / OUTPUT_NUM_BOX_FEATURE_;
  for (size_t i = 0; i < num_objects; i++)
  {
    autoware_msgs::DetectedObject object;
    object.header = in_header;
    object.valid = true;
    object.pose_reliable = true;

    object.pose.position.x = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 0];
    object.pose.position.y = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 1];
    object.pose.position.z = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 2];

    // Trained this way
    float yaw = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 6];
    yaw += M_PI / 2;
    yaw = std::atan2(std::sin(yaw), std::cos(yaw));
    geometry_msgs::Quaternion q = tf::createQuaternionMsgFromYaw(-yaw);
    object.pose.orientation = q;

    if (baselink_support_)
    {
      object.pose.position.z += TRAINED_SENSOR_HEIGHT_;
      object.pose = getTransformedPose(object.pose, angle_transform_inversed_);
    }

    // Again: Trained this way
    object.dimensions.x = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 4];
    object.dimensions.y = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 3];
    object.dimensions.z = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 5];

    //Only detects car in Version 1.0
    object.label = "car";

    objects.objects.push_back(object);

    // fill feature_object
    autoware_perception_msgs::DynamicObjectWithFeature feature_object;
    feature_object.object.semantic.type = autoware_perception_msgs::Semantic::CAR;
    feature_object.object.state.pose_covariance.pose = object.pose;
    feature_object.object.state.orientation_reliable = false;
    feature_object.object.shape.type = autoware_perception_msgs::Shape::BOUNDING_BOX;
    feature_object.object.shape.dimensions = object.dimensions;

    feature_objects.feature_objects.push_back(feature_object);
  }
  pub_objects_.publish(objects);
  pub_feature_objects_.publish(feature_objects);
}

void MMdetPPROS::getBaselinkToLidarTF(const std::string &target_frameid)
{
  try
  {
    tf_listener_.waitForTransform(BASELINK_FRAME_, target_frameid, ros::Time(0), ros::Duration(1.0));
    tf_listener_.lookupTransform(BASELINK_FRAME_, target_frameid, ros::Time(0), baselink2lidar_); // should be lidar2base
    analyzeTFInfo(baselink2lidar_);
    has_subscribed_baselink_ = true;
  }
  catch (tf::TransformException ex)
  {
    ROS_ERROR("%s", ex.what());
  }
}

void MMdetPPROS::analyzeTFInfo(tf::StampedTransform baselink2lidar)
{
  tf::Vector3 v = baselink2lidar.getOrigin();
  offset_z_from_trained_data_ = v.getZ() - TRAINED_SENSOR_HEIGHT_;

  tf::Quaternion q = baselink2lidar_.getRotation();
  angle_transform_ = tf::Transform(q);
  angle_transform_inversed_ = angle_transform_.inverse();
}

void MMdetPPROS::pclToArray(const pcl::PointCloud<pcl::PointXYZI>::Ptr &in_pcl_pc_ptr, float *out_points_array,
                                 const float offset_z)
{
  for (size_t i = 0; i < in_pcl_pc_ptr->size(); i++)
  {
    pcl::PointXYZI point = in_pcl_pc_ptr->at(i);
    out_points_array[i * NUM_POINT_FEATURE_ + 0] = point.x;
    out_points_array[i * NUM_POINT_FEATURE_ + 1] = point.y;
    out_points_array[i * NUM_POINT_FEATURE_ + 2] = point.z + offset_z;
    out_points_array[i * NUM_POINT_FEATURE_ + 3] = float(point.intensity / NORMALIZING_INTENSITY_VALUE_);
  }

  // // debug pcd points
  // ofstream ofFile;
  // ofFile.open("/home/jz/yxb/lidar_point_pillars_ws/src/mmdet_pp/test/data/testdata/pcd_points.txt" , std::ios::out );  
  // if (ofFile.is_open()) {
  //     for (int i = 0 ; i < in_pcl_pc_ptr->size() ; ++i) {
  //         ofFile << out_points_array[i * NUM_POINT_FEATURE_ + 0] << " ";
  //         ofFile << out_points_array[i * NUM_POINT_FEATURE_ + 1] << " ";
  //         ofFile << out_points_array[i * NUM_POINT_FEATURE_ + 2] << " ";
  //         ofFile << out_points_array[i * NUM_POINT_FEATURE_ + 3] << " ";
  //         ofFile << "\n";
  //     }
  // }
  // ofFile.close();
}

void MMdetPPROS::pointsCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*msg, *pcl_pc_ptr);
  if (baselink_support_)
  {
    if (!has_subscribed_baselink_)
    {
      getBaselinkToLidarTF(msg->header.frame_id);
    }
    pcl_ros::transformPointCloud(*pcl_pc_ptr, *pcl_pc_ptr, angle_transform_);
  }

  float *points_array = new float[pcl_pc_ptr->size() * NUM_POINT_FEATURE_];
  if (baselink_support_ && has_subscribed_baselink_)
  {
    pclToArray(pcl_pc_ptr, points_array, offset_z_from_trained_data_);
  }
  else
  {
    pclToArray(pcl_pc_ptr, points_array);
  }

  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

  std::vector<float> out_detections;
  std::vector<int> out_labels;
  
  std::vector<float> out_scores;
  point_pillars_ptr_->DoInference(points_array, pcl_pc_ptr->size(), anchor_data_, &out_detections, &out_labels , &out_scores);

  int num_objects = out_detections.size() / OUTPUT_NUM_BOX_FEATURE_;
  ROS_INFO("detect objects %d ...", num_objects);

  std::chrono::duration<double> time_elapse =
      std::chrono::steady_clock::now() - start;
  ROS_INFO("%.2lf ms", std::chrono::duration_cast<std::chrono::nanoseconds>(time_elapse)
                               .count() *
                           1.0e-6);

  delete[] points_array;
  pubDetectedObject(out_detections, msg->header);
}
