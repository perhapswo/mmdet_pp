// headers in STL
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <ros/ros.h>

// headers in 3rd-part
#include "../../pointpillars/pointpillars.h"
#include <gtest/gtest.h>
using namespace std;

int Txt2Arrary( float* &points_array , string file_name , int num_feature = 4)
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
  points_array = new float[temp_points.size()];
  for (int i = 0 ; i < temp_points.size() ; ++i) {
    points_array[i] = temp_points[i];
  }

  InFile.close();  
  return temp_points.size() / num_feature;
  // printf("Done");
};

int Bin2Arrary( float* &points_array , string file_name , int in_num_feature = 4, int out_num_feature = 4)
{
  ifstream InFile;
  InFile.open(file_name.data(), ios::binary);
  assert(InFile.is_open());
  vector<float> temp_points;
  float f;

  while (!InFile.eof())
  {
      InFile.read((char*)&f, sizeof(f));

      temp_points.push_back(f);
  }
  points_array = new float[temp_points.size()];
  int size = temp_points.size() / in_num_feature;
  for (int i = 0 ; i < size ; ++i) {
    for (int j = 0 ; j < out_num_feature; ++j)
    {
      points_array[i*out_num_feature + j] = temp_points[i*in_num_feature + j];
    }
  }

  InFile.close();  
  return size;
  // printf("Done");
};

void Boxes2Txt( std::vector<float> boxes , string file_name , int num_feature = 7)
{
    ofstream ofFile;
    ofFile.open(file_name , std::ios::out );  
    if (ofFile.is_open()) {
        for (int i = 0 ; i < boxes.size() / num_feature ; ++i) {
            for (int j = 0 ; j < num_feature ; ++j) {
                ofFile << boxes.at(i * num_feature + j) << " ";
            }
            ofFile << "\n";
        }
    }
    ofFile.close();
    return ;
};

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
  cout << "temp_points.size() :" << temp_points.size() << std::endl;
  for (int i = 0 ; i < temp_points.size() ; ++i) {
      anchor_data[i] = temp_points[i];
  }
  InFile.close();
  return;
}

TEST(PointPillars, __build_model__) {
  const std::string DB_CONF = "/home/jz/yxb/lidar_point_pillars_ws/src/mmdet_pp/bootstrap.yaml";
  std::cout << DB_CONF << std::endl;
  YAML::Node config = YAML::LoadFile(DB_CONF);
  std::string pfe_file,backbone_file; 
  if(config["UseOnnx"].as<bool>()) {
    pfe_file = config["PfeOnnx"].as<std::string>();
    backbone_file = config["BackboneOnnx"].as<std::string>();
  }else {
    pfe_file = config["PfeTrt"].as<std::string>();
    backbone_file = config["BackboneTrt"].as<std::string>();
  }
  std::cout << backbone_file << std::endl;
  const std::string pp_config = config["ModelConfig"].as<std::string>();
  PointPillars pp(
    config["ScoreThreshold"].as<float>(),
    config["NmsOverlapThreshold"].as<float>(),
    config["UseOnnx"].as<bool>(),
    pfe_file,
    backbone_file,
    pp_config
  );
  std::string file_name = config["InputFile"].as<std::string>();
  std::string anchor_file = config["AnchorFile"].as<std::string>();
  float* points_array;
  float* anchor_data;
  int in_num_points;
  // in_num_points = Txt2Arrary(points_array,file_name,5);
  in_num_points = Bin2Arrary(points_array,file_name,5);
  load_anchors(anchor_data, anchor_file);

  for (int _ = 0 ; _ < 4 ; _++)
  {

    std::vector<float> out_detections;
    std::vector<int> out_labels;
    std::vector<float> out_scores;

    cudaDeviceSynchronize();
    pp.DoInference(points_array, in_num_points, anchor_data, &out_detections, &out_labels , &out_scores);
    cudaDeviceSynchronize();
    int BoxFeature = 7;
    int num_objects = out_detections.size() / BoxFeature;

    std::string boxes_file_name = config["OutputFile"].as<std::string>();
    Boxes2Txt(out_detections , boxes_file_name );
    EXPECT_EQ(num_objects,228);
  }
};

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "TestNode");
  return RUN_ALL_TESTS();
}

// auto main(int argc, char **argv) -> int {
//   ::testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
