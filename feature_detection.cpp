/*
This script was written by Jeremy Kilbride in April 2024
*/

#include <iostream>
#include <vector>
#include <map> 
#include <cfloat>
#include <string>
#include <fstream>
#include <iomanip>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "DBoW2/DBoW2.h"
// #include "json/json.h"
// #include "json/json-forwards.h"

// class SemanticSegment:  //for later
// {
// public:
//     int image_id=0;
//     cv::Vec3f segment_color={0,0,0};
//     std::string class_label="";
//     cv::Mat descriptors;
//     std::vector<cv::Mat> features;

//     
// }

void create_feature_vector(const cv::Mat &descriptors, std::vector<cv::Mat> &features)
{
    features.resize(descriptors.rows);
    for (int i=0;i<descriptors.rows;++i)
        {
            features[i]=descriptors.row(i);
        }
}

void concatenate_features (std::vector<cv::Mat> current_features, const std::vector<cv::Mat> &incoming_features)
{
    for (int i=0; i<incoming_features.size();++i)
    {
        current_features.push_back(incoming_features[i]);
    }
}

void mask_detect_compute(cv::Ptr<cv::ORB> &detector,cv::Mat &image,cv::Mat &seg_image,std::vector<cv::Mat> &descriptors,std::vector<cv::Mat> &masks, std::vector<cv::Vec3f> &colors,std::vector<std::vector<cv::KeyPoint>> &keypoints,std::vector<cv::Mat> &inv_masks)
{
    for (int i=0;i<colors.size();++i)
    {
        cv::inRange(seg_image,colors[i],colors[i],masks[i]);
        cv::bitwise_not(masks[i],inv_masks[i]);
        detector->detectAndCompute(image,masks[i],keypoints[i],descriptors[i]);
    }
    cv::Mat total_inv_mask=inv_masks[0];
    if (colors.size()>1)
    {
       for (int j=1;j<colors.size();++j)
       {
        cv::bitwise_and(inv_masks[j],total_inv_mask,total_inv_mask);
       } 
    }
    inv_masks.push_back(total_inv_mask);
}


int main (void)
{
int nimages=30;
int npairs=1;

//get date and time
std::time_t t = std::time(nullptr);
std::tm tm = *std::localtime(&t);
std::stringstream ss;
ss<<std::put_time(&tm,"%F_%H_%M_%S");

//get paths for image data and vocab
std::string base_path="../../two_forklifts_data/";
std::string base_rgb_path=base_path+"rgb_0";
std::string base_semantic_path=base_path+"instance_segmentation_0";
std::string vocab_file_path="../src/ORBvoc.txt";

//create ORB detectors and vocabulary
cv::Ptr<cv::ORB> detector=cv::ORB::create();
cv::Ptr<cv::ORB> base_detector=cv::ORB::create();
std::cout<<"creating vocab"<<std::endl;
OrbVocabulary voc(9,3,DBoW2::TF_IDF,DBoW2::L1_NORM);
voc.loadFromTextFile(vocab_file_path);
std::cout<<"done"<<std::endl;

//create the output file
ofstream outputfile("output_data"+ss.str()+".txt");
outputfile<<"path: "<< base_path<<std::endl;
outputfile<<"num_images: "<< nimages<<std::endl;

cv::Vec3f forklift1_color={25, 25, 255};
cv::Vec3f wheel_loader_color={255, 25, 140};
cv::Vec3f forklift2_color={255, 255, 25};
cv::Vec3f kp_clr2={0,0,255};
cv::Vec3f kp_clr1={0,255,0};
cv::Vec3f kp_clr3={255,0,255};
std::vector<cv::Vec3f> colors;
colors.push_back(forklift1_color);
colors.push_back(forklift2_color);

for(int i=0;i<nimages-1;++i)
    {
        //variable declarations for reading in first image and detecting features
        cv::Mat image1, segmented_image1;
        std::vector<cv::Mat> masks1;
        masks1.resize(2);
        std::vector<cv::Mat> inv_masks1;
        inv_masks1.resize(2);
        std::vector<cv::Mat> features11;
        std::vector<cv::Mat> features12;
        std::vector<cv::Mat> concat_features;
        cv::Mat descriptors11;
        cv::Mat descriptors12;
        std::vector<cv::Mat> descriptors1;
        descriptors1.push_back(descriptors11);
        descriptors1.push_back(descriptors12);
        std::vector<cv::KeyPoint> keypoints11;
        std::vector<cv::KeyPoint> keypoints12;
        std::vector<std::vector<cv::KeyPoint>> keypoints1;
        keypoints1.push_back(keypoints11);
        keypoints1.push_back(keypoints12);

        cv::Mat image1_filtered_kps;
        cv::Mat image2_filtered_kps;


        std::cout<<"reading in image "<<i<<std::endl;
        if (i<10)
        {
            image1=cv::imread(base_rgb_path+"00"+std::to_string(i)+".png");
            segmented_image1=cv::imread(base_semantic_path+"00"+std::to_string(i)+".png");
        }
        else if(i>=10 && i<100)
        {
            image1=cv::imread(base_rgb_path+"0"+std::to_string(i)+".png");
            segmented_image1=cv::imread(base_semantic_path+"0"+std::to_string(i)+".png");
        }
        else 
        {
            image1=cv::imread(base_rgb_path+std::to_string(i)+".png");
            segmented_image1=cv::imread(base_semantic_path+std::to_string(i)+".png");
        }
        image1_filtered_kps=image1.clone();
        mask_detect_compute(detector,image1,segmented_image1,descriptors1,masks1,colors,keypoints1,inv_masks1);

        if (i==0)
        {
            cv::imwrite("mask.png",masks1[0]);      //save the mask for presentation purposes
            cv::imwrite("inverse_mask1.png",inv_masks1[0]);
            cv::drawKeypoints(image1,keypoints1[0],image1,kp_clr1);
            cv::imwrite("mask2.png",masks1[1]);      //save the masks for presentation purposes
            cv::imwrite("inverse_mask2.png",inv_masks1[1]);
            cv::imwrite("inverse_mask3.png",inv_masks1[2]);
            cv::drawKeypoints(image1,keypoints1[1],image1,kp_clr2);
            cv::imwrite("two_forks_w_kps.png",image1);

        }
   
        create_feature_vector(descriptors1[0],features11);
        create_feature_vector(descriptors1[1],features12);
        std::vector<std::vector<cv::Mat>> first_features={features11,features12};

        //create the data base and add features from first image
        OrbDatabase db(voc, false, 0);
        std::cout<<"adding features to database"<<std::endl;
        db.add(first_features[0]);
        db.add(first_features[1]);
        std::cout<<"adding features complete"<<std::endl;

        for(int j=i+1;j<nimages;++j)
        {
            std::cout<<"pair"<<i<<"/"<<j<<std::endl;
            outputfile<<"P: "<<i<<"/"<<j<<std::endl;
            ++npairs;
            //create necessary variables and objects
            cv::Mat image2, segmented_image2;
            cv::Mat mask2;
            cv::Mat inv_mask21;
            cv::Mat inv_mask22;
            cv::Mat inv_mask23;
            
            std::vector<cv::KeyPoint> keypoints21;
            std::vector<cv::KeyPoint> keypoints22;
            std::vector<cv::KeyPoint> base_keypoints2;
            std::vector<cv::KeyPoint> base_keypoints1;
            
            cv::Mat descriptors21;
            cv::Mat descriptors22;
            cv::Mat base_descriptors2;
            cv::Mat base_descriptors1;

            //read second raw image and segmented image
            if (j<10)
            {
                image2=cv::imread(base_rgb_path+"00"+std::to_string(j)+".png");
                segmented_image2=cv::imread(base_semantic_path+"00"+std::to_string(j)+".png");
            }
            else if(j>=10 && j<100)
            {
                image2=cv::imread(base_rgb_path+"0"+std::to_string(j)+".png");
                segmented_image2=cv::imread(base_semantic_path+"0"+std::to_string(j)+".png");
            }
            else
            {
                image2=cv::imread(base_rgb_path+std::to_string(j)+".png");
                segmented_image2=cv::imread(base_semantic_path+std::to_string(j)+".png");
            }
            image2_filtered_kps=image2.clone();

            //create the masks for the first forklift
            cv::inRange(segmented_image2,forklift1_color,forklift1_color,mask2);
            cv::bitwise_not(mask2,inv_mask21);
            
            //pass the mask and image to the detector
            detector->detectAndCompute(image2,mask2,keypoints21,descriptors21);

            //create the masks for the forklift
            cv::inRange(segmented_image2,forklift2_color,forklift2_color,mask2);
            cv::bitwise_not(mask2,inv_mask22);
            cv::bitwise_and(inv_mask21,inv_mask22,inv_mask23);

            //pass new mask and original image to detector
            detector->detectAndCompute(image2,mask2,keypoints22,descriptors22);


            //run detection with inverse masking as baseline comparison
            base_detector->detectAndCompute(image1,inv_masks1[2],base_keypoints1,base_descriptors1);
            base_detector->detectAndCompute(image2,inv_mask23,base_keypoints2,base_descriptors2);
            
            if(i==0 && j==1)
            {
                cv::drawKeypoints(image1_filtered_kps,base_keypoints1,image1_filtered_kps,kp_clr3);
                cv::imwrite("filter_ORB_detection.png",image1_filtered_kps);
            }

            
            std::vector<cv::Mat> features21;
            std::vector<cv::Mat> features22;
            std::vector<cv::Mat> base_features1;
            std::vector<cv::Mat> base_features2;
            //create feature vectors for second image
            create_feature_vector(descriptors21,features21);
            create_feature_vector(descriptors22,features22);
            create_feature_vector(base_descriptors1,base_features1);
            create_feature_vector(base_descriptors2,base_features2);
            



            std::vector<std::vector<cv::Mat>> second_features={features21,features22};
            std::vector<std::string> labels={"forklift1","forklift2"};

            //query database with features from second image
            DBoW2::QueryResults result;
            for(int i=0; i<second_features.size();++i)
            {
                outputfile<<"L: "<<labels[i]<<std::endl;
                db.query(second_features[i],result,2);
                std::cout<<result<<std::endl;
                if (result.size()!=0)
                {
                    float sc=result[0].Score;
                    outputfile<<"S: "<<sc<<std::endl;
                    if (result.size()>1)
                    {
                        float sc2=result[1].Score;
                        outputfile<<"S2: "<<sc2<<std::endl;
                    }
                    }
                else
                {
                    outputfile<<"S: "<<"0"<<std::endl;
                }

            }
            //find feature correspondences between images
            std::vector<std::vector<cv::DMatch>> matches;
            std::vector<cv::DMatch> good_matches;
            cv::Ptr<cv::DescriptorMatcher> matcher=cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_L1);
            matcher->knnMatch(descriptors1[0],descriptors21,matches,2);
            for (std::vector<cv::DMatch> m :matches)
            {
                if (0.7*m[1].distance > m[0].distance)
                {
                    // std::cout<<m[0].distance/m[1].distance<<std::endl;
                    good_matches.push_back(m[0]);
                }
            }
            outputfile<<"N: "<<good_matches.size()<<std::endl;
            if (i==0 && j==1)
            {
                
                cv::Mat matched_img;
                cv::drawMatches(image1,keypoints1[0],image2,keypoints21,good_matches,matched_img,kp_clr2,kp_clr2);
                cv::imwrite("matches1.png",matched_img);
                
            }
            // if (i==0 & j<=3)
            // {
            //     concatenate_features(concat_features,features21);
            //     db.add(concat_features);
            // }
        }
    }
outputfile.close();
return 0;
}
