#include <iostream>
#include <vector>
#include <map> 
#include <cfloat>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "DBoW2/DBoW2.h"
#include "json/json.h"
#include "json/json-forwards.h"

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

int main (void)
{
int nimages=15;
int npairs=1;
std::string base_rgb_path="../../sequential_rotation_data/rgb_00";
std::string base_semantic_path="../../sequential_rotation_data/semantic_segmentation_00";
std::string vocab_file_path="../src/ORBvoc.txt";
cv::Ptr<cv::ORB> detector=cv::ORB::create();
cv::Ptr<cv::ORB> base_detector=cv::ORB::create();
std::cout<<"creating vocab"<<std::endl;
OrbVocabulary voc(9,3,DBoW2::TF_IDF,DBoW2::L1_NORM);
voc.loadFromTextFile(vocab_file_path);
std::cout<<"done"<<std::endl;
// voc.loadFromTextFile(vocab_file_path);

for(int i=0;i<nimages-1;++i)
    {
        //variable declarations for reading in first image and detecting features
        cv::Mat image1, segmented_image1;
        cv::Mat mask1;
        std::vector<cv::Mat> features11;
        std::vector<cv::Mat> features12;
        cv::Mat descriptors11;
        cv::Mat descriptors12;
        std::vector<cv::KeyPoint> keypoints11;
        std::vector<cv::KeyPoint> keypoints12;
        cv::Vec3f forklift_color={25, 197, 255};
        cv::Vec3f wheel_loader_color={255, 25, 140};
        cv::Vec3f husky_color={25, 255, 140};

        std::cout<<"reading in image "<<i<<std::endl;
        if (i<10)
        {
            image1=cv::imread(base_rgb_path+"0"+std::to_string(i)+".png");
            segmented_image1=cv::imread(base_semantic_path+"0"+std::to_string(i)+".png");
        }
        else
        {
            image1=cv::imread(base_rgb_path+std::to_string(i)+".png");
            segmented_image1=cv::imread(base_semantic_path+std::to_string(i)+".png");
        }
        
        cv::inRange(segmented_image1,wheel_loader_color,wheel_loader_color,mask1);
        detector->detectAndCompute(image1,mask1,keypoints11,descriptors11);
        cv::inRange(segmented_image1,husky_color,husky_color,mask1);
        detector->detectAndCompute(image1,mask1,keypoints12,descriptors12);
        create_feature_vector(descriptors11,features11);
        create_feature_vector(descriptors12,features12);
        std::vector<std::vector<cv::Mat>> first_features={features11,features12};

        
        OrbDatabase db(voc, false, 0);
        std::cout<<"adding features to database"<<std::endl;
        db.add(first_features[0]);
        db.add(first_features[1]);
        std::cout<<"adding features complete"<<std::endl;

        for(int j=i+1;j<nimages;++j)
        {
            std::cout<<npairs<<". "<<"image pair: "<<i<<"/"<<j<<std::endl;
            ++npairs;
            //create necessary variables and objects
            cv::Mat image2, segmented_image2;
            cv::Mat mask2;
            
            std::vector<cv::KeyPoint> keypoints21;
            std::vector<cv::KeyPoint> keypoints22;
            std::vector<cv::KeyPoint> base_keypoints2;
            std::vector<cv::KeyPoint> base_keypoints1;
            
            cv::Mat descriptors21;
            cv::Mat descriptors22;
            cv::Mat base_descriptors2;
            cv::Mat base_descriptors1;
            cv::Vec3f kp_clr1={0,255,0};
            cv::Vec3f kp_clr2={0,0,255};

            //read raw image and segmented image
            if (j<10)
            {
                image2=cv::imread(base_rgb_path+"0"+std::to_string(j)+".png");
                segmented_image2=cv::imread(base_semantic_path+"0"+std::to_string(j)+".png");
            }
            else
            {
                image2=cv::imread(base_rgb_path+std::to_string(j)+".png");
                segmented_image2=cv::imread(base_semantic_path+std::to_string(j)+".png");
            }
            //create the mask for the wheel loader
            cv::inRange(segmented_image2,wheel_loader_color,wheel_loader_color,mask2);

            //pass the mask and image to the detector
            detector->detectAndCompute(image2,mask2,keypoints21,descriptors21);

            //create the mask for the forklift
            cv::inRange(segmented_image2,husky_color,husky_color,mask2);

            //pass new mask and original image to detector
            detector->detectAndCompute(image2,mask2,keypoints22,descriptors22);


            //run detection without masking as baseline comparison
            base_detector->detectAndCompute(image1,cv::noArray(),base_keypoints1,base_descriptors1);
            base_detector->detectAndCompute(image2,cv::noArray(),base_keypoints2,base_descriptors2);
            //create feature vectors
            
            std::vector<cv::Mat> features21;
            std::vector<cv::Mat> features22;
            std::vector<cv::Mat> base_features1;
            std::vector<cv::Mat> base_features2;
            
            create_feature_vector(descriptors21,features21);
            create_feature_vector(descriptors22,features22);
            create_feature_vector(base_descriptors1,base_features1);
            create_feature_vector(base_descriptors2,base_features2);



            std::vector<std::vector<cv::Mat>> second_features={features21,features22};
            std::vector<std::string> labels={"wheel loader","husky"};

            DBoW2::QueryResults result;
            for(int i=0; i<second_features.size();++i)
            {
                std::cout<<"Querying Database for "<<labels[i]<<std::endl;
                db.query(second_features[i],result,2);
                std::cout<<result<<std::endl;
            }

            // std::vector<std::vector<cv::DMatch>> matches;
            // std::vector<cv::DMatch> good_matches;
            // cv::Ptr<cv::DescriptorMatcher> matcher=cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_L1);
            // matcher->knnMatch(descriptors11,descriptors21,matches,2);
            // std::cout<<"L1 distance ratio for first match "<<matches[0][0].distance/matches[0][1].distance<<std::endl;

            // for (std::vector<cv::DMatch> m :matches)
            // {
                
            //     if (0.7*m[1].distance > m[0].distance)
            //     {
            //         std::cout<<m[0].distance/m[1].distance<<std::endl;
            //         good_matches.push_back(m[0]);
            //     }
            // }
            // cv::Mat matched_img;
            // cv::drawMatches(image1,keypoints11,image2,keypoints21,good_matches,matched_img,kp_clr2,kp_clr2);
            // cv::imwrite("matches.png",matched_img);
            // cv::imshow("matched points",matched_img);
            // cv::waitKey();
        }
    }
return 0;
}
