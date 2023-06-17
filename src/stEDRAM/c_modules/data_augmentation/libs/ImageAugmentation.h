//
// g++ imAugmentation.cpp` -o img `pkg-config --libs opencv` ggdb `pkg-config --cflags  `imgRotation.cpp`
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <ctime>
#include <cstdlib>
namespace img{
    class AugmentationManager{
        private:

        cv::Mat IMG;
        double angle;
        void rotation(cv::Mat &scr, double angle );
        void flipping(cv::Mat &scr, char direction);
        void shearing(const cv::Mat & input, float Bx, float By);
        void cropping(cv::Mat &image, const int cropSizeW, const int cropSizeH);
        void AddGaussianNoise(const cv::Mat &image, double Mean, double StdDev);
        void contrast_brightness(cv::Mat &image, double alpha, int beta);
        void algorithmSelector(cv::Mat &image, int random_number, double angle, int crop_w, int crop_h, float bright_alpha, int contrast, int noise_mean, float stdDev);
        void shear(cv::Mat&image);
        inline int randNumberGenerator(int &ceiling){
            std::srand(time(0));
            return (int)(std::rand()%ceiling) +1;
        }

        public:
        AugmentationManager(cv::Mat &scr, int random_number, double angle, int crop_w, int crop_h, float bright_alpha, int contrast, int noise_mean, float stdDev);
        AugmentationManager(const AugmentationManager&augmentor){
            std::cout<<"COPIED"<<std::endl;
            //IMG = augmentor.IMG
        }
        inline cv::Mat getAugmentedImage(int rows, int cols){
            cv::Mat resized;
            cv::resize(IMG, resized, cv::Size(cols,rows), cv::INTER_LINEAR);
            return resized;
        }
        ~AugmentationManager(){
            IMG.release();
        }
    };
}