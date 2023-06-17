//
// g++ imAugmentation.cpp` -o img `pkg-config --libs opencv` ggdb `pkg-config --cflags  `imgRotation.cpp`
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <ctime>
#include <cstdlib>
class ImgAugmentation{
    private:

    cv::Mat img;
    double angle;
    void rotation(cv::Mat scr, double angle ){
        cv::Point2f pt(scr.cols/2, scr.rows/2);
        cv::Mat temp = getRotationMatrix2D(pt, angle, 1.0);
        cv::warpAffine(scr, this->img, temp, cv::Size(scr.cols, scr.rows));
    }
    void flipping(cv::Mat scr, char direction){
        if(direction==0 || direction==1 || direction==-1){
             cv::flip(scr, this->img, direction);
        }else{
            std::cout<<"please, select a valid selection value '0', '1' or '-1' "<<std::endl;
            //throw("please, select a valid selection value '0', '1' or '-1' ");
        }

    }
    void shearing(const cv::Mat & input, float Bx, float By){
        //https://stackoverflow.com/questions/46998895/image-shearing-c
        if (Bx*By == 1)
        {
            std::cout<<"Shearing: Bx*By==1 is forbidden"<<std::endl;
            //throw("Shearing: Bx*By==1 is forbidden");
        }
        //std::cout<<"channels ->"<< input.channels()<< " type ->"<< input.type()<<std::endl;
        if (input.type() == CV_8UC3 || input.type() ==CV_8UC2) {
            std::cout<<"not valid type"<<":" <<input.type()<<" "<< "valid type:"<< CV_8UC1 <<std::endl;
            //throw("not valid type");
        }
        // shear the extreme positions to find out new image size:
        std::vector<cv::Point2f> extremePoints; //vector<(0,1)>
        extremePoints.push_back(cv::Point2f(0, 0));
        extremePoints.push_back(cv::Point2f(input.cols, 0));
        extremePoints.push_back(cv::Point2f(input.cols, input.rows));
        extremePoints.push_back(cv::Point2f(0, input.rows));

        for (unsigned int i = 0; i < extremePoints.size(); ++i)
        {
            cv::Point2f & pt = extremePoints[i];
            pt = cv::Point2f(pt.x + pt.y*Bx, pt.y + pt.x*By);
        }

        cv::Rect offsets = cv::boundingRect(extremePoints); //[900 x 283 from (0, 0)]

        cv::Point2f offset = -offsets.tl(); //[0, 0]
        cv::Size resultSize = offsets.size();//[900 x 283]

        this->img = cv::Mat::zeros(resultSize, input.type()); // every pixel here is implicitely shifted by "offset"

        // perform the shearing by back-transformation
        for (int j = 0; j < img.rows; ++j)
        {
            for (int i = 0; i < img.cols; ++i)
            {
                cv::Point2f pp(i, j);
                pp = pp - offset; // go back to original coordinate system

                // go back to original pixel:
                cv::Point2f p;
                p.x = (-pp.y*Bx + pp.x) / (1 - By*Bx);
                p.y = pp.y - p.x*By;

                if ((p.x >= 0 && p.x < input.cols) && (p.y >= 0 && p.y < input.rows))
                {
                    // TODO: interpolate, if wanted (p is floating point precision and can be placed between two pixels)!
                    //img.at<cv::Vec3b>(j, i) = input.at<cv::Vec3b>(p);
                    img.at<uchar>(j,i) = input.at<uchar>(p);
                    //std::cout<<"pixel -> "<<(int) img.at<uchar>(j, i)<<std::endl;
                }
            }
        }
    }
    void cropping(cv::Mat image, const int cropSizeW, const int cropSizeH){
        cv::Mat croppedImage, image2;
        //cv::Mat background(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        const int offsetW = (image.cols - cropSizeW) / 2;
        const int offsetH = (image.rows - cropSizeH) / 2;
        //std::cout<< "W -> "<<offsetW<<" H -> "<< offsetH << " Y -> "<<cropSizeW << " X -> "<<cropSizeH<<std::endl;
        cv::Rect roi(offsetW, offsetH, cropSizeW, cropSizeW);
        croppedImage = image(roi);
        image2 = image.clone();
        cv::Mat copy(image2, cv::Rect(offsetW,offsetH,croppedImage.cols, croppedImage.rows)); //copy(image.rows, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        image2.setTo(0);
        croppedImage.copyTo(copy);
        this->img = image2;
        //std::cout << "Cropped image dimension: " << image.cols << " X " << image.rows << std::endl;

    }
    void contrast_brightness(cv::Mat image, double alpha, int beta){

        this->img = cv::Mat::zeros( image.size(), image.type() );
        for(int y =0; y<image.rows; y++){
            for(int x =0;x<image.cols;x++ ){
                img.at<uchar>(y,x) = cv::saturate_cast<uchar>(alpha*image.at<uchar>(y,x) + beta);
                //for(int c=0; c<image.channels(); c++){
                    //std::cout<< cv::saturate_cast<uchar>( alpha*image.at<uchar>(y,x) + beta )<<std::endl;
                    //img.at<cv::Vec3b>(y,x)[c]=cv::saturate_cast<uchar>( alpha*image.at<cv::Vec3b>(y,x)[c] + beta);
                //}
            }
        }
    }
    void AddGaussianNoise(const cv::Mat image, double Mean=0.0, double StdDev=10.0){
        if(image.empty())
        {
            std::cout<<"[Error]! Input Image Empty!";
        }

        cv::Mat image_16SC;
        cv::Mat copy(image.size(), image.type());
        cv::Mat mGaussian_noise(image.size(),image.type(), image.type());
        cv::randn(mGaussian_noise,cv::Scalar::all(Mean), cv::Scalar::all(StdDev));

        image.convertTo(image_16SC,image_16SC.type());
        //std::cout << "channels -> " << mGaussian_noise.channels()<< " -> "<<copy.channels() <<" -> "<< image_16SC.channels()<<std::endl;
        cv::addWeighted(image_16SC, 1.0, mGaussian_noise, 1.0, 0.0, image_16SC);
        image_16SC.convertTo(copy,image.type());

        this->img = copy;
    }
    int randNumberGenerator(int ceiling){
        std::srand(time(0));
        return (int)(std::rand()%ceiling) +1;
    }
    void algorithmSelector(cv::Mat image, int random_number, double angle, int crop_w, int crop_h, float bright_alpha, int contrast, int noise_mean, float stdDev){

        if(random_number ==1){
            rotation(image, angle);
        }else if(random_number==2){
            flipping(image, 0);
        }else if(random_number==3){
            flipping(image, 1);
        }else if(random_number==4){
            flipping(image, -1);
        }else if(random_number==5){
            shearing(image, 0.7,0.0);
        }else if(random_number==6){
            shearing(image,0.0, 0.7);
        }else if(random_number==7){
            cropping(image, crop_w, crop_h);
        }else if(random_number==8){
            contrast_brightness(image, bright_alpha, contrast);
        }else if(random_number==9){
            AddGaussianNoise(image, noise_mean, stdDev);
        }
        else{
            std::cout << "this value has none function associated  -> "<< random_number<<std::endl;
        }
    }
    public:
    ImgAugmentation(cv::Mat scr, double angle, int crop_w, int crop_h, float bright_alpha, int contrast, int noise_mean, float stdDev){
        //rotation(scr, angle);
        //flipping(scr, 0);
        //shear(scr, bx,by);
       //cropping(scr, 98, 108);
       //changing_contrast_brightness(scr, 2.0, 2);
       //AddGaussianNoise(scr, 50,20.0);

       int random_number = randNumberGenerator(8);
       //std::cout << random_number<<std::endl;
       algorithmSelector(scr, random_number, angle, crop_w, crop_h, bright_alpha, contrast, noise_mean, stdDev);
    }
    cv::Mat get_rotated_img(){
        return img;
    }
};


int main(){
    cv::Mat scr=cv::imread("/home/cristian/PycharmProjects/tEDRAM/tEDRAM2/training_data/binocular_imgs/binocular_img_left11.png", cv::IMREAD_GRAYSCALE);
    if( !scr.data )
    {
        std::cout<<"Error loadind src n"<<std::endl;
        return -1;
    }
    std::cout<<scr.type() <<" "<<CV_8UC1<<std::endl;
    ImgAugmentation img(scr,30.0, 98, 108, 2.0, 2, 70, 40.0);
    cv::imshow("source:", scr);
    cv::imshow("rotated:", img.get_rotated_img());
    cv::waitKey(0);
    return 0;
}