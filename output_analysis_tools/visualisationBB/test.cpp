#include <iostream>
#include <opencv2/core/core.hpp>
//#include <opencv2/tracking.hpp>
// Drawing shapes
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;
//g++ test.cpp -o test `pkg-config --libs --cflags opencv`

/*
// Driver Code
int main(int argc, char** argv)
{


    int arr[3][3] = {{1,2,3},{4,5,6},{7,8,9}};
    int* array;
    for(int i=0;i<3;i++){
        if(i==0){
        array = arr[i];
        }
    }
    for(int i=0;i<3;i++){
        std::cout<<array[i]<<std::endl;
    }
    //Mat img(120,160,CV_8UC3, Scalar(255, 0, 0));
    // Creating a blank image with
    // white background
    Mat image(500, 500, CV_8UC3,
              Scalar(255, 255, 255));

    // Check if the image is created
    // successfully or not
    if (!image.data) {
        std::cout << "Could not open or "
                  << "find the image\n";

        return 0;
    }

    // Top Left Corner
    Point p1(10, 80);

    // Bottom Right Corner
    Point p2(355, 555);

    int thickness = 2;

    // Drawing the Rectangle
    rectangle(image, p1, p2,
              Scalar(255, 0, 0),
              thickness, LINE_8);

    // Show our image inside a window
    imshow("Output", image);
    waitKey(0);


    return 0;
}
**/



int main (int argc, char **arv){

    // Read image
    Mat im = imread("image.jpg");
    // Select ROI
    Rect2d r = selectROI(im);
    // Crop image
    Mat imCrop = im(r);
    // Display Cropped Image
    imshow("Image", imCrop);
    waitKey(0);
    return 0;
}
