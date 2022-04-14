

#include <iostream>
#include <opencv2/core/core.hpp>
#include <string>
// Drawing shapes
#include <opencv2/imgproc.hpp>

typedef unsigned char uchar;
std::string labels[6] = {"NHC","HC","NFC","FC","NC","DPC"};
struct Points{
    int start;
    int end;
};
class BoundingBox{
    public:
        BoundingBox(cv::Mat***images, uchar labels[100][10][6],  double positions [100][10][6], size_t batch_size );
        ~BoundingBox();
        bool check_labels(uchar labels[1000][10][6]);
        void create_BB(cv::Mat***images, uchar labels[1000][10][6],  double positions [100][10][6]);
        void fill_BB(Mat***images, uchar labels[100][10][6], double positions [100][10][6], int steps, int lab_o, int lab_t,int i, int j, int x, int y, int w, int h,
                      double*bb_o, int clip, bint predicted);
    private:
        static bool comp(uchar a, uchar b){
                return (a < b);
            }
        int max_elem(uchar *array, size_t size){
            int index, max;
            max = (int) array[0];
            index = 0;
            for(int i=1;i<size;i++){
                if(array[i]>max){
                    max = (int) array[i];
                    index = i;
                }
            }
            return index;
        }
        cv::Mat create_image(cv::Mat input){
            cv::Mat image(120,160,  CV_8UC3,
              cv::Scalar(255, 255, 255));
            return image;
        }
        cv::Mat hstack(cv::Mat&output, cv::Mat&image, cv::Mat&lab){
            cv::Mat out(cv::Size(image.cols*3, image.rows), image.type(), cv::Scalar(0,0,0));
            cv::Mat temp = out(cv::Rect(0,0,output.cols,output.rows));
            output.copyTo(temp);
            temp = out(cv::Rect(0,0,image.cols,image.rows));
            image.copyTo(temp);
            temp = out(cv::Rect(0,0,lab.cols,lab.rows));
            lab.copyTo(temp);
        }
        std::pair<Points, Points> bb_dimensions(int x, int y, int w, int h, double*bb_t, int clip);
        int N, steps;
        int x,y,h,w,clip=30, last_loc =1;
        bool collisions;
        size_t batch_size, seq_size, dis_size=7, height=120, width=160;
        cv::Mat output, output2;

};


