#include "boundingBox.h"
#include <algorithm>
BoundingBox::BoundingBox(cv::Mat***images, uchar labels[100][10][6],  double positions [100][10][6],size_t batch_size ){
      this->seq_size = 1100 -1000; //10
      this->batch_size = batch_size; //100
      this->steps = batch_size/(seq_size-1);
      create_BB(images, labels,positions);

}

BoundingBox::~BoundingBox(){

}
std::pair<Points, Points> BoundingBox::bb_dimensions(int x, int y, int w, int h, double*bb_t, int clip){
        Points limit_1, limit_2;
        std::pair<Points,Points> dimensions;
        this->w = (int)bb_t[0]*100;
        this->h = (int)bb_t[4]*100;
        if (w > clip) w = clip;
        if (h > clip) h = clip;
        if (w < 1) w = 1;
        if (h < 1) h = 1;
        this->x = (int)(bb_t[2]*50+50)-w/2;
        this->y = (int)(bb_t[5]*50+50)-h/2;
        if (x<0) x = 0;
        if (y<0) y = 0;
        if (x+w>100) w = 99-x;
        if (y+h>100) h = 99-y;
        limit_1.start = x;
        limit_1.end = x+w;
        limit_2.start = y;
        limit_2.end = y+h;
        dimensions.first = limit_1;
        dimensions.second = limit_2;
}

void BoundingBox::create_BB(cv::Mat***images, uchar labels[100][10][6],  double positions [100][10][6]){
    bool check = check_labels(labels);
    if(check){
        this->collisions = true;
    }

    cv::Mat image, blank, lab, out;
    cv::Mat subImage, resizedImg;
    double *bb_t;
    int lab_t, lab_o;
    float*var;
    double * bb_o;
    std::pair<Points,Points> dim;

    for(int i =0;i<batch_size;i++){ //N
        for(int j =0; j<seq_size;j++){ //10
            for(int k=0;k<dis_size;k++){ //7
                image = images[i][j][k].clone();
                bb_t = positions[i][j];
                lab_t = max_elem(labels[i][j], 6); //std::max_element(labels[i]+0, labels[i]+5,comp);
                var = new float[6];
                std::memset(var,0,sizeof(var));
                if (collisions ==false){
                    //draw true bb
                    dim = bb_dimensions(x,y,w,h,bb_t,clip);

                    cv::Point point_1(dim.first.start, dim.second.start);
                    cv::Point point_2(dim.first.end,dim.second.end);
                    cv::rectangle(image, point_1,point_2,cv::Scalar(0, 255, 0), 1, cv::LINE_8);
                }
                bb_o = positions[i][j];
                lab_o = max_elem(labels[i][j], 6);
                var[lab_o] = 1;
                bool perfect;
                if(lab_t!=lab_o) perfect = false;

                //draw prediction
                dim = bb_dimensions(x,y,w,h,bb_t,clip);

                cv::Point point_1(dim.first.start,dim.second.start);
                cv::Point point_2(dim.first.end ,dim.second.end);
                cv::rectangle(image, point_1, point_2, cv::Scalar(0, 0, 255), 1, cv::LINE_8);
                blank = create_image(blank);//blank = np.zeros((100,20,3))+255
                output = create_image(out); //output = np.zeros((100,20,3))+255
                cv::putText(output,std::to_string(lab_t), cv::Point(3,18), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 2);

                lab = create_image(lab);//lab = np.zeros((100,20,3))+255

                cv::putText(lab,std::to_string(lab_o), cv::Point(3,18), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(128,128,128), 2);

                this->output = hstack(out, image, lab);
                if (x<0) x = 0;
                if (y<0) y = 0;
                image = images[i][j][k].clone();
                subImage = image(cv::Range(dim.second.start, dim.second.end), cv::Range(dim.first.start, dim.second.end));
                cv::resize(subImage, resizedImg, cv::Size(26,26), cv::INTER_LINEAR);
                cv::resize(resizedImg, resizedImg, cv::Size(100,100), cv::INTER_CUBIC);
                cv::rectangle(image, cv::Point(0, 0), cv::Point(99, 99), cv::Scalar(0, 0, 255), 2);
                this->output2 = hstack(blank, image, blank);
            }
        }
    }
}
void BoundingBox::fill_BB(Mat***images, uchar labels[100][10][6], double positions [100][10][6], int steps, int lab_o, int lab_t,int i, int j, int x, int y, int w, int h,
                      double*bb_o, int clip, bint predicted){
    for(int i =0;i<batch_size;i++){ //N
        for(int j =0; j<seq_size;j++){ //10
            for(int k=0;k<dis_size;k++){ //7


            }
        }
    }
}
bool BoundingBox::check_labels(uchar labels[100][10][6]){
    size_t size = sizeof(labels[0]) / sizeof(labels[0][0][0]);
    if(labels!=NULL && size ==6){
        return true;
    } else{
        return false;
    }
}


int main(){

    BoundingBox bb;


}



