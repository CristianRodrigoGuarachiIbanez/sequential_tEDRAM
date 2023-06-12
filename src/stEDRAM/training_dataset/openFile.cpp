#include<fstream>
#include<iostream>
#include<string>
int main() {
    int sum = 0;
    int x;
    std::ifstream inFile;

    inFile.open("./label_data.txt", std::ios::in);
    if (!inFile) {
        std::cout << "Unable to open file";
        exit(1); // terminate with error
    }

    //while (inFile >> x)
    std::string file;
    while(std::getline(inFile,file )){
        //sum = sum + x;
        std::cout<<" cotent -> "<<file<<std::endl;
    }

    inFile.close();
    std::cout << "Sum = " << sum << std::endl;
    return 0;
}