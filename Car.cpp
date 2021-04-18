#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
using namespace std;


// include omp header file here

#define RGB_COMPONENT_COLOR 255

struct PPMPixel {
    int red;
    int green;
    int blue;
};

typedef struct{
    int x, y, all;
    PPMPixel * data;
} PPMImage;

void readPPM(const char *filename, PPMImage& img){
    std::ifstream file (filename);
    if (file){
        std::string s;
        int rgb_comp_color;
        file >> s;
        if (s!="P3") {std::cout<< "error in format"<<std::endl; exit(9);}
        file >> img.x >>img.y;
        file >>rgb_comp_color;
        img.all = img.x*img.y;
        std::cout << s << std::endl;
        std::cout << "x=" << img.x << " y=" << img.y << " all=" <<img.all << std::endl;
        img.data = new PPMPixel[img.all];
        for (int i=0; i<img.all; i++){
            file >> img.data[i].red >>img.data[i].green >> img.data[i].blue;
        }

    }else{
        std::cout << "the file:" << filename << "was not found" << std::endl;
    }
    file.close();
}

void writePPM(char *filename, PPMImage & img){
    std::ofstream file (filename, std::ofstream::out);
    file << "P3"<<std::endl;
    file << img.x << " " << img.y << " "<< std::endl;
    file << RGB_COMPONENT_COLOR << std::endl;

    for(int i=0; i<img.all; i++){
        file << img.data[i].red << " " << img.data[i].green << " " << img.data[i].blue << (((i+1)%img.x ==0)? "\n" : " ");
    }
    file.close();
}

//
//write the function for shifting
//


void shiftPPM (PPMImage & image) {


    // initialization of three 1D arrays - RGB

    int* red = new int[image.all];
    
    for(int i = 0; i < image.all; ++i)
        red[i] = image.data[i].red;
    
    int* green = new int[image.all];
    
    for(int i = 0; i < image.all; ++i)
        green[i] = image.data[i].green;
          
    int* blue = new int[image.all];
    
    for(int i = 0; i < image.all; ++i)
        blue[i] = image.data[i].blue;
      
    
    // initialization of 3D array
    
    int*** image_array = new int**[image.x];
    
    for(int i = 0; i < image.x; ++i) {
        image_array[i] = new int*[image.y];
        for (int j = 0; j < image.y; j++) {
            image_array[i][j] = new int[3];       
        }
    }
    
    
    for(int i = 0; i < image.x; ++i) {
        for (int j = 0; j < image.y; j++) {
	    image_array[i][j][0] = red[i*image.y + j];
	    image_array[i][j][1] = green[i*image.y + j];  
	    image_array[i][j][2] = blue[i*image.y + j]; 
        }
    }
    

    // shift image
    
    for(int i = 0; i < image.x; ++i) {
        for (int j = 0; j < image.y; j++) {
            if (j == image.y - 1) {
            	image_array[i][j][0] = image_array[i][0][0];
            	image_array[i][j][1] = image_array[i][0][1];
            	image_array[i][j][2] = image_array[i][0][2];
            }
            else {
	        image_array[i][j][0] = image_array[i][j+1][0];  
	        image_array[i][j][1] = image_array[i][j+1][1];
	        image_array[i][j][2] = image_array[i][j+1][2];
	    } 
        }
    }
    
    // convert 2D to 1D array
    
    for (int i = 0; i < image.x; ++i) 
    {
        for (int j = 0; j < image.y; j++)
        {
            red[i*image.y + j] = image_array[i][j][0];
            green[i*image.y + j] = image_array[i][j][1];
            blue[i*image.y + j] = image_array[i][j][2];
        }
    }
    
    // convert 1D array to image PPM
    
    for (int i=0; i<image.all; i++){
            image.data[i].red = red[i];
            image.data[i].blue = blue[i];
            image.data[i].green = green[i];
    }


    // deallocate memory
    
    for (int i = 0; i < image.x; i++)
    {
        for (int j = 0; j < image.y; j++) {
            delete[] image_array[i][j];
        }
        delete[] image_array[i];
    }
    delete[] image_array;
    
    delete[] blue;
    delete[] green;
    delete[] red;
    
    cout << "Shift is done." << endl;

}





int main(int argc, char *argv[]){
    PPMImage image;
    
    readPPM("car.ppm", image);
    for (int i = 0; i < 10; i++) {		
        shiftPPM(image);
        string filename = "new_car_" + to_string(i) + ".ppm";
        char arr[filename.length() + 1]; 
 
        strcpy(arr, filename.c_str()); 
        cout<<"String to char array conversion:\n";
        for (int i = 0; i < filename.length(); i++) 
	    cout << arr[i]; 
        
        cout << filename << endl;
        writePPM(arr, image);
    }

        
    
    
    
    
    
    return 0;
}
