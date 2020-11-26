#include <src/image_processing/image_processing.h>

std::pair<std::vector<uint64>, uint64> get_pixels(std::string image_path){
    
    std::vector<uint64> pixels;

    cv::Mat src_image = cv::imread(image_path);
    if (!src_image.data) {
        return std::make_pair(std::move(pixels),(uint64)1);
    }

    pixels.reserve(src_image.rows * src_image.cols);

    for (auto i = 0; i < src_image.rows; i++)
    {
        for (auto j = 0; j < src_image.cols; j++){
            cv::Vec3b& color = src_image.at<cv::Vec3b>(i,j); // getting rgb color
            
            std::stringstream string_s;
            string_s << std::hex << (color[0] << 16 | color[1] << 8 | color[2]); // getting hex color by transforming rgb

            uint64 dec_color{};
            string_s >> dec_color;

            pixels.push_back(dec_color); // hex color -> dec value and putting it to pixels vector
        }   
    }

    return std::make_pair(std::move(pixels),(uint64)0);
}