#ifndef _IMAGE_PROCESSING_H_
#define _IMAGE_PROCESSING_H_

#include <inc/anet_core.h>
#include <utility>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>

std::pair<std::vector<uint64>, uint64> get_pixels(std::string image_path);


#endif // _IMAGE_PROCESSING_H_