#ifndef __EDGE_FILTER_HPP__
#define __EDGE_FILTER_HPP__

#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

void contourEdgeFilter( const cv::Mat &edge_in,
                        cv::Mat &edge_out,
                        const int edge_lower_th=50, 
                        const int edge_upper_th=100 ) {

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(edge_in, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    cv::Mat edge_new = cv::Mat_<uchar>(edge_in.size(), 0);

    for (auto &cnt : contours) {

        auto rect = cv::minAreaRect(cnt);

        int w = rect.size.width;
        int h = rect.size.height;
        int l = cnt.size();
        
        bool flag = false;
        if (w > edge_upper_th || h > edge_upper_th) {
            flag = true;
        }
        else if (w > edge_lower_th || h > edge_lower_th) {
            if (l < (w+h)*4) {
                flag = true;
            }
        }
            
        if (flag) {
            for (auto &pt : cnt) {
                edge_new.at<uchar>(pt.y, pt.x) = 255;
            }
        }
    }

    edge_new.copyTo(edge_out);
}

#endif