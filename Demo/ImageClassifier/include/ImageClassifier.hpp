
#include<map>
#include<pair>
#include<vector>
#include<string>
#include<iostream>

#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

#include<caffe/caffe.hpp>
using namespace caffe;

// 标签，得分
typedef std::pair<std::string, float> Predction;

class ImageClassifier
{
    public:
        ImageClassifier(const std::string& model_file, const std::string& train_filed,
                        const std::string& mean_file,  const std::string label_file);
        std::vector<Prediction> Classify(const cv::Mat& img, int, label);

    private:
        void SetMean(const std::string& mean_file);
        std::vector<float> Predict(const cv::Mat& img);
        void WrapInputLayer(std::vector<cv::Mat>* input_channels);
        void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

    private:

        int         m_num_channels;

        cv::Mat     m_mean;
        cv::Size    m_input_geometry;

        shared_ptr< Net<float> > m_net;
        std::vector<std::string> m_labels;
}
