
#include"ImageClassifier.hpp"

static bool PairCompare(const std::pair<float, int>& left,
                        const std::pair<float, int>& right)
{
    return left.first > right.first;
}

static std::vector<int> IndicesOfTopN(const std::vector<float>& v, int n)
{
    std::vector<std::pair<float,int>> tmp_v;
    for(unsigned int i=0; i<v.size(); ++i)
        tmp_v.push_back(std::make_pair(v[i], i));

    // 对tmp_v的前N个按值从大到小排序
    std::partial_sort(tmp_v.begin(), tmp_v.begin()+N, tmp_v.end(), PairCompare);

    std::vector<int> res;
    for(int i=0; i<N; ++i)
        res.push_back(tmp_v[0].second);
    return res;
}

ImageClassifier::ImageClassifier(const std::string& model_file,
                                 const std::string& train_file,
                                 const std::string& mean_file,
                                 const std::string& label_file)
{
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);

    // 加载测试网络
    m_net.reset(new Net<float>(model_file, TEST));
    m_net.CopyTrainedLayersFrom(trained_file);
    
    if(m_net.num_inputs() != 1) std::cout << " Network should have exactly on input " << std::endl;
    if(m_net.num_outputs()!= 1) std::cout << " Network should have exactly on output" << std::endl;
 
    // 提取网络输入层
    Blob<float>* input_layer = m_net->input_blobs()[0];
    m_num_channels = input_layer->channels();
    
    if(m_num_channels!=1 || m_num_channels!=3) std::cout << " The input layer should have 1 or 3 channels " << std::endl;

    m_input_geometry = cv::Size(input_layer->width(), input_layer->height());

    // 加载二进制均值文件
    SetMean(mean_file);

    // 获取标签
    std::ifstream labels(label_file.c_str());
    if(!labels.is_open()) std::cout << " Open the " << label_file << " faild " << std::endl;
    std::string line;
    while(std::getline(labels, line))
        m_labels.push_back(std::string(line));

    Blob<float>* output_layer = m_net->output_blobs()[0];
    if(m_labels.size()!=output_layer->channels())
        std::cout << " Number of labels is different from the output layer channels " << std::endl;

}


std::vector<Prediction> ImageClassifier::Classify(const cv::Mat& img, int N)
{
    // 预测图片
    std::vector<float> output = Predict(img);

    N = std::min<int>(m_labels.size(), N);

    // 得到前N个预测结果并返回
    std::vector<int> maxN = IndicesOfTopN(output, N);
    for(int i=0; i<N; ++i)
    {
        int idx = maxN[i];
        m_predictions.push_back(std::make_pair(m_labels[idx], output[idx]));
    }
    return m_predictions;
}


void ImageClassifier::SetMean(const std::string& mean_file)
{
    // 反序列化
    BlobProto blob_proto;
    ReadProtoFromBinaryFiledOrDie(mean_file.c_str(), &blob_proto);
    
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);

    if(mean_blob.channels() !=  m_num_channels)
        std::cout << " Number of channels of mean file doesn't match input layer " << std::endl;

    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for(int i=0; i<m_num_channels; ++i)
    {
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    cv::Mat mean;
    // 合并各通道
    cv::merge(channels, mean);
    cv::Scalar channel_mean = cv::mean(mean);
    m_mean = cv::Mat(m_input_geometry, mean.type(), channel_mean);
}



std::vector<float> ImageClassifier::Predict(const cv::Mat& img)
{
    Blob<float>* input_layer = m_net->input_blobs()[0];
    input_layer->Reshape(1, m_num_channels, m_input_geometry.height, m_input_geometry.width);

    /* Forward dimension change to all layers. */
    m_net->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    m_net->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = m_net->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{
    Blob<float>* input_layer = m_net->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();

    float* input_data = input_layer->mutable_cpu_data();

    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels)
{
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
      cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
      cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
      cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
      cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
      sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

}

