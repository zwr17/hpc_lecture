#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <iostream>
//#include <random>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cudnn.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec)+double(tv.tv_usec)*1e-6;
}

void readImages(const char *filename, std::vector<uint8_t> &data, int &width, int &height) {
  uint32_t image[4];
  FILE * fid = fopen(filename, "r");
  size_t size = fread(&image, sizeof(uint32_t), 4, fid);
  uint32_t nimage = __builtin_bswap32(image[1]);
  height = __builtin_bswap32(image[2]);
  width = __builtin_bswap32(image[3]);
  data.resize(nimage * width * height);
  size = fread(&data[0], sizeof(uint8_t), nimage * width * height, fid);
  fclose(fid);
}

void readLabels(const char *filename, std::vector<uint8_t> &labels) {
  uint32_t label[2];
  FILE * fid = fopen(filename, "r");
  size_t size = fread(&label, sizeof(uint32_t), 2, fid);
  uint32_t nlabel = __builtin_bswap32(label[1]);
  labels.resize(nlabel);
  size = fread(&labels[0], sizeof(uint8_t), nlabel, fid);
  fclose(fid);
}

#define BW 128

static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator) {
  return (nominator + denominator - 1) / denominator;
}

struct ConvBiasLayer {
  int in_channels, out_channels, kernel_size;
  int in_width, in_height, out_width, out_height;
  std::vector<float> pconv, pbias;
  ConvBiasLayer(int in_channels_, int out_channels_, int kernel_size_, int in_width_, int in_height_) :
    pconv(in_channels_ * kernel_size_ * kernel_size_ * out_channels_), pbias(out_channels_) {
    in_channels = in_channels_;
    out_channels = out_channels_;
    kernel_size = kernel_size_;
    in_width = in_width_;
    in_height = in_height_;
    out_width = in_width_ - kernel_size_ + 1;
    out_height = in_height_ - kernel_size_ + 1;
  }
};

struct MaxPoolLayer {
  int size, stride;
  MaxPoolLayer(int size_, int stride_) : size(size_), stride(stride_) {}
};

struct FullyConnectedLayer {
  int inputs, outputs;
  std::vector<float> pneurons, pbias;
  FullyConnectedLayer(int inputs_, int outputs_) :
    outputs(outputs_), inputs(inputs_), pneurons(inputs_ * outputs_), pbias(outputs_) {}
};

__global__ void FillOnes(float *vec, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  vec[idx] = 1.0f;
}

__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size)
    return;
  const int label_value = static_cast<int>(label[idx]);
  diff[idx * num_labels + label_value] -= 1.0f;
}

struct TrainingContext {
  cudnnHandle_t cudnnHandle;
  cublasHandle_t cublasHandle;
  cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor, pool1Tensor, conv2Tensor, conv2BiasTensor, pool2Tensor, fc1Tensor, fc2Tensor;
  cudnnFilterDescriptor_t conv1filterDesc, conv2filterDesc;
  cudnnConvolutionDescriptor_t conv1Desc, conv2Desc;
  cudnnConvolutionFwdAlgo_t conv1algo, conv2algo;
  cudnnConvolutionBwdFilterAlgo_t conv1bwfalgo, conv2bwfalgo;
  cudnnConvolutionBwdDataAlgo_t conv2bwdalgo;
  cudnnPoolingDescriptor_t poolDesc;
  cudnnActivationDescriptor_t fc1Activation;

  int m_batchSize;
  size_t m_workspaceSize;
  FullyConnectedLayer& ref_fc1, &ref_fc2;
  TrainingContext& operator=(const TrainingContext&) = delete;
  TrainingContext(const TrainingContext&) = delete;
  TrainingContext(int batch_size, ConvBiasLayer& conv1, MaxPoolLayer& pool1, ConvBiasLayer& conv2, MaxPoolLayer& pool2,
                  FullyConnectedLayer& fc1, FullyConnectedLayer& fc2) : ref_fc1(fc1), ref_fc2(fc2) {
    m_batchSize = batch_size;
    cublasCreate(&cublasHandle);
    cudnnCreate(&cudnnHandle);
    cudnnCreateTensorDescriptor(&dataTensor);
    cudnnCreateTensorDescriptor(&conv1Tensor);
    cudnnCreateTensorDescriptor(&conv1BiasTensor);
    cudnnCreateTensorDescriptor(&pool1Tensor);
    cudnnCreateTensorDescriptor(&conv2Tensor);
    cudnnCreateTensorDescriptor(&conv2BiasTensor);
    cudnnCreateTensorDescriptor(&pool2Tensor);
    cudnnCreateTensorDescriptor(&fc1Tensor);
    cudnnCreateTensorDescriptor(&fc2Tensor);
    cudnnCreateActivationDescriptor(&fc1Activation);
    cudnnCreateFilterDescriptor(&conv1filterDesc);
    cudnnCreateFilterDescriptor(&conv2filterDesc);
    cudnnCreateConvolutionDescriptor(&conv1Desc);
    cudnnCreateConvolutionDescriptor(&conv2Desc);
    cudnnCreatePoolingDescriptor(&poolDesc);
    cudnnSetTensor4dDescriptor(conv1BiasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, conv1.out_channels, 1, 1);
    cudnnSetTensor4dDescriptor(conv2BiasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, conv2.out_channels, 1, 1);
    cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, pool1.size, pool1.size, 0, 0, pool1.stride, pool1.stride);
    cudnnSetTensor4dDescriptor(pool2Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, conv2.out_channels, conv2.out_height / pool2.stride, conv2.out_width / pool2.stride);
    cudnnSetTensor4dDescriptor(fc1Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, fc1.outputs, 1, 1);
    cudnnSetTensor4dDescriptor(fc2Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, fc2.outputs, 1, 1);
    cudnnSetActivationDescriptor(fc1Activation, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0);
    size_t workspace = 0;
    workspace = std::max(workspace, SetFwdConvolutionTensors(conv1, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo));
    workspace = std::max(workspace, SetBwdConvolutionTensors(dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, &conv1bwfalgo, NULL));
    workspace = std::max(workspace, SetFwdConvolutionTensors(conv2, pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, conv2algo));
    workspace = std::max(workspace, SetBwdConvolutionTensors(pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, &conv2bwfalgo, &conv2bwdalgo));
    m_workspaceSize = workspace;
  }

  ~TrainingContext() {
    cublasDestroy(cublasHandle);
    cudnnDestroy(cudnnHandle);
    cudnnDestroyTensorDescriptor(dataTensor);
    cudnnDestroyTensorDescriptor(conv1Tensor);
    cudnnDestroyTensorDescriptor(conv1BiasTensor);
    cudnnDestroyTensorDescriptor(pool1Tensor);
    cudnnDestroyTensorDescriptor(conv2Tensor);
    cudnnDestroyTensorDescriptor(conv2BiasTensor);
    cudnnDestroyTensorDescriptor(pool2Tensor);
    cudnnDestroyTensorDescriptor(fc1Tensor);
    cudnnDestroyTensorDescriptor(fc2Tensor);
    cudnnDestroyActivationDescriptor(fc1Activation);
    cudnnDestroyFilterDescriptor(conv1filterDesc);
    cudnnDestroyFilterDescriptor(conv2filterDesc);
    cudnnDestroyConvolutionDescriptor(conv1Desc);
    cudnnDestroyConvolutionDescriptor(conv2Desc);
    cudnnDestroyPoolingDescriptor(poolDesc);
  }

  size_t SetFwdConvolutionTensors(ConvBiasLayer& conv, cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                  cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc, cudnnConvolutionFwdAlgo_t& algo) {
    size_t sizeInBytes = 0;
    int n = m_batchSize;
    int c = conv.in_channels;
    int h = conv.in_height;
    int w = conv.in_width;
    cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, conv.out_channels, conv.in_channels, conv.kernel_size, conv.kernel_size);
    cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnGetConvolution2dForwardOutputDim(convDesc, srcTensorDesc, filterDesc, &n, &c, &h, &w);
    cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    cudnnGetConvolutionForwardAlgorithm(cudnnHandle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc, algo, &sizeInBytes);
    return sizeInBytes;
  }

  void ForwardPropagation(float *data, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu, float *fc2, float *result,
                          float *pconv1, float *pconv1bias, float *pconv2, float *pconv2bias, float *pfc1, float *pfc1bias, float *pfc2, float *pfc2bias, void *workspace, float *onevec) {
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor, data, conv1filterDesc, pconv1, conv1Desc, conv1algo, workspace, m_workspaceSize, &beta, conv1Tensor, conv1);
    cudnnAddTensor(cudnnHandle, &alpha, conv1BiasTensor, pconv1bias, &alpha, conv1Tensor, conv1);
    cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv1Tensor, conv1, &beta, pool1Tensor, pool1);
    cudnnConvolutionForward(cudnnHandle, &alpha, pool1Tensor, pool1, conv2filterDesc, pconv2, conv2Desc, conv2algo, workspace, m_workspaceSize, &beta, conv2Tensor, conv2);
    cudnnAddTensor(cudnnHandle, &alpha, conv2BiasTensor, pconv2bias, &alpha, conv2Tensor, conv2);
    cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv2Tensor, conv2, &beta, pool2Tensor, pool2);
    cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, ref_fc1.outputs, m_batchSize, ref_fc1.inputs, &alpha, pfc1, ref_fc1.inputs, pool2, ref_fc1.inputs, &beta, fc1, ref_fc1.outputs);
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc1.outputs, m_batchSize, 1, &alpha, pfc1bias, ref_fc1.outputs, onevec, 1, &alpha, fc1, ref_fc1.outputs);
    cudnnActivationForward(cudnnHandle, fc1Activation, &alpha, fc1Tensor, fc1, &beta, fc1Tensor, fc1relu);
    cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, ref_fc2.outputs, m_batchSize, ref_fc2.inputs, &alpha, pfc2, ref_fc2.inputs, fc1relu, ref_fc2.inputs, &beta, fc2, ref_fc2.outputs);
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc2.outputs, m_batchSize, 1, &alpha, pfc2bias, ref_fc2.outputs, onevec, 1, &alpha, fc2, ref_fc2.outputs);
    cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, fc2Tensor, fc2, &beta, fc2Tensor, result);
  }

  size_t SetBwdConvolutionTensors(cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                  cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc,
                                  cudnnConvolutionBwdFilterAlgo_t *falgo, cudnnConvolutionBwdDataAlgo_t *dalgo) {
    size_t sizeInBytes = 0, tmpsize = 0;
    if (falgo) {
      cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, falgo);
      cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc, *falgo, &tmpsize);
      sizeInBytes = std::max(sizeInBytes, tmpsize);
    }
    if (dalgo) {
      cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, dalgo);
      cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc, *dalgo, &tmpsize);
      sizeInBytes = std::max(sizeInBytes, tmpsize);
    }
    return sizeInBytes;
  }

  void Backpropagation(ConvBiasLayer& layer_conv1, MaxPoolLayer& layer_pool1, ConvBiasLayer& layer_conv2, MaxPoolLayer& layer_pool2,
                       float *data, float *labels, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu, float *fc2, float *fc2smax, float *dloss_data,
                       float *pconv1, float *pconv1bias, float *pconv2, float *pconv2bias, float *pfc1, float *pfc1bias, float *pfc2, float *pfc2bias,
                       float *gconv1, float *gconv1bias, float *dpool1, float *gconv2, float *gconv2bias, float *dconv2, float *dpool2,
                       float *gfc1, float *gfc1bias, float *dfc1, float *dfc1relu, float *gfc2, float *gfc2bias, float *dfc2, void *workspace, float *onevec) {
    float alpha = 1.0f, beta = 0.0f;
    float scalVal = 1.0f / static_cast<float>(m_batchSize);
    cudaMemcpyAsync(dloss_data, fc2smax, sizeof(float) * m_batchSize * ref_fc2.outputs, cudaMemcpyDeviceToDevice);
    SoftmaxLossBackprop<<<RoundUp(m_batchSize, BW), BW>>>(labels, ref_fc2.outputs, m_batchSize, dloss_data);
    cublasSscal(cublasHandle, ref_fc2.outputs * m_batchSize, &scalVal, dloss_data, 1);
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc2.inputs, ref_fc2.outputs, m_batchSize, &alpha, fc1relu, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, gfc2, ref_fc2.inputs);
    cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc2.outputs, m_batchSize, &alpha, dloss_data, ref_fc2.outputs, onevec, 1, &beta, gfc2bias, 1);
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc2.inputs, m_batchSize, ref_fc2.outputs, &alpha, pfc2, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, dfc2, ref_fc2.inputs);
    cudnnActivationBackward(cudnnHandle, fc1Activation, &alpha, fc1Tensor, fc1relu, fc1Tensor, dfc2, fc1Tensor, fc1, &beta, fc1Tensor, dfc1relu);
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc1.inputs, ref_fc1.outputs, m_batchSize, &alpha, pool2, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, gfc1, ref_fc1.inputs);
    cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc1.outputs, m_batchSize, &alpha, dfc1relu, ref_fc1.outputs, onevec, 1, &beta, gfc1bias, 1);
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc1.inputs, m_batchSize, ref_fc1.outputs, &alpha, pfc1, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, dfc1, ref_fc1.inputs);
    cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, pool2Tensor, pool2, pool2Tensor, dfc1, conv2Tensor, conv2, &beta, conv2Tensor, dpool2);
    cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv2Tensor, dpool2, &beta, conv2BiasTensor, gconv2bias);
    cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, pool1Tensor, pool1, conv2Tensor, dpool2, conv2Desc, conv2bwfalgo, workspace, m_workspaceSize, &beta, conv2filterDesc, gconv2);
    cudnnConvolutionBackwardData(cudnnHandle, &alpha, conv2filterDesc, pconv2, conv2Tensor, dpool2, conv2Desc, conv2bwdalgo, workspace, m_workspaceSize, &beta, pool1Tensor, dconv2);
    cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, pool1Tensor, pool1, pool1Tensor, dconv2, conv1Tensor, conv1, &beta, conv1Tensor, dpool1);
    cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv1Tensor, dpool1, &beta, conv1BiasTensor, gconv1bias);
    cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, dataTensor, data, conv1Tensor, dpool1, conv1Desc, conv1bwfalgo, workspace, m_workspaceSize, &beta, conv1filterDesc, gconv1);
  }

  void UpdateWeights(float learning_rate, ConvBiasLayer& conv1, ConvBiasLayer& conv2, float *pconv1, float *pconv1bias, float *pconv2, float *pconv2bias,
                     float *pfc1, float *pfc1bias, float *pfc2, float *pfc2bias, float *gconv1, float *gconv1bias, float *gconv2, float *gconv2bias, float *gfc1, float *gfc1bias, float *gfc2, float *gfc2bias) {
    float alpha = -learning_rate;
    cublasSaxpy(cublasHandle, static_cast<int>(conv1.pconv.size()), &alpha, gconv1, 1, pconv1, 1);
    cublasSaxpy(cublasHandle, static_cast<int>(conv1.pbias.size()), &alpha, gconv1bias, 1, pconv1bias, 1);
    cublasSaxpy(cublasHandle, static_cast<int>(conv2.pconv.size()), &alpha, gconv2, 1, pconv2, 1);
    cublasSaxpy(cublasHandle, static_cast<int>(conv2.pbias.size()), &alpha, gconv2bias, 1, pconv2bias, 1);
    cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pneurons.size()), &alpha, gfc1, 1, pfc1, 1);
    cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pbias.size()), &alpha, gfc1bias, 1, pfc1bias, 1);
    cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pneurons.size()), &alpha, gfc2, 1, pfc2, 1);
    cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pbias.size()), &alpha, gfc2bias, 1, pfc2bias, 1);
  }
};

int main(int argc, char **argv) {
  int iterations = 1000;
  int batch_size = 64;
  double learning_rate = 0.01;
  double lr_gamma = 0.0001;
  double lr_power = -0.75;
  int width, height;
  printf("Reading input data\n");
  std::vector<uint8_t> train_images, train_labels;
  readImages("train-images-idx3-ubyte", train_images, width, height);
  readLabels("train-labels-idx1-ubyte", train_labels);

  ConvBiasLayer conv1(1, 20, 5, (int)width, (int)height);
  MaxPoolLayer pool1(2, 2);
  ConvBiasLayer conv2(conv1.out_channels, 50, 5, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride);
  MaxPoolLayer pool2(2, 2);
  FullyConnectedLayer fc1((conv2.out_channels*conv2.out_width*conv2.out_height) / (pool2.stride * pool2.stride), 500);
  FullyConnectedLayer fc2(fc1.outputs, 10);
  TrainingContext context(batch_size, conv1, pool1, conv2, pool2, fc1, fc2);

  float wconv1 = sqrt(3.0f / (conv1.kernel_size * conv1.kernel_size * conv1.in_channels));
  float wconv2 = sqrt(3.0f / (conv2.kernel_size * conv2.kernel_size * conv2.in_channels));
  float wfc1 = sqrt(3.0f / (fc1.inputs * fc1.outputs));
  float wfc2 = sqrt(3.0f / (fc2.inputs * fc2.outputs));
  for (int i=0; i<conv1.pconv.size(); i++)
    conv1.pconv[i] = drand48() * wconv1 * 2 - wconv1;
  for (int i=0; i<conv1.pbias.size(); i++)
    conv1.pbias[i] = drand48() * wconv1 * 2 - wconv1;
  for (int i=0; i<conv2.pconv.size(); i++)
    conv2.pconv[i] = drand48() * wconv2 * 2 - wconv2;
  for (int i=0; i<conv2.pbias.size(); i++)
    conv2.pbias[i] = drand48() * wconv2 * 2 - wconv2;
  for (int i=0; i<fc1.pneurons.size(); i++)
    fc1.pneurons[i] = drand48() * wfc1 * 2 - wfc1;
  for (int i=0; i<fc1.pbias.size(); i++)
    fc1.pbias[i] = drand48() * wfc1 * 2 - wfc1;
  for (int i=0; i<fc2.pneurons.size(); i++)
    fc2.pneurons[i] = drand48() * wfc2 * 2 - wfc2;
  for (int i=0; i<fc2.pbias.size(); i++)
    fc2.pbias[i] = drand48() * wfc2 * 2 - wfc2;

  float *d_data, *d_labels, *d_conv1, *d_pool1, *d_conv2, *d_pool2, *d_fc1, *d_fc1relu, *d_fc2, *d_fc2smax;
  cudaMalloc(&d_data,    sizeof(float) * context.m_batchSize * 1                  * height                            * width);
  cudaMalloc(&d_labels,  sizeof(float) * context.m_batchSize * 1                  * 1                                 * 1);
  cudaMalloc(&d_conv1,   sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width);
  cudaMalloc(&d_pool1,   sizeof(float) * context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride));
  cudaMalloc(&d_conv2,   sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width);
  cudaMalloc(&d_pool2,   sizeof(float) * context.m_batchSize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride));
  cudaMalloc(&d_fc1,     sizeof(float) * context.m_batchSize * fc1.outputs);
  cudaMalloc(&d_fc1relu, sizeof(float) * context.m_batchSize * fc1.outputs);
  cudaMalloc(&d_fc2,     sizeof(float) * context.m_batchSize * fc2.outputs);
  cudaMalloc(&d_fc2smax, sizeof(float) * context.m_batchSize * fc2.outputs);

  float *d_pconv1, *d_pconv1bias, *d_pconv2, *d_pconv2bias;
  float *d_pfc1, *d_pfc1bias, *d_pfc2, *d_pfc2bias;
  cudaMalloc(&d_pconv1,     sizeof(float) * conv1.pconv.size());
  cudaMalloc(&d_pconv1bias, sizeof(float) * conv1.pbias.size());
  cudaMalloc(&d_pconv2,     sizeof(float) * conv2.pconv.size());
  cudaMalloc(&d_pconv2bias, sizeof(float) * conv2.pbias.size());
  cudaMalloc(&d_pfc1,       sizeof(float) * fc1.pneurons.size());
  cudaMalloc(&d_pfc1bias,   sizeof(float) * fc1.pbias.size());
  cudaMalloc(&d_pfc2,       sizeof(float) * fc2.pneurons.size());
  cudaMalloc(&d_pfc2bias,   sizeof(float) * fc2.pbias.size());

  float *d_gconv1, *d_gconv1bias, *d_gconv2, *d_gconv2bias;
  float *d_gfc1, *d_gfc1bias, *d_gfc2, *d_gfc2bias;
  cudaMalloc(&d_gconv1,     sizeof(float) * conv1.pconv.size());
  cudaMalloc(&d_gconv1bias, sizeof(float) * conv1.pbias.size());
  cudaMalloc(&d_gconv2,     sizeof(float) * conv2.pconv.size());
  cudaMalloc(&d_gconv2bias, sizeof(float) * conv2.pbias.size());
  cudaMalloc(&d_gfc1,       sizeof(float) * fc1.pneurons.size());
  cudaMalloc(&d_gfc1bias,   sizeof(float) * fc1.pbias.size());
  cudaMalloc(&d_gfc2,       sizeof(float) * fc2.pneurons.size());
  cudaMalloc(&d_gfc2bias,   sizeof(float) * fc2.pbias.size());

  float *d_dpool1, *d_dpool2, *d_dconv2, *d_dfc1, *d_dfc1relu, *d_dfc2, *d_dfc2smax, *d_dlossdata;
  cudaMalloc(&d_dpool1,   sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width);
  cudaMalloc(&d_dpool2,   sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width);
  cudaMalloc(&d_dconv2,   sizeof(float) * context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride));
  cudaMalloc(&d_dfc1,     sizeof(float) * context.m_batchSize * fc1.inputs);
  cudaMalloc(&d_dfc1relu, sizeof(float) * context.m_batchSize * fc1.outputs);
  cudaMalloc(&d_dfc2,     sizeof(float) * context.m_batchSize * fc2.inputs);
  cudaMalloc(&d_dfc2smax, sizeof(float) * context.m_batchSize * fc2.outputs);
  cudaMalloc(&d_dlossdata,sizeof(float) * context.m_batchSize * fc2.outputs);

  float *d_onevec;
  void *d_cudnn_workspace = NULL;
  cudaMalloc(&d_onevec, sizeof(float)* context.m_batchSize);
  cudaMalloc(&d_cudnn_workspace, context.m_workspaceSize);
  cudaMemcpyAsync(d_pconv1, &conv1.pconv[0],     sizeof(float) * conv1.pconv.size(),  cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pconv1bias, &conv1.pbias[0], sizeof(float) * conv1.pbias.size(),  cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pconv2, &conv2.pconv[0],     sizeof(float) * conv2.pconv.size(),  cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pconv2bias, &conv2.pbias[0], sizeof(float) * conv2.pbias.size(),  cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pfc1, &fc1.pneurons[0],      sizeof(float) * fc1.pneurons.size(), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pfc1bias, &fc1.pbias[0],     sizeof(float) * fc1.pbias.size(),    cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pfc2, &fc2.pneurons[0],      sizeof(float) * fc2.pneurons.size(), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_pfc2bias, &fc2.pbias[0],     sizeof(float) * fc2.pbias.size(),    cudaMemcpyHostToDevice);

  FillOnes<<<RoundUp(context.m_batchSize, BW), BW>>>(d_onevec, context.m_batchSize);

  printf("Preparing dataset\n");
  std::vector<float> train_images_float(train_images.size()), train_labels_float(train_labels.size());
  for (size_t i=0; i<train_images.size(); i++)
    train_images_float[i] = (float)train_images[i] / 255.0f;
  for (size_t i=0; i<train_labels.size(); i++)
    train_labels_float[i] = (float)train_labels[i];

  printf("Training...\n");
  cudaDeviceSynchronize();
  double t1 = get_time();
  for (int iter=0; iter<iterations; iter++) {
    int imageid = iter % (train_labels.size() / context.m_batchSize);
    cudaMemcpyAsync(d_data, &train_images_float[imageid * context.m_batchSize * width*height], sizeof(float) * context.m_batchSize * width * height, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_labels, &train_labels_float[imageid * context.m_batchSize], sizeof(float) * context.m_batchSize, cudaMemcpyHostToDevice);
    context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax, d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias,
                               d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias, d_cudnn_workspace, d_onevec);
    context.Backpropagation(conv1, pool1, conv2, pool2, d_data, d_labels, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax, d_dlossdata,
                            d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                            d_gconv1, d_gconv1bias, d_dpool1, d_gconv2, d_gconv2bias, d_dconv2, d_dpool2, d_gfc1, d_gfc1bias,
                            d_dfc1, d_dfc1relu, d_gfc2, d_gfc2bias, d_dfc2, d_cudnn_workspace, d_onevec);
    float learningRate = static_cast<float>(learning_rate * pow((1.0 + lr_gamma * iter), lr_power));
    context.UpdateWeights(learningRate, conv1, conv2,
                          d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                          d_gconv1, d_gconv1bias, d_gconv2, d_gconv2bias, d_gfc1, d_gfc1bias, d_gfc2, d_gfc2bias);
  }
  cudaDeviceSynchronize();
  double t2 = get_time();
  printf("Iteration time: %f ms\n", (t2 - t1) * 1000.0f / iterations);

  std::vector<uint8_t> test_images, test_labels;
  readImages("t10k-images-idx3-ubyte", test_images, width, height);
  readLabels("t10k-labels-idx1-ubyte", test_labels);
  float classification_error = 1.0f;
  int num_errors = 0;
  for (int i=0; i<test_labels.size(); i++) {
    std::vector<float> data(width * height);
    for (int j=0; j<width*height; j++)
      data[j] = (float)test_images[i*width*height+j] / 255.0f;
    cudaMemcpyAsync(d_data, &data[0], sizeof(float) * width * height, cudaMemcpyHostToDevice);
    context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax, d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias,
                               d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias, d_cudnn_workspace, d_onevec);
    std::vector<float> class_vec(10);
    cudaMemcpy(&class_vec[0], d_fc2smax, sizeof(float) * 10, cudaMemcpyDeviceToHost);
    int chosen = 0;
    for (int id=1; id<10; id++) {
      if (class_vec[chosen] < class_vec[id]) chosen = id;
    }
    if (chosen != test_labels[i])
      num_errors++;
  }
  classification_error = (float)num_errors / (float)test_labels.size();
  printf("Classification result: %.2f%% error (used %d images)\n", classification_error * 100.0f, (int)test_labels.size());

  cudaFree(d_data);
  cudaFree(d_conv1);
  cudaFree(d_pool1);
  cudaFree(d_conv2);
  cudaFree(d_pool2);
  cudaFree(d_fc1);
  cudaFree(d_fc2);
  cudaFree(d_pconv1);
  cudaFree(d_pconv1bias);
  cudaFree(d_pconv2);
  cudaFree(d_pconv2bias);
  cudaFree(d_pfc1);
  cudaFree(d_pfc1bias);
  cudaFree(d_pfc2);
  cudaFree(d_pfc2bias);
  cudaFree(d_gconv1);
  cudaFree(d_gconv1bias);
  cudaFree(d_gconv2);
  cudaFree(d_gconv2bias);
  cudaFree(d_gfc1);
  cudaFree(d_gfc1bias);
  cudaFree(d_dfc1);
  cudaFree(d_gfc2);
  cudaFree(d_gfc2bias);
  cudaFree(d_dfc2);
  cudaFree(d_dpool1);
  cudaFree(d_dconv2);
  cudaFree(d_dpool2);
  cudaFree(d_labels);
  cudaFree(d_dlossdata);
  cudaFree(d_onevec);
  if (d_cudnn_workspace != NULL)
    cudaFree(d_cudnn_workspace);
  return 0;
}
