#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

#define bswap(x) __builtin_bswap32(x)

struct UByteImageDataset {
  uint32_t magic;
  uint32_t length;
  uint32_t height;
  uint32_t width;
  void Swap() {
    magic = bswap(magic);
    length = bswap(length);
    height = bswap(height);
    width = bswap(width);
  }
};

struct UByteLabelDataset {
  uint32_t magic;
  uint32_t length;
  void Swap() {
    magic = bswap(magic);
    length = bswap(length);
  }
};

size_t ReadUByteDataset(const char *image_filename, const char *label_filename,
                        uint8_t *data, uint8_t *labels, size_t& width, size_t& height) {
  UByteImageDataset image_header;
  UByteLabelDataset label_header;
  FILE * imfp = fopen(image_filename, "r");
  FILE * lbfp = fopen(label_filename, "r");
  size_t size = fread(&image_header, sizeof(UByteImageDataset), 1, imfp);
  size = fread(&label_header, sizeof(UByteLabelDataset), 1, lbfp);
  image_header.Swap();
  label_header.Swap();
  width = image_header.width;
  height = image_header.height;
  if(data != NULL)
    size = fread(data, sizeof(uint8_t), image_header.length * width * height, imfp);
  if(labels != NULL)
    size = fread(labels, sizeof(uint8_t), label_header.length, lbfp);
  fclose(imfp);
  fclose(lbfp);
  return image_header.length;
}

#define BW 128

#define DEFINE_int32(flag, default_value, description) const int FLAGS_##flag = (default_value)
#define DEFINE_uint64(flag, default_value, description) const unsigned long long FLAGS_##flag = (default_value)
#define DEFINE_bool(flag, default_value, description) const bool FLAGS_##flag = (default_value)
#define DEFINE_double(flag, default_value, description) const double FLAGS_##flag = (default_value)
#define DEFINE_string(flag, default_value, description) const std::string FLAGS_##flag ((default_value))

static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator) {
  return (nominator + denominator - 1) / denominator;
}

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)


DEFINE_int32(gpu, 0, "The GPU ID to use");
DEFINE_int32(iterations, 1000, "Number of iterations for training");
DEFINE_int32(random_seed, -1, "Override random seed (default uses std::random_device)");
DEFINE_int32(classify, -1, "Number of images to classify to compute error rate (default uses entire test set)");
DEFINE_uint64(batch_size, 64, "Batch size for training");
DEFINE_string(train_images, "train-images-idx3-ubyte", "Training images filename");
DEFINE_string(train_labels, "train-labels-idx1-ubyte", "Training labels filename");
DEFINE_string(test_images, "t10k-images-idx3-ubyte", "Test images filename");
DEFINE_string(test_labels, "t10k-labels-idx1-ubyte", "Test labels filename");
DEFINE_double(learning_rate, 0.01, "Base learning rate");
DEFINE_double(lr_gamma, 0.0001, "Learning rate policy gamma");
DEFINE_double(lr_power, 0.75, "Learning rate policy power");

struct ConvBiasLayer {
  int in_channels, out_channels, kernel_size;
  int in_width, in_height, out_width, out_height;
  std::vector<float> pconv, pbias;
  ConvBiasLayer(int in_channels_, int out_channels_, int kernel_size_, int in_w_, int in_h_) :
    pconv(in_channels_ * kernel_size_ * kernel_size_ * out_channels_), pbias(out_channels_) {
    in_channels = in_channels_;
    out_channels = out_channels_;
    kernel_size = kernel_size_;
    in_width = in_w_;
    in_height = in_h_;
    out_width = in_w_ - kernel_size_ + 1;
    out_height = in_h_ - kernel_size_ + 1;
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

  int m_gpuid;
  int m_batchSize;
  size_t m_workspaceSize;
  FullyConnectedLayer& ref_fc1, &ref_fc2;
  TrainingContext& operator=(const TrainingContext&) = delete;
  TrainingContext(const TrainingContext&) = delete;
  TrainingContext(int gpuid, int batch_size, ConvBiasLayer& conv1, MaxPoolLayer& pool1, ConvBiasLayer& conv2, MaxPoolLayer& pool2,
                  FullyConnectedLayer& fc1, FullyConnectedLayer& fc2) : ref_fc1(fc1), ref_fc2(fc2), m_gpuid(gpuid) {
    m_batchSize = batch_size;
    checkCudaErrors(cudaSetDevice(gpuid));
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCUDNN(cudnnCreate(&cudnnHandle));
    checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&conv1Tensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&conv1BiasTensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&pool1Tensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&conv2Tensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&conv2BiasTensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&pool2Tensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&fc1Tensor));
    checkCUDNN(cudnnCreateTensorDescriptor(&fc2Tensor));
    checkCUDNN(cudnnCreateActivationDescriptor(&fc1Activation));
    checkCUDNN(cudnnCreateFilterDescriptor(&conv1filterDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&conv2filterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1Desc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv2Desc));
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(conv1BiasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, conv1.out_channels, 1, 1));
    checkCUDNN(cudnnSetTensor4dDescriptor(conv2BiasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, conv2.out_channels, 1, 1));
    checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, pool1.size, pool1.size, 0, 0, pool1.stride, pool1.stride));
    checkCUDNN(cudnnSetTensor4dDescriptor(pool2Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, conv2.out_channels, conv2.out_height / pool2.stride, conv2.out_width / pool2.stride));
    checkCUDNN(cudnnSetTensor4dDescriptor(fc1Tensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          batch_size, fc1.outputs, 1, 1));
    checkCUDNN(cudnnSetTensor4dDescriptor(fc2Tensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          batch_size, fc2.outputs, 1, 1));
    checkCUDNN(cudnnSetActivationDescriptor(fc1Activation, CUDNN_ACTIVATION_RELU,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    size_t workspace = 0;
    workspace = std::max(workspace, SetFwdConvolutionTensors(conv1, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo));
    workspace = std::max(workspace, SetBwdConvolutionTensors(dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, &conv1bwfalgo, nullptr));
    workspace = std::max(workspace, SetFwdConvolutionTensors(conv2, pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, conv2algo));
    workspace = std::max(workspace, SetBwdConvolutionTensors(pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, &conv2bwfalgo, &conv2bwdalgo));
    m_workspaceSize = workspace;
  }

  ~TrainingContext() {
    checkCudaErrors(cudaSetDevice(m_gpuid));
    checkCudaErrors(cublasDestroy(cublasHandle));
    checkCUDNN(cudnnDestroy(cudnnHandle));
    checkCUDNN(cudnnDestroyTensorDescriptor(dataTensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(conv1Tensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(conv1BiasTensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(pool1Tensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(conv2Tensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(conv2BiasTensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(pool2Tensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(fc1Tensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(fc2Tensor));
    checkCUDNN(cudnnDestroyActivationDescriptor(fc1Activation));
    checkCUDNN(cudnnDestroyFilterDescriptor(conv1filterDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(conv2filterDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv1Desc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv2Desc));
    checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
  }

  size_t SetFwdConvolutionTensors(ConvBiasLayer& conv, cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                  cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc,
                                  cudnnConvolutionFwdAlgo_t& algo) {
    size_t sizeInBytes = 0;
    int n = m_batchSize;
    int c = conv.in_channels;
    int h = conv.in_height;
    int w = conv.in_width;
    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, conv.out_channels, conv.in_channels, conv.kernel_size, conv.kernel_size));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, srcTensorDesc, filterDesc, &n, &c, &h, &w));
    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc, algo, &sizeInBytes));
    return sizeInBytes;
  }

  void ForwardPropagation(float *data, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
                          float *fc2, float *result,
                          float *pconv1, float *pconv1bias,
                          float *pconv2, float *pconv2bias,
                          float *pfc1, float *pfc1bias,
                          float *pfc2, float *pfc2bias, void *workspace, float *onevec) {
    float alpha = 1.0f, beta = 0.0f;
    checkCudaErrors(cudaSetDevice(m_gpuid));
    checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor,
                                       data, conv1filterDesc, pconv1, conv1Desc,
                                       conv1algo, workspace, m_workspaceSize, &beta,
                                       conv1Tensor, conv1));
    checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv1BiasTensor,
                              pconv1bias, &alpha, conv1Tensor, conv1));
    checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv1Tensor,
                                   conv1, &beta, pool1Tensor, pool1));
    checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, pool1Tensor,
                                       pool1, conv2filterDesc, pconv2, conv2Desc,
                                       conv2algo, workspace, m_workspaceSize, &beta,
                                       conv2Tensor, conv2));
    checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv2BiasTensor,
                              pconv2bias, &alpha, conv2Tensor, conv2));
    checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv2Tensor,
                                   conv2, &beta, pool2Tensor, pool2));
    checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                ref_fc1.outputs, m_batchSize, ref_fc1.inputs,
                                &alpha,
                                pfc1, ref_fc1.inputs,
                                pool2, ref_fc1.inputs,
                                &beta,
                                fc1, ref_fc1.outputs));
    checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                ref_fc1.outputs, m_batchSize, 1,
                                &alpha,
                                pfc1bias, ref_fc1.outputs,
                                onevec, 1,
                                &alpha,
                                fc1, ref_fc1.outputs));
    checkCUDNN(cudnnActivationForward(cudnnHandle, fc1Activation, &alpha,
                                      fc1Tensor, fc1, &beta, fc1Tensor, fc1relu));
    checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                ref_fc2.outputs, m_batchSize, ref_fc2.inputs,
                                &alpha,
                                pfc2, ref_fc2.inputs,
                                fc1relu, ref_fc2.inputs,
                                &beta,
                                fc2, ref_fc2.outputs));
    checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                ref_fc2.outputs, m_batchSize, 1,
                                &alpha,
                                pfc2bias, ref_fc2.outputs,
                                onevec, 1,
                                &alpha,
                                fc2, ref_fc2.outputs));
    checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                   &alpha, fc2Tensor, fc2, &beta, fc2Tensor, result));
  }

  size_t SetBwdConvolutionTensors(cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                  cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc,
                                  cudnnConvolutionBwdFilterAlgo_t *falgo, cudnnConvolutionBwdDataAlgo_t *dalgo) {
    size_t sizeInBytes = 0, tmpsize = 0;
    if (falgo) {
      checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                                                            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, falgo));
      checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                                                                *falgo, &tmpsize));
      sizeInBytes = std::max(sizeInBytes, tmpsize);
    }
    if (dalgo) {
      checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
                                                          CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, dalgo));
      checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
                                                              *dalgo, &tmpsize));
      sizeInBytes = std::max(sizeInBytes, tmpsize);
    }
    return sizeInBytes;
  }

  void Backpropagation(ConvBiasLayer& layer_conv1, MaxPoolLayer& layer_pool1, ConvBiasLayer& layer_conv2, MaxPoolLayer& layer_pool2,
                       float *data, float *labels, float *conv1, float *pool1, float *conv2, float *pool2, float *fc1, float *fc1relu,
                       float *fc2, float *fc2smax, float *dloss_data,
                       float *pconv1, float *pconv1bias,
                       float *pconv2, float *pconv2bias,
                       float *pfc1, float *pfc1bias,
                       float *pfc2, float *pfc2bias,
                       float *gconv1, float *gconv1bias, float *dpool1,
                       float *gconv2, float *gconv2bias, float *dconv2, float *dpool2,
                       float *gfc1, float *gfc1bias, float *dfc1, float *dfc1relu,
                       float *gfc2, float *gfc2bias, float *dfc2,
                       void *workspace, float *onevec) {
    float alpha = 1.0f, beta = 0.0f;
    float scalVal = 1.0f / static_cast<float>(m_batchSize);
    checkCudaErrors(cudaSetDevice(m_gpuid));
    checkCudaErrors(cudaMemcpyAsync(dloss_data, fc2smax, sizeof(float) * m_batchSize * ref_fc2.outputs, cudaMemcpyDeviceToDevice));
    SoftmaxLossBackprop<<<RoundUp(m_batchSize, BW), BW>>>(labels, ref_fc2.outputs, m_batchSize, dloss_data);
    checkCudaErrors(cublasSscal(cublasHandle, ref_fc2.outputs * m_batchSize, &scalVal, dloss_data, 1));
    checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc2.inputs, ref_fc2.outputs, m_batchSize,
                                &alpha, fc1relu, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, gfc2, ref_fc2.inputs));
    checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc2.outputs, m_batchSize,
                                &alpha, dloss_data, ref_fc2.outputs, onevec, 1, &beta, gfc2bias, 1));
    checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc2.inputs, m_batchSize, ref_fc2.outputs,
                                &alpha, pfc2, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, dfc2, ref_fc2.inputs));
    checkCUDNN(cudnnActivationBackward(cudnnHandle, fc1Activation, &alpha,
                                       fc1Tensor, fc1relu, fc1Tensor, dfc2,
                                       fc1Tensor, fc1, &beta, fc1Tensor, dfc1relu));
    checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc1.inputs, ref_fc1.outputs, m_batchSize,
                                &alpha, pool2, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, gfc1, ref_fc1.inputs));
    checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc1.outputs, m_batchSize,
                                &alpha, dfc1relu, ref_fc1.outputs, onevec, 1, &beta, gfc1bias, 1));
    checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc1.inputs, m_batchSize, ref_fc1.outputs,
                                &alpha, pfc1, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, dfc1, ref_fc1.inputs));
    checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha,
                                    pool2Tensor, pool2, pool2Tensor, dfc1,
                                    conv2Tensor, conv2, &beta, conv2Tensor, dpool2));
    checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv2Tensor,
                                            dpool2, &beta, conv2BiasTensor, gconv2bias));
    checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, pool1Tensor,
                                              pool1, conv2Tensor, dpool2, conv2Desc,
                                              conv2bwfalgo, workspace, m_workspaceSize,
                                              &beta, conv2filterDesc, gconv2));
    checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha, conv2filterDesc,
                                            pconv2, conv2Tensor, dpool2, conv2Desc,
                                            conv2bwdalgo, workspace, m_workspaceSize,
                                            &beta, pool1Tensor, dconv2));
    checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha,
                                    pool1Tensor, pool1, pool1Tensor, dconv2,
                                    conv1Tensor, conv1, &beta, conv1Tensor, dpool1));
    checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv1Tensor,
                                            dpool1, &beta, conv1BiasTensor, gconv1bias));
    checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, dataTensor,
                                              data, conv1Tensor, dpool1, conv1Desc,
                                              conv1bwfalgo, workspace, m_workspaceSize,
                                              &beta, conv1filterDesc, gconv1));
  }

  void UpdateWeights(float learning_rate,
                     ConvBiasLayer& conv1, ConvBiasLayer& conv2,
                     float *pconv1, float *pconv1bias,
                     float *pconv2, float *pconv2bias,
                     float *pfc1, float *pfc1bias,
                     float *pfc2, float *pfc2bias,
                     float *gconv1, float *gconv1bias,
                     float *gconv2, float *gconv2bias,
                     float *gfc1, float *gfc1bias,
                     float *gfc2, float *gfc2bias) {
    float alpha = -learning_rate;
    checkCudaErrors(cudaSetDevice(m_gpuid));
    checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pconv.size()),
                                &alpha, gconv1, 1, pconv1, 1));
    checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pbias.size()),
                                &alpha, gconv1bias, 1, pconv1bias, 1));
    checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pconv.size()),
                                &alpha, gconv2, 1, pconv2, 1));
    checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pbias.size()),
                                &alpha, gconv2bias, 1, pconv2bias, 1));
    checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pneurons.size()),
                                &alpha, gfc1, 1, pfc1, 1));
    checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pbias.size()),
                                &alpha, gfc1bias, 1, pfc1bias, 1));
    checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pneurons.size()),
                                &alpha, gfc2, 1, pfc2, 1));
    checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pbias.size()),
                                &alpha, gfc2bias, 1, pfc2bias, 1));
  }
};

int main(int argc, char **argv) {
  size_t width, height, channels = 1;
  printf("Reading input data\n");
  size_t train_size = ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), NULL, NULL, width, height);
  size_t test_size = ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), NULL, NULL, width, height);
  std::vector<uint8_t> train_images(train_size * width * height * channels), train_labels(train_size);
  std::vector<uint8_t> test_images(test_size * width * height * channels), test_labels(test_size);
  size_t size = ReadUByteDataset(FLAGS_train_images.c_str(), FLAGS_train_labels.c_str(), &train_images[0], &train_labels[0], width, height);
  size = ReadUByteDataset(FLAGS_test_images.c_str(), FLAGS_test_labels.c_str(), &test_images[0], &test_labels[0], width, height);
  printf("Done. Training dataset size: %d, Test dataset size: %d\n", (int)train_size, (int)test_size);
  printf("Batch size: %lld, iterations: %d\n", FLAGS_batch_size, FLAGS_iterations);

  ConvBiasLayer conv1((int)channels, 20, 5, (int)width, (int)height);
  MaxPoolLayer pool1(2, 2);
  ConvBiasLayer conv2(conv1.out_channels, 50, 5, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride);
  MaxPoolLayer pool2(2, 2);
  FullyConnectedLayer fc1((conv2.out_channels*conv2.out_width*conv2.out_height) / (pool2.stride * pool2.stride), 500);
  FullyConnectedLayer fc2(fc1.outputs, 10);
  TrainingContext context(FLAGS_gpu, FLAGS_batch_size, conv1, pool1, conv2, pool2, fc1, fc2);

  std::random_device rd;
  std::mt19937 gen(FLAGS_random_seed < 0 ? rd() : static_cast<unsigned int>(FLAGS_random_seed));

  float wconv1 = sqrt(3.0f / (conv1.kernel_size * conv1.kernel_size * conv1.in_channels));
  std::uniform_real_distribution<> dconv1(-wconv1, wconv1);
  float wconv2 = sqrt(3.0f / (conv2.kernel_size * conv2.kernel_size * conv2.in_channels));
  std::uniform_real_distribution<> dconv2(-wconv2, wconv2);
  float wfc1 = sqrt(3.0f / (fc1.inputs * fc1.outputs));
  std::uniform_real_distribution<> dfc1(-wfc1, wfc1);
  float wfc2 = sqrt(3.0f / (fc2.inputs * fc2.outputs));
  std::uniform_real_distribution<> dfc2(-wfc2, wfc2);

  for (auto&& iter : conv1.pconv)
    iter = static_cast<float>(dconv1(gen));
  for (auto&& iter : conv1.pbias)
    iter = static_cast<float>(dconv1(gen));
  for (auto&& iter : conv2.pconv)
    iter = static_cast<float>(dconv2(gen));
  for (auto&& iter : conv2.pbias)
    iter = static_cast<float>(dconv2(gen));
  for (auto&& iter : fc1.pneurons)
    iter = static_cast<float>(dfc1(gen));
  for (auto&& iter : fc1.pbias)
    iter = static_cast<float>(dfc1(gen));
  for (auto&& iter : fc2.pneurons)
    iter = static_cast<float>(dfc2(gen));
  for (auto&& iter : fc2.pbias)
    iter = static_cast<float>(dfc2(gen));

  float *d_data, *d_labels, *d_conv1, *d_pool1, *d_conv2, *d_pool2, *d_fc1, *d_fc1relu, *d_fc2, *d_fc2smax;
  checkCudaErrors(cudaMalloc(&d_data,    sizeof(float) * context.m_batchSize * channels           * height                            * width));
  checkCudaErrors(cudaMalloc(&d_labels,  sizeof(float) * context.m_batchSize * 1                  * 1                                 * 1));
  checkCudaErrors(cudaMalloc(&d_conv1,   sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
  checkCudaErrors(cudaMalloc(&d_pool1,   sizeof(float) * context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
  checkCudaErrors(cudaMalloc(&d_conv2,   sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
  checkCudaErrors(cudaMalloc(&d_pool2,   sizeof(float) * context.m_batchSize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride)));
  checkCudaErrors(cudaMalloc(&d_fc1,     sizeof(float) * context.m_batchSize * fc1.outputs));
  checkCudaErrors(cudaMalloc(&d_fc1relu, sizeof(float) * context.m_batchSize * fc1.outputs));
  checkCudaErrors(cudaMalloc(&d_fc2,     sizeof(float) * context.m_batchSize * fc2.outputs));
  checkCudaErrors(cudaMalloc(&d_fc2smax, sizeof(float) * context.m_batchSize * fc2.outputs));

  float *d_pconv1, *d_pconv1bias, *d_pconv2, *d_pconv2bias;
  float *d_pfc1, *d_pfc1bias, *d_pfc2, *d_pfc2bias;
  checkCudaErrors(cudaMalloc(&d_pconv1,     sizeof(float) * conv1.pconv.size()));
  checkCudaErrors(cudaMalloc(&d_pconv1bias, sizeof(float) * conv1.pbias.size()));
  checkCudaErrors(cudaMalloc(&d_pconv2,     sizeof(float) * conv2.pconv.size()));
  checkCudaErrors(cudaMalloc(&d_pconv2bias, sizeof(float) * conv2.pbias.size()));
  checkCudaErrors(cudaMalloc(&d_pfc1,       sizeof(float) * fc1.pneurons.size()));
  checkCudaErrors(cudaMalloc(&d_pfc1bias,   sizeof(float) * fc1.pbias.size()));
  checkCudaErrors(cudaMalloc(&d_pfc2,       sizeof(float) * fc2.pneurons.size()));
  checkCudaErrors(cudaMalloc(&d_pfc2bias,   sizeof(float) * fc2.pbias.size()));

  float *d_gconv1, *d_gconv1bias, *d_gconv2, *d_gconv2bias;
  float *d_gfc1, *d_gfc1bias, *d_gfc2, *d_gfc2bias;
  checkCudaErrors(cudaMalloc(&d_gconv1,     sizeof(float) * conv1.pconv.size()));
  checkCudaErrors(cudaMalloc(&d_gconv1bias, sizeof(float) * conv1.pbias.size()));
  checkCudaErrors(cudaMalloc(&d_gconv2,     sizeof(float) * conv2.pconv.size()));
  checkCudaErrors(cudaMalloc(&d_gconv2bias, sizeof(float) * conv2.pbias.size()));
  checkCudaErrors(cudaMalloc(&d_gfc1,       sizeof(float) * fc1.pneurons.size()));
  checkCudaErrors(cudaMalloc(&d_gfc1bias,   sizeof(float) * fc1.pbias.size()));
  checkCudaErrors(cudaMalloc(&d_gfc2,       sizeof(float) * fc2.pneurons.size()));
  checkCudaErrors(cudaMalloc(&d_gfc2bias,   sizeof(float) * fc2.pbias.size()));

  float *d_dpool1, *d_dpool2, *d_dconv2, *d_dfc1, *d_dfc1relu, *d_dfc2, *d_dfc2smax, *d_dlossdata;
  checkCudaErrors(cudaMalloc(&d_dpool1,   sizeof(float) * context.m_batchSize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
  checkCudaErrors(cudaMalloc(&d_dpool2,   sizeof(float) * context.m_batchSize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
  checkCudaErrors(cudaMalloc(&d_dconv2,   sizeof(float) * context.m_batchSize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
  checkCudaErrors(cudaMalloc(&d_dfc1,     sizeof(float) * context.m_batchSize * fc1.inputs));
  checkCudaErrors(cudaMalloc(&d_dfc1relu, sizeof(float) * context.m_batchSize * fc1.outputs));
  checkCudaErrors(cudaMalloc(&d_dfc2,     sizeof(float) * context.m_batchSize * fc2.inputs));
  checkCudaErrors(cudaMalloc(&d_dfc2smax, sizeof(float) * context.m_batchSize * fc2.outputs));
  checkCudaErrors(cudaMalloc(&d_dlossdata,sizeof(float) * context.m_batchSize * fc2.outputs));

  float *d_onevec;
  void *d_cudnn_workspace = nullptr;
  checkCudaErrors(cudaMalloc(&d_onevec, sizeof(float)* context.m_batchSize));
  if (context.m_workspaceSize > 0)
    checkCudaErrors(cudaMalloc(&d_cudnn_workspace, context.m_workspaceSize));

  checkCudaErrors(cudaMemcpyAsync(d_pconv1, &conv1.pconv[0],     sizeof(float) * conv1.pconv.size(),  cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyAsync(d_pconv1bias, &conv1.pbias[0], sizeof(float) * conv1.pbias.size(),  cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyAsync(d_pconv2, &conv2.pconv[0],     sizeof(float) * conv2.pconv.size(),  cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyAsync(d_pconv2bias, &conv2.pbias[0], sizeof(float) * conv2.pbias.size(),  cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyAsync(d_pfc1, &fc1.pneurons[0],      sizeof(float) * fc1.pneurons.size(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyAsync(d_pfc1bias, &fc1.pbias[0],     sizeof(float) * fc1.pbias.size(),    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyAsync(d_pfc2, &fc2.pneurons[0],      sizeof(float) * fc2.pneurons.size(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyAsync(d_pfc2bias, &fc2.pbias[0],     sizeof(float) * fc2.pbias.size(),    cudaMemcpyHostToDevice));

  FillOnes<<<RoundUp(context.m_batchSize, BW), BW>>>(d_onevec, context.m_batchSize);

  printf("Preparing dataset\n");
  std::vector<float> train_images_float(train_images.size()), train_labels_float(train_size);
  for (size_t i = 0; i < train_size * channels * width * height; ++i)
    train_images_float[i] = (float)train_images[i] / 255.0f;
  for (size_t i = 0; i < train_size; ++i)
    train_labels_float[i] = (float)train_labels[i];

  printf("Training...\n");
  checkCudaErrors(cudaDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < FLAGS_iterations; ++iter) {
    int imageid = iter % (train_size / context.m_batchSize);
    checkCudaErrors(cudaMemcpyAsync(d_data, &train_images_float[imageid * context.m_batchSize * width*height*channels],
                                    sizeof(float) * context.m_batchSize * channels * width * height, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_labels, &train_labels_float[imageid * context.m_batchSize],
                                    sizeof(float) * context.m_batchSize, cudaMemcpyHostToDevice));
    context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax,
                               d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                               d_cudnn_workspace, d_onevec);
    context.Backpropagation(conv1, pool1, conv2, pool2,
                            d_data, d_labels, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax, d_dlossdata,
                            d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                            d_gconv1, d_gconv1bias, d_dpool1, d_gconv2, d_gconv2bias, d_dconv2, d_dpool2, d_gfc1, d_gfc1bias,
                            d_dfc1, d_dfc1relu, d_gfc2, d_gfc2bias, d_dfc2, d_cudnn_workspace, d_onevec);
    float learningRate = static_cast<float>(FLAGS_learning_rate * pow((1.0 + FLAGS_lr_gamma * iter), (-FLAGS_lr_power)));
    context.UpdateWeights(learningRate, conv1, conv2,
                          d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias,
                          d_gconv1, d_gconv1bias, d_gconv2, d_gconv2bias, d_gfc1, d_gfc1bias, d_gfc2, d_gfc2bias);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  auto t2 = std::chrono::high_resolution_clock::now();
  printf("Iteration time: %f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / FLAGS_iterations);

  float classification_error = 1.0f;
  int classifications = FLAGS_classify;
  if (classifications < 0)
    classifications = (int)test_size;
  if (classifications > 0) {
    TrainingContext test_context(FLAGS_gpu, 1, conv1, pool1, conv2, pool2, fc1, fc2);
    if (context.m_workspaceSize < test_context.m_workspaceSize) {
      checkCudaErrors(cudaFree(d_cudnn_workspace));
      checkCudaErrors(cudaMalloc(&d_cudnn_workspace, test_context.m_workspaceSize));
    }
    int num_errors = 0;
    for (int i = 0; i < classifications; ++i) {
      std::vector<float> data(width * height);
      for (int j = 0; j < width * height; ++j)
        data[j] = (float)test_images[i * width*height*channels + j] / 255.0f;
      checkCudaErrors(cudaMemcpyAsync(d_data, &data[0], sizeof(float) * width * height, cudaMemcpyHostToDevice));
      test_context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_fc1, d_fc1relu, d_fc2, d_fc2smax,
                                      d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_pfc1, d_pfc1bias,
                                      d_pfc2, d_pfc2bias, d_cudnn_workspace, d_onevec);
      std::vector<float> class_vec(10);
      checkCudaErrors(cudaMemcpy(&class_vec[0], d_fc2smax, sizeof(float) * 10, cudaMemcpyDeviceToHost));
      int chosen = 0;
      for (int id = 1; id < 10; ++id) {
        if (class_vec[chosen] < class_vec[id]) chosen = id;
      }
      if (chosen != test_labels[i])
        ++num_errors;
    }
    classification_error = (float)num_errors / (float)classifications;
    printf("Classification result: %.2f%% error (used %d images)\n", classification_error * 100.0f, (int)classifications);
  }

  checkCudaErrors(cudaFree(d_data));
  checkCudaErrors(cudaFree(d_conv1));
  checkCudaErrors(cudaFree(d_pool1));
  checkCudaErrors(cudaFree(d_conv2));
  checkCudaErrors(cudaFree(d_pool2));
  checkCudaErrors(cudaFree(d_fc1));
  checkCudaErrors(cudaFree(d_fc2));
  checkCudaErrors(cudaFree(d_pconv1));
  checkCudaErrors(cudaFree(d_pconv1bias));
  checkCudaErrors(cudaFree(d_pconv2));
  checkCudaErrors(cudaFree(d_pconv2bias));
  checkCudaErrors(cudaFree(d_pfc1));
  checkCudaErrors(cudaFree(d_pfc1bias));
  checkCudaErrors(cudaFree(d_pfc2));
  checkCudaErrors(cudaFree(d_pfc2bias));
  checkCudaErrors(cudaFree(d_gconv1));
  checkCudaErrors(cudaFree(d_gconv1bias));
  checkCudaErrors(cudaFree(d_gconv2));
  checkCudaErrors(cudaFree(d_gconv2bias));
  checkCudaErrors(cudaFree(d_gfc1));
  checkCudaErrors(cudaFree(d_gfc1bias));
  checkCudaErrors(cudaFree(d_dfc1));
  checkCudaErrors(cudaFree(d_gfc2));
  checkCudaErrors(cudaFree(d_gfc2bias));
  checkCudaErrors(cudaFree(d_dfc2));
  checkCudaErrors(cudaFree(d_dpool1));
  checkCudaErrors(cudaFree(d_dconv2));
  checkCudaErrors(cudaFree(d_dpool2));
  checkCudaErrors(cudaFree(d_labels));
  checkCudaErrors(cudaFree(d_dlossdata));
  checkCudaErrors(cudaFree(d_onevec));
  if (d_cudnn_workspace != nullptr)
    checkCudaErrors(cudaFree(d_cudnn_workspace));
  return 0;
}
