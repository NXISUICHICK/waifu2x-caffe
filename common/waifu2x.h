#pragma once

#include <stdint.h>
#include <string>
#include <vector>
#include <utility>
#include <functional>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <opencv2/core.hpp>

#define CUDNN_DLL_NAME "cudnn64_8.dll"
#define CUDNN_REQUIRE_VERION_TEXT "v8.0.3"
#define CUDNN_REQUIRE_VERION 8003


namespace caffe
{
	template <typename Dtype>
	class Net;
	class NetParameter;
};

class cNet;
class stImage;


class Factor
{
private:
	double mNumerator; // 分子
	double mDenominator; // 分母

public:
	Factor() : mNumerator(1.0), mDenominator(1.0)
	{
	}

	Factor(const double numerator, const double denominator) : mNumerator(numerator), mDenominator(denominator)
	{
	}

	Factor MultiNumerator(const double numerator) const
	{
		return Factor(mNumerator * numerator, mDenominator);
	}

	Factor MultiDenominator(const double denominator) const
	{
		return Factor(mNumerator, mDenominator * denominator);
	}

	double toDouble() const
	{
		return mNumerator / mDenominator;
	}
};

class Waifu2x
{
public:
	struct stInfo
	{
		struct stParam
		{
			int scale_factor;
			int offset;
			int recommended_crop_size;

			stParam() : scale_factor(1), offset(0), recommended_crop_size(-1) {}
		};

		std::string name;
		std::string arch_name;
		bool has_noise_scale;
		bool has_noise_only;
		int channels;
		int force_divisible_crop_size;

		stParam noise;
		stParam scale;
		stParam noise_scale;
	};

	enum eWaifu2xModelType
	{
		eWaifu2xModelTypeNoise = 0,
		eWaifu2xModelTypeScale = 1,
		eWaifu2xModelTypeNoiseScale = 2,
		eWaifu2xModelTypeAutoScale = 3,
	};

	enum eWaifu2xError
	{
		eWaifu2xError_OK = 0,
		eWaifu2xError_Cancel,
		eWaifu2xError_NotInitialized,
		eWaifu2xError_InvalidParameter,
		eWaifu2xError_FailedOpenInputFile,
		eWaifu2xError_FailedOpenOutputFile,
		eWaifu2xError_FailedOpenModelFile,
		eWaifu2xError_FailedParseModelFile,
		eWaifu2xError_FailedWriteModelFile,
		eWaifu2xError_FailedConstructModel,
		eWaifu2xError_FailedProcessCaffe,
		eWaifu2xError_FailedCudaCheck,
		eWaifu2xError_FailedVulkanCheck,
		eWaifu2xError_FailedUnknownType,
	};

	enum eWaifu2xCudaError
	{
		eWaifu2xCudaError_OK = 0,
		eWaifu2xCudaError_NotFind,
		eWaifu2xCudaError_OldVersion,
		eWaifu2xCudaError_OldDevice,
	};

	enum eWaifu2xcuDNNError
	{
		eWaifu2xcuDNNError_OK = 0,
		eWaifu2xcuDNNError_NotFind,
		eWaifu2xcuDNNError_OldVersion,
		eWaifu2xcuDNNError_CannotCreate,
		eWaifu2xcuDNNError_OldCudaVersion,
	};

	// Basic Vulkan availability enum
	enum eWaifu2xVulkanError
	{
		eWaifu2xVulkanError_OK = 0,
		eWaifu2xVulkanError_NotFind,
		eWaifu2xVulkanError_Unsupported,
		eWaifu2xVulkanError_CannotCreate,
	};

	typedef std::function<bool()> waifu2xCancelFunc;

	static std::string ExeDir;

private:
	bool mIsInited;

	eWaifu2xModelType mMode;
	int mNoiseLevel;
	std::string mProcess;
	int mGPUNo;

	bool mIsCuda;

	std::shared_ptr<cNet> mNoiseNet;
	std::shared_ptr<cNet> mScaleNet;

	int mInputPlane; // ネットワークが想定するチャンネル数
	int mMaxNetOffset; // ネットワークが必要とするパディング
	bool mHasNoiseScaleOnly;

	float *mOutputBlock;
	size_t mOutputBlockSize;

private:
	static boost::filesystem::path GetModeDirPath(const boost::filesystem::path &model_dir);
	static boost::filesystem::path GetInfoPath(const boost::filesystem::path &model_dir);

	static Factor CalcScaleRatio(const boost::optional<double> scale_ratio, const boost::optional<int> scale_width, const boost::optional<int> scale_height,
		const stImage &image);

	static int GetcuDNNAlgorithm(const char *layer_name, int num_input, int num_output, int batch_size,
		int width, int height, int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h);

	static void SetcuDNNAlgorithm(int algo, const char *layer_name, int num_input, int num_output, int batch_size,
		int width, int height, int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h);

	Waifu2x::eWaifu2xError ReconstructImage(const Factor factor, const int crop_w, const int crop_h, const bool use_tta, const int batch_size,
		const bool isReconstructNoise, const bool isReconstructScale, const Waifu2x::waifu2xCancelFunc cancel_func, stImage &image);
	Waifu2x::eWaifu2xError ReconstructScale(const int crop_w, const int crop_h, const bool use_tta, const int batch_size,
		const Waifu2x::waifu2xCancelFunc cancel_func, stImage &image);
	Waifu2x::eWaifu2xError ReconstructNoiseScale(const int crop_w, const int crop_h, const bool use_tta, const int batch_size,
		const Waifu2x::waifu2xCancelFunc cancel_func, stImage &image);
	Waifu2x::eWaifu2xError ReconstructByNet(std::shared_ptr<cNet> net, const int crop_w, const int crop_h, const bool use_tta, const int batch_size,
		const Waifu2x::waifu2xCancelFunc cancel_func, cv::Mat &im);
	Waifu2x::eWaifu2xError ProcessNet(std::shared_ptr<cNet> net, const int crop_w, const int crop_h, const bool use_tta, const int batch_size, cv::Mat &im);

public:
	Waifu2x();
	~Waifu2x();

	static eWaifu2xCudaError can_use_CUDA();
	static eWaifu2xcuDNNError can_use_cuDNN();
	static eWaifu2xVulkanError can_use_Vulkan();

	static void init_liblary(int argc, char** argv);
	static void quit_liblary();
	static void quit_thread_liblary();

	// mode: noise or scale or noise_scale or auto_scale
	// process: cpu or gpu or cudnn or vulkan
	eWaifu2xError Init(const eWaifu2xModelType mode, const int noise_level,
		const boost::filesystem::path &model_dir, const std::string &process, const int gpu_no = 0);

	eWaifu2xError waifu2x(const boost::filesystem::path &input_file, const boost::filesystem::path &output_file,
		const boost::optional<double> scale_ratio, const boost::optional<int> scale_width, const boost::optional<int> scale_height,
		const waifu2xCancelFunc cancel_func = nullptr, const int crop_w = 128, const int crop_h = 128,
		const boost::optional<int> output_quality = boost::optional<int>(), const int output_depth = 8, const bool use_tta = false,
		const int batch_size = 1);

	// factor: 倍率
	// source: (4byte aligned when RGBA) source RGBA/BGRA etc
	// dest: (4byte aligned when RGBA) destination buffer
	// in_stride: source stride (bytes per row)
	// out_stride: dest stride (bytes per row)
	eWaifu2xError waifu2x(const double factor, const void* source, void* dest, const int width, const int height,
		const int in_channel, const int in_stride, const int out_channel, const int out_stride,
		const int crop_w = 128, const int crop_h = 128,  const bool use_tta = false, const int batch_size = 1);

	void Destroy();

	const std::string& used_process() const;

	static std::string GetModelName(const boost::filesystem::path &model_dir);
	static bool GetInfo(const boost::filesystem::path &model_dir, stInfo &info);
};
