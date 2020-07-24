#include "LaunchParameter.cuh"

#define MAXBLOCKSIZE 65535

CUDA_HOST LaunchParameter::LaunchParameter()
{

}

CUDA_HOST LaunchParameter LaunchParameter::make_2D1D_launchParameter(int size, int threadDim)
{
	LaunchParameter para;
	para.threads = dim3(threadDim, 1, 1);

	int blocks_ = (int)(ceil((size / ((real)threadDim))));
	para.blocks = dim3(blocks_, 1, 1);

	if (blocks_ > MAXBLOCKSIZE) {
		blocks_ = (int)sqrt((double)blocks_);
		para.blocks = dim3(blocks_, blocks_, 1);
	}
	return para;
}

CUDA_HOST LaunchParameter LaunchParameter::make_1D1D_launchParameter(int size, int threadDim)
{
	LaunchParameter para;
	para.threads = dim3(threadDim, 1, 1);
	int blocks_ = (int)(ceil((real)size / (real)threadDim));
	para.blocks = dim3(blocks_, 1);
	return para;
}

DEVICE int LaunchParameter::getGlobalIdx_2D_1D()
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int threadId = blockId * blockDim.x + threadIdx.x;
	return threadId;
}

DEVICE int LaunchParameter::getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

CUDA_HOST void LaunchParameter::print() const
{
	*logging::out << logging::Logger::INFO_INTERMEDIATE << "blocks: (" << blocks.x << ", " << blocks.y << ", " << blocks.z << ")"
		<< ", threads: (" << threads.x << ", " << threads.y << ", " << threads.z << ")\n";
}