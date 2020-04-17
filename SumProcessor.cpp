#include "SumProcessor.h"

using namespace std::chrono_literals;

namespace AVUN 
{
    SumProcessor::SumProcessor(int aThreadNum) :
        mThreadNum(aThreadNum) 
    {
        StartThread();
    }

    SumProcessor::~SumProcessor() {
        mStopThread = true;
        mProcess = true;
        mProcessCV.notify_all();
        StopThread();
    }

    void SumProcessor::Reset() {
        mSource = nullptr;
        mDestination = nullptr;
        mCount = 0;
    }

    void SumProcessor::SendProcessComplete() {
        {
            std::lock_guard<std::mutex> lock(mProcessMutex);
            mProcess = false;
        }
        mProcessCV.notify_all();
    }

    void SumProcessor::ThreadFunc() {
        while (IsThreadActive()) {
            // Wait for processing event
            std::unique_lock<std::mutex> lock(mProcessMutex);
            mProcessCV.wait(lock, [&] { return mProcess; });

            if (!IsThreadActive()) {
                // Thread stop called
                break;
            }

            // Process data
            try
            {
                for (int i = 0; i < mCount; ++i) {
                    // Check overflow
                    if (mDestination[i] > std::numeric_limits<uint16_t>::max() - mSource[i]) {
                        mDestination[i] = std::numeric_limits<uint16_t>::max();
                    }
                    else {
                        mDestination[i] += mSource[i];
                    }
                }
                // Reset data
                Reset();
            }
            catch (...) {
            }

            // Free mutex
            lock.unlock();

            // Complete
            SendProcessComplete();
        }

        // Thread complete
        SendProcessComplete();
    }

    void SumProcessor::Process(const uint16_t* aSource, uint16_t* aDestination, size_t aCount) {
        // Set data
        {
            std::lock_guard<std::mutex> lock(mProcessMutex);
            mSource = aSource;
            mDestination = aDestination;
            mCount = aCount;
            mProcess = true;
        }
        mProcessCV.notify_all();
    }

    void SumProcessor::WaitForComplete() {
        std::unique_lock<std::mutex> lock(mProcessMutex);
        mProcessCV.wait(lock, [&] { return !mProcess; });
    }
}