#pragma once

#include "BaseThread.h"

namespace AVUN 
{
    class SumProcessor : public BaseThread
    {
        std::mutex mProcessMutex;
        std::condition_variable mProcessCV;

        bool mProcess { false };

        const uint16_t* mSource = nullptr;
        uint16_t* mDestination = nullptr;
        size_t mCount = 0;

        int mThreadNum = 0;

    private:
        void Reset();
        void SendProcessComplete();

    protected: 
        virtual void ThreadFunc() override;

    public:
        SumProcessor(int aThreadNum);
        ~SumProcessor();

        void Process(const uint16_t* aSource, uint16_t* aDestination, size_t aCount);
        void WaitForComplete();
    };
}