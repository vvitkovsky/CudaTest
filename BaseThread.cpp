// Copyright 2019 ЦНИИ Циклон

#include "BaseThread.h"

namespace AVUN {
BaseThread::BaseThread() {
}

BaseThread::~BaseThread() {
    StopThread();
}

bool
BaseThread::StartThread() {

    if (mStopThread)
    {
        std::unique_lock<std::mutex> lock(mMutex);

        if (mStopThread) {
            // Wait for previous thread finish
            if (mThread.joinable()) mThread.join();

            // Launch new thread
            mStopThread = false;
            mThread = std::thread([&] {
                ThreadFunc();
                mStopThread = true;
            });

            return true;
        }
    }
	return false;
}

void
BaseThread::StopThread() {
    mStopThread = true;
    std::lock_guard<std::mutex> lock(mMutex);
    if (mThread.joinable()) mThread.join();
}

bool
BaseThread::IsThreadActive() {
	return !mStopThread;
}
}
