#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <deque>
#include <opencv2/opencv.hpp>

namespace cuda_filter
{

    struct TimingRecord
    {
        std::string stageName;
        double durationMs; // milliseconds
    };

    struct FrameTiming
    {
        std::vector<TimingRecord> stageTimings;
        double totalPipelineMs;
        double uploadMs;    // H2D transfer
        double downloadMs;  // D2H transfer
        double transitionMs;
        std::chrono::steady_clock::time_point timestamp;
    };

    class PipelineProfiler
    {
    public:
        PipelineProfiler(size_t historySize = 120); // ~2 seconds at 60fps

        // Recording
        void beginFrame();
        void recordStage(const std::string &name, double durationMs);
        void recordUpload(double ms);
        void recordDownload(double ms);
        void recordTransition(double ms);
        void endFrame();

        // Querying
        double getAverageFPS() const;
        double getAveragePipelineMs() const;
        double getAverageStageMs(const std::string &name) const;
        const std::deque<FrameTiming> &getHistory() const;

        // Visualization - renders a timing overlay onto the frame
        void renderOverlay(cv::Mat &frame) const;

        // Comparison mode
        void setSingleStreamMs(double ms);
        void setMultiStreamMs(double ms);
        double getSingleStreamMs() const;
        double getMultiStreamMs() const;

    private:
        size_t m_historySize;
        std::deque<FrameTiming> m_history;
        FrameTiming m_currentFrame;
        bool m_frameInProgress;

        double m_singleStreamMs;
        double m_multiStreamMs;

        // Color palette for stages
        cv::Scalar getStageColor(size_t index) const;
    };

} // namespace cuda_filter
