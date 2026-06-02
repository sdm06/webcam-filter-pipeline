#include "pipeline_profiler.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace cuda_filter
{

    PipelineProfiler::PipelineProfiler(size_t historySize)
        : m_historySize(historySize), m_frameInProgress(false),
          m_singleStreamMs(0.0), m_multiStreamMs(0.0)
    {
    }

    void PipelineProfiler::beginFrame()
    {
        m_currentFrame = FrameTiming{};
        m_currentFrame.timestamp = std::chrono::steady_clock::now();
        m_currentFrame.totalPipelineMs = 0.0;
        m_currentFrame.uploadMs = 0.0;
        m_currentFrame.downloadMs = 0.0;
        m_currentFrame.transitionMs = 0.0;
        m_frameInProgress = true;
    }

    void PipelineProfiler::recordStage(const std::string &name, double durationMs)
    {
        if (!m_frameInProgress)
            return;
        m_currentFrame.stageTimings.push_back({name, durationMs});
    }

    void PipelineProfiler::recordUpload(double ms)
    {
        if (!m_frameInProgress)
            return;
        m_currentFrame.uploadMs = ms;
    }

    void PipelineProfiler::recordDownload(double ms)
    {
        if (!m_frameInProgress)
            return;
        m_currentFrame.downloadMs = ms;
    }

    void PipelineProfiler::recordTransition(double ms)
    {
        if (!m_frameInProgress)
            return;
        m_currentFrame.transitionMs = ms;
    }

    void PipelineProfiler::endFrame()
    {
        if (!m_frameInProgress)
            return;

        // Compute total pipeline time from stages + transfer
        double stageTotal = 0.0;
        for (const auto &s : m_currentFrame.stageTimings)
        {
            stageTotal += s.durationMs;
        }
        m_currentFrame.totalPipelineMs = stageTotal + m_currentFrame.uploadMs +
                                          m_currentFrame.downloadMs + m_currentFrame.transitionMs;

        m_history.push_back(m_currentFrame);
        while (m_history.size() > m_historySize)
        {
            m_history.pop_front();
        }
        m_frameInProgress = false;
    }

    double PipelineProfiler::getAverageFPS() const
    {
        if (m_history.size() < 2)
            return 0.0;

        auto first = m_history.front().timestamp;
        auto last = m_history.back().timestamp;
        double elapsed = std::chrono::duration<double>(last - first).count();

        if (elapsed <= 0.0)
            return 0.0;

        return static_cast<double>(m_history.size() - 1) / elapsed;
    }

    double PipelineProfiler::getAveragePipelineMs() const
    {
        if (m_history.empty())
            return 0.0;

        double total = 0.0;
        for (const auto &frame : m_history)
        {
            total += frame.totalPipelineMs;
        }
        return total / static_cast<double>(m_history.size());
    }

    double PipelineProfiler::getAverageStageMs(const std::string &name) const
    {
        double total = 0.0;
        int count = 0;

        for (const auto &frame : m_history)
        {
            for (const auto &stage : frame.stageTimings)
            {
                if (stage.stageName == name)
                {
                    total += stage.durationMs;
                    count++;
                    break; // Only count once per frame
                }
            }
        }

        return count > 0 ? total / count : 0.0;
    }

    const std::deque<FrameTiming> &PipelineProfiler::getHistory() const
    {
        return m_history;
    }

    void PipelineProfiler::setSingleStreamMs(double ms)
    {
        m_singleStreamMs = ms;
    }

    void PipelineProfiler::setMultiStreamMs(double ms)
    {
        m_multiStreamMs = ms;
    }

    double PipelineProfiler::getSingleStreamMs() const
    {
        return m_singleStreamMs;
    }

    double PipelineProfiler::getMultiStreamMs() const
    {
        return m_multiStreamMs;
    }

    cv::Scalar PipelineProfiler::getStageColor(size_t index) const
    {
        // Generate distinct colors using HSV color wheel
        static const std::vector<cv::Scalar> palette = {
            cv::Scalar(0, 200, 255),   // Orange (BGR)
            cv::Scalar(255, 100, 50),  // Blue
            cv::Scalar(50, 255, 50),   // Green
            cv::Scalar(50, 50, 255),   // Red
            cv::Scalar(255, 255, 50),  // Cyan
            cv::Scalar(255, 50, 255),  // Magenta
            cv::Scalar(50, 255, 255),  // Yellow
            cv::Scalar(200, 200, 200), // Light gray
        };
        return palette[index % palette.size()];
    }

    void PipelineProfiler::renderOverlay(cv::Mat &frame) const
    {
        if (m_history.empty() || frame.empty())
            return;

        const int overlayHeight = 200;
        const int margin = 10;
        const int barChartHeight = 100;
        const int legendHeight = 80;
        const int textLineHeight = 16;

        // Semi-transparent overlay background
        int overlayY = frame.rows - overlayHeight - margin;
        if (overlayY < 0)
            overlayY = 0;

        cv::Mat overlayRegion = frame(cv::Rect(margin, overlayY, frame.cols - 2 * margin, overlayHeight));

        // Create dark semi-transparent background
        cv::Mat darkBg(overlayRegion.size(), overlayRegion.type(), cv::Scalar(20, 20, 20));
        cv::addWeighted(overlayRegion, 0.3, darkBg, 0.7, 0, overlayRegion);

        // Draw border
        cv::rectangle(frame,
                      cv::Point(margin, overlayY),
                      cv::Point(frame.cols - margin, overlayY + overlayHeight),
                      cv::Scalar(100, 100, 100), 1);

        // --- Header: FPS and total timing ---
        double fps = getAverageFPS();
        double avgMs = getAveragePipelineMs();

        std::ostringstream headerSS;
        headerSS << std::fixed << std::setprecision(1);
        headerSS << "Pipeline FPS: " << fps << "  |  Avg: " << avgMs << " ms";

        cv::putText(frame, headerSS.str(),
                    cv::Point(margin + 10, overlayY + 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 200), 1);

        // --- Bar chart: timeline of recent frames ---
        int chartX = margin + 10;
        int chartY = overlayY + 28;
        int chartW = frame.cols - 2 * margin - 20;

        // Find max pipeline time for scaling
        double maxMs = 1.0;
        for (const auto &ft : m_history)
        {
            if (ft.totalPipelineMs > maxMs)
                maxMs = ft.totalPipelineMs;
        }
        // Round up to nice number
        maxMs = std::ceil(maxMs / 5.0) * 5.0;
        if (maxMs < 5.0)
            maxMs = 5.0;

        // Draw time scale
        cv::putText(frame, std::to_string(static_cast<int>(maxMs)) + "ms",
                    cv::Point(chartX, chartY + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(150, 150, 150), 1);
        cv::putText(frame, "0ms",
                    cv::Point(chartX, chartY + barChartHeight),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(150, 150, 150), 1);

        int barAreaX = chartX + 40;
        int barAreaW = chartW - 40;

        if (!m_history.empty() && barAreaW > 0)
        {
            int barWidth = std::max(1, barAreaW / static_cast<int>(m_historySize));

            for (size_t i = 0; i < m_history.size(); i++)
            {
                const auto &ft = m_history[i];
                int x = barAreaX + static_cast<int>(i) * barWidth;

                if (x + barWidth > barAreaX + barAreaW)
                    break;

                // Draw stacked bar for this frame
                double cumulativeMs = 0.0;

                // Upload bar
                if (ft.uploadMs > 0)
                {
                    int h = static_cast<int>((ft.uploadMs / maxMs) * barChartHeight);
                    int yBottom = chartY + barChartHeight - static_cast<int>((cumulativeMs / maxMs) * barChartHeight);
                    cv::rectangle(frame,
                                  cv::Point(x, yBottom - h),
                                  cv::Point(x + barWidth - 1, yBottom),
                                  cv::Scalar(128, 128, 128), cv::FILLED);
                    cumulativeMs += ft.uploadMs;
                }

                // Stage bars
                for (size_t s = 0; s < ft.stageTimings.size(); s++)
                {
                    int h = static_cast<int>((ft.stageTimings[s].durationMs / maxMs) * barChartHeight);
                    int yBottom = chartY + barChartHeight - static_cast<int>((cumulativeMs / maxMs) * barChartHeight);
                    cv::rectangle(frame,
                                  cv::Point(x, yBottom - h),
                                  cv::Point(x + barWidth - 1, yBottom),
                                  getStageColor(s), cv::FILLED);
                    cumulativeMs += ft.stageTimings[s].durationMs;
                }

                // Download bar
                if (ft.downloadMs > 0)
                {
                    int h = static_cast<int>((ft.downloadMs / maxMs) * barChartHeight);
                    int yBottom = chartY + barChartHeight - static_cast<int>((cumulativeMs / maxMs) * barChartHeight);
                    cv::rectangle(frame,
                                  cv::Point(x, yBottom - h),
                                  cv::Point(x + barWidth - 1, yBottom),
                                  cv::Scalar(180, 180, 180), cv::FILLED);
                    cumulativeMs += ft.downloadMs;
                }

                // Transition bar
                if (ft.transitionMs > 0)
                {
                    int h = static_cast<int>((ft.transitionMs / maxMs) * barChartHeight);
                    int yBottom = chartY + barChartHeight - static_cast<int>((cumulativeMs / maxMs) * barChartHeight);
                    cv::rectangle(frame,
                                  cv::Point(x, yBottom - h),
                                  cv::Point(x + barWidth - 1, yBottom),
                                  cv::Scalar(200, 100, 255), cv::FILLED);
                }
            }
        }

        // --- Legend area ---
        int legendY = chartY + barChartHeight + 8;

        // Collect unique stage names from latest frame
        if (!m_history.empty())
        {
            const auto &latestFrame = m_history.back();
            int legendX = margin + 15;

            // Upload/Download legend
            cv::rectangle(frame,
                          cv::Point(legendX, legendY),
                          cv::Point(legendX + 10, legendY + 10),
                          cv::Scalar(128, 128, 128), cv::FILLED);
            cv::putText(frame, "Upload",
                        cv::Point(legendX + 14, legendY + 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200), 1);
            legendX += 70;

            cv::rectangle(frame,
                          cv::Point(legendX, legendY),
                          cv::Point(legendX + 10, legendY + 10),
                          cv::Scalar(180, 180, 180), cv::FILLED);
            cv::putText(frame, "Download",
                        cv::Point(legendX + 14, legendY + 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200), 1);
            legendX += 80;

            // Stage legends
            for (size_t s = 0; s < latestFrame.stageTimings.size(); s++)
            {
                cv::Scalar color = getStageColor(s);
                cv::rectangle(frame,
                              cv::Point(legendX, legendY),
                              cv::Point(legendX + 10, legendY + 10),
                              color, cv::FILLED);

                std::ostringstream label;
                label << latestFrame.stageTimings[s].stageName << " ("
                      << std::fixed << std::setprecision(2)
                      << getAverageStageMs(latestFrame.stageTimings[s].stageName) << "ms)";

                cv::putText(frame, label.str(),
                            cv::Point(legendX + 14, legendY + 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200), 1);
                legendX += static_cast<int>(label.str().size()) * 7 + 20;

                // Wrap to next line if too wide
                if (legendX > frame.cols - margin - 100)
                {
                    legendX = margin + 15;
                    legendY += textLineHeight;
                }
            }

            // Transition legend
            if (latestFrame.transitionMs > 0)
            {
                cv::rectangle(frame,
                              cv::Point(legendX, legendY),
                              cv::Point(legendX + 10, legendY + 10),
                              cv::Scalar(200, 100, 255), cv::FILLED);
                cv::putText(frame, "Transition",
                            cv::Point(legendX + 14, legendY + 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(200, 200, 200), 1);
            }
        }

        // --- Single-stream vs Multi-stream comparison ---
        if (m_singleStreamMs > 0 && m_multiStreamMs > 0)
        {
            legendY += textLineHeight + 4;

            std::ostringstream compSS;
            compSS << std::fixed << std::setprecision(2);
            compSS << "Single-stream: " << m_singleStreamMs << " ms  |  "
                   << "Multi-stream: " << m_multiStreamMs << " ms  |  "
                   << "Speedup: " << (m_singleStreamMs / m_multiStreamMs) << "x";

            cv::Scalar compColor = (m_multiStreamMs < m_singleStreamMs)
                                       ? cv::Scalar(0, 255, 100)   // Green if multi is faster
                                       : cv::Scalar(0, 100, 255);  // Orange if single is faster

            cv::putText(frame, compSS.str(),
                        cv::Point(margin + 15, legendY + 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, compColor, 1);
        }
    }

} // namespace cuda_filter
