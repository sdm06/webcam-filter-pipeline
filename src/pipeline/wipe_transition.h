#pragma once

#include <chrono>
#include <opencv2/opencv.hpp>
#include "filter_pipeline.h"

namespace cuda_filter
{

    /// Performs a horizontal wipe transition between two filter pipelines.
    ///
    /// The wipe sweeps left-to-right: at progress 0.0 the full frame comes
    /// from pipelineA, and at 1.0 it comes entirely from pipelineB.
    ///
    /// Usage:
    /// @code
    ///   WipeTransition wipe;
    ///   wipe.setDuration(1.5f);   // 1.5-second transition
    ///   wipe.start();
    ///   while (wipe.isActive()) {
    ///       wipe.apply(frame, output, pipeA, pipeB);
    ///   }
    /// @endcode
    class WipeTransition
    {
    public:
        WipeTransition();

        // -- Configuration ----------------------------------------------------

        /// Set the transition duration in seconds (default: 1.0).
        void setDuration(float seconds);
        float getDuration() const;

        // -- Lifecycle --------------------------------------------------------

        /// Start (or restart) the transition clock.
        void start();

        /// Reset to inactive state.
        void reset();

        /// @return true while the transition is in progress.
        bool isActive() const;

        /// @return progress in [0.0, 1.0].
        float getProgress() const;

        // -- Rendering --------------------------------------------------------

        /// Execute both pipelines on @p input and composite the results into
        /// @p output using the current wipe progress. Automatically deactivates
        /// when the transition completes (progress >= 1.0).
        void apply(const cv::Mat& input, cv::Mat& output,
                   FilterPipeline& pipelineA, FilterPipeline& pipelineB);

    private:
        float m_duration; ///< seconds
        bool m_active;
        std::chrono::steady_clock::time_point m_startTime;
    };

} // namespace cuda_filter
