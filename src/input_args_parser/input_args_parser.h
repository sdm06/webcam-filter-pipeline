#pragma once

#include <string>
#include <vector>
#include <cxxopts.hpp>

namespace cuda_filter
{

    enum class InputSource
    {
        WEBCAM,
        IMAGE,
        VIDEO,
        SYNTHETIC
    };

    enum class SyntheticPattern
    {
        CHECKERBOARD,
        GRADIENT,
        NOISE
    };

    // Describes a single filter stage in the pipeline (parsed from CLI)
    struct PipelineStageConfig
    {
        std::string filterType;
        int kernelSize;
        float intensity;
    };

    struct FilterOptions
    {
        InputSource inputSource;
        std::string inputPath;
        SyntheticPattern syntheticPattern;
        int deviceId;
        std::string filterType;
        int kernelSize;
        float sigma;
        float intensity;
        bool preview;

        // Pipeline options
        bool usePipeline;                              // --pipeline flag
        std::vector<PipelineStageConfig> pipelineStages; // Parsed from --stages
        bool multiStream;                              // --multi-stream
        bool showProfiler;                             // --profiler
        bool benchmarkMode;                            // --benchmark (runs single vs multi comparison)

        // Transition options
        bool enableTransition;       // --transition
        std::string transitionFilterA; // First filter for transition
        std::string transitionFilterB; // Second filter for transition
        float transitionDuration;    // --transition-duration (seconds)
    };

    class InputArgsParser
    {
    public:
        InputArgsParser(int argc, char **argv);

        FilterOptions parseArgs();

    private:
        int m_argc;
        char **m_argv;

        void setupOptions(cxxopts::Options &options);
        InputSource stringToInputSource(const std::string &str);
        SyntheticPattern stringToSyntheticPattern(const std::string &str);
        std::vector<PipelineStageConfig> parsePipelineStages(const std::string &stagesStr);
    };

} // namespace cuda_filter
