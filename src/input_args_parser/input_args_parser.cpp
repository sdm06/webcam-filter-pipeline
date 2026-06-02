#include "input_args_parser.h"
#include <iostream>
#include <sstream>
#include "../utils/version.h"

namespace cuda_filter
{

    InputArgsParser::InputArgsParser(int argc, char **argv)
        : m_argc(argc), m_argv(argv)
    {
    }

    InputSource InputArgsParser::stringToInputSource(const std::string &str)
    {
        if (str == "webcam")
            return InputSource::WEBCAM;
        if (str == "image")
            return InputSource::IMAGE;
        if (str == "video")
            return InputSource::VIDEO;
        if (str == "synthetic")
            return InputSource::SYNTHETIC;
        throw std::runtime_error("Invalid input source: " + str);
    }

    SyntheticPattern InputArgsParser::stringToSyntheticPattern(const std::string &str)
    {
        if (str == "checkerboard")
            return SyntheticPattern::CHECKERBOARD;
        if (str == "gradient")
            return SyntheticPattern::GRADIENT;
        if (str == "noise")
            return SyntheticPattern::NOISE;
        throw std::runtime_error("Invalid synthetic pattern: " + str);
    }

    std::vector<PipelineStageConfig> InputArgsParser::parsePipelineStages(const std::string &stagesStr)
    {
        // Format: "blur:3:1.0,sharpen:5:1.5,edge:3:1.0"
        // Each stage is "filterType:kernelSize:intensity"
        std::vector<PipelineStageConfig> stages;
        std::stringstream ss(stagesStr);
        std::string stageStr;

        while (std::getline(ss, stageStr, ','))
        {
            PipelineStageConfig stage;
            std::stringstream stageSS(stageStr);
            std::string token;

            // Parse filter type
            if (std::getline(stageSS, token, ':'))
            {
                stage.filterType = token;
            }
            else
            {
                throw std::runtime_error("Invalid pipeline stage format: " + stageStr);
            }

            // Parse kernel size (optional, default 3)
            if (std::getline(stageSS, token, ':'))
            {
                stage.kernelSize = std::stoi(token);
            }
            else
            {
                stage.kernelSize = 3;
            }

            // Parse intensity (optional, default 1.0)
            if (std::getline(stageSS, token, ':'))
            {
                stage.intensity = std::stof(token);
            }
            else
            {
                stage.intensity = 1.0f;
            }

            stages.push_back(stage);
        }

        return stages;
    }

    FilterOptions InputArgsParser::parseArgs()
    {
        cxxopts::Options options("cuda-webcam-filter", "Real-time webcam filter with CUDA acceleration and pipeline support");

        setupOptions(options);

        auto result = options.parse(m_argc, m_argv);

        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        if (result.count("version"))
        {
            std::cout << "CUDA Webcam Filter version " << CUDA_WEBCAM_FILTER_VERSION << std::endl;
            exit(0);
        }

        FilterOptions filterOptions;

        // Parse input source
        std::string inputType = result["input"].as<std::string>();
        filterOptions.inputSource = stringToInputSource(inputType);
        filterOptions.inputPath = result["path"].as<std::string>();

        if (filterOptions.inputSource == InputSource::SYNTHETIC)
        {
            std::string patternType = result["synthetic"].as<std::string>();
            filterOptions.syntheticPattern = stringToSyntheticPattern(patternType);
        }
        else if (filterOptions.inputSource == InputSource::WEBCAM)
        {
            filterOptions.deviceId = result["device"].as<int>();
        }

        filterOptions.filterType = result["filter"].as<std::string>();
        filterOptions.kernelSize = result["kernel-size"].as<int>();
        filterOptions.sigma = result["sigma"].as<float>();
        filterOptions.intensity = result["intensity"].as<float>();
        filterOptions.preview = result.count("preview") > 0;

        // Pipeline options
        filterOptions.usePipeline = result.count("pipeline") > 0;
        filterOptions.multiStream = result.count("multi-stream") > 0;
        filterOptions.showProfiler = result.count("profiler") > 0;
        filterOptions.benchmarkMode = result.count("benchmark") > 0;

        if (filterOptions.usePipeline && result.count("stages"))
        {
            std::string stagesStr = result["stages"].as<std::string>();
            filterOptions.pipelineStages = parsePipelineStages(stagesStr);
        }
        else if (filterOptions.usePipeline)
        {
            // Default pipeline: blur -> sharpen
            filterOptions.pipelineStages = {
                {"blur", 3, 1.0f},
                {"sharpen", 3, 1.0f}};
        }

        // Transition options
        filterOptions.enableTransition = result.count("transition") > 0;
        filterOptions.transitionFilterA = result["transition-filter-a"].as<std::string>();
        filterOptions.transitionFilterB = result["transition-filter-b"].as<std::string>();
        filterOptions.transitionDuration = result["transition-duration"].as<float>();

        return filterOptions;
    }

    void InputArgsParser::setupOptions(cxxopts::Options &options)
    {
        options.add_options()
            // Original options
            ("i,input", "Input source: 'webcam', 'image', 'video', or 'synthetic'",
             cxxopts::value<std::string>()->default_value("webcam"))
            ("p,path", "Path to input image or video file (when not using webcam)",
             cxxopts::value<std::string>()->default_value("test_image.jpg"))
            ("s,synthetic", "Synthetic pattern type: 'checkerboard', 'gradient', 'noise'",
             cxxopts::value<std::string>()->default_value("checkerboard"))
            ("d,device", "Camera device ID",
             cxxopts::value<int>()->default_value("0"))
            ("f,filter", "Filter type: blur, sharpen, edge, emboss",
             cxxopts::value<std::string>()->default_value("blur"))
            ("k,kernel-size", "Kernel size for filters",
             cxxopts::value<int>()->default_value("3"))
            ("sigma", "Sigma value for Gaussian blur",
             cxxopts::value<float>()->default_value("1.0"))
            ("intensity", "Filter intensity",
             cxxopts::value<float>()->default_value("1.0"))
            ("preview", "Show original video alongside filtered")
            ("h,help", "Print usage")
            ("v,version", "Print version information")

            // Pipeline options
            ("pipeline", "Enable filter pipeline mode (apply multiple filters in sequence)")
            ("stages", "Pipeline stages in format 'filter:kernelSize:intensity,...'\n"
                        "  Example: 'blur:5:1.0,sharpen:3:1.5,edge:3:1.0'",
             cxxopts::value<std::string>())
            ("multi-stream", "Use multiple CUDA streams for pipeline execution")
            ("profiler", "Show real-time performance profiler overlay")
            ("benchmark", "Run single-stream vs multi-stream benchmark comparison")

            // Transition options
            ("transition", "Enable wipe transition between two filter configurations")
            ("transition-filter-a", "First filter for wipe transition",
             cxxopts::value<std::string>()->default_value("blur"))
            ("transition-filter-b", "Second filter for wipe transition",
             cxxopts::value<std::string>()->default_value("edge"))
            ("transition-duration", "Transition duration in seconds",
             cxxopts::value<float>()->default_value("3.0"));
    }

} // namespace cuda_filter
