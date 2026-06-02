#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>
#include "input_args_parser/input_args_parser.h"
#include "utils/input_handler.h"
#include "utils/filter_utils.h"
#include "pipeline/filter_pipeline.h"
#include "pipeline/wipe_transition.h"
#include "pipeline/pipeline_profiler.h"
#include "kernels/kernels.h"
#include <chrono>

int main(int argc, char **argv)
{
    // Initialize logger
    plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::info, &consoleAppender);

    // Parse command line arguments
    cuda_filter::InputArgsParser parser(argc, argv);
    cuda_filter::FilterOptions options = parser.parseArgs();

    // Initialize input handler
    cuda_filter::InputHandler inputHandler(options);
    if (!inputHandler.isOpened())
    {
        PLOG_ERROR << "Failed to initialize input source";
        return -1;
    }

    cuda_filter::FilterPipeline pipeline;
    cuda_filter::FilterPipeline pipelineA;
    cuda_filter::FilterPipeline pipelineB;
    cuda_filter::WipeTransition transition;
    cuda_filter::PipelineProfiler profiler;

    if (options.enableTransition)
    {
        pipelineA.addStage("A", cuda_filter::FilterUtils::stringToFilterType(options.transitionFilterA), options.kernelSize, options.intensity);
        pipelineB.addStage("B", cuda_filter::FilterUtils::stringToFilterType(options.transitionFilterB), options.kernelSize, options.intensity);
        transition.setDuration(options.transitionDuration);
        transition.start();
    }
    else if (options.usePipeline)
    {
        for (const auto& stage : options.pipelineStages)
        {
            pipeline.addStage(stage.filterType, cuda_filter::FilterUtils::stringToFilterType(stage.filterType), stage.kernelSize, stage.intensity);
        }
    }
    else
    {
        pipeline.addStage(options.filterType, cuda_filter::FilterUtils::stringToFilterType(options.filterType), options.kernelSize, options.intensity);
    }

    cv::Mat frame, outputFrame;
    double fps = 0.0;
    int frameCount = 0;
    double startTime = static_cast<double>(cv::getTickCount());

    PLOG_INFO << "Press 'ESC' to exit";
    PLOG_INFO << "Press '1' to add Blur filter";
    PLOG_INFO << "Press '2' to add Sharpen filter";
    PLOG_INFO << "Press '3' to add Edge Detection filter";
    PLOG_INFO << "Press '4' to remove the last filter";
    if (options.enableTransition) {
        PLOG_INFO << "Press 't' to trigger transition";
    }

    while (true)
    {
        if (!inputHandler.readFrame(frame))
        {
            PLOG_ERROR << "Failed to read frame";
            break;
        }

        profiler.beginFrame();

        const double procStart = static_cast<double>(cv::getTickCount());

        if (options.enableTransition)
        {
            transition.apply(frame, outputFrame, pipelineA, pipelineB);
        }
        else if (options.benchmarkMode)
        {
            cv::Mat outMulti;
            const double s1 = static_cast<double>(cv::getTickCount());
            pipeline.executeGPU(frame, outputFrame);
            const double e1 = static_cast<double>(cv::getTickCount());
            
            const double s2 = static_cast<double>(cv::getTickCount());
            pipeline.executeGPUMultiStream(frame, outMulti);
            const double e2 = static_cast<double>(cv::getTickCount());

            profiler.setSingleStreamMs((e1 - s1) * 1000.0 / cv::getTickFrequency());
            profiler.setMultiStreamMs((e2 - s2) * 1000.0 / cv::getTickFrequency());
            
            // Show multi stream result as output
            outputFrame = outMulti;
        }
        else if (options.usePipeline)
        {
            if (options.multiStream)
            {
                pipeline.executeGPUMultiStream(frame, outputFrame);
            }
            else
            {
                pipeline.executeGPU(frame, outputFrame);
            }
        }
        else
        {
            pipeline.executeGPU(frame, outputFrame);
        }

        const double procEnd = static_cast<double>(cv::getTickCount());
        const double procTimeMs = (procEnd - procStart) * 1000.0 / cv::getTickFrequency();
        
        profiler.recordStage("Processing", procTimeMs);
        profiler.endFrame();

        frameCount++;
        if ((procEnd - startTime) / cv::getTickFrequency() >= 1.0)
        {
            fps = frameCount;
            frameCount = 0;
            startTime = procEnd;
        }

        if (options.showProfiler)
        {
            profiler.renderOverlay(outputFrame);
        }
        
        std::string statusText = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(outputFrame, statusText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        if (options.preview)
        {
            inputHandler.displaySideBySide(frame, outputFrame);
        }
        else
        {
            inputHandler.displayFrame(outputFrame);
        }

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) // ESC
        {
            break;
        }
        else if (key == '1')
        {
            pipeline.addStage("blur", cuda_filter::FilterType::BLUR, 3, 1.0f);
            PLOG_INFO << "Added blur filter. Pipeline: " << pipeline.getPipelineDescription();
        }
        else if (key == '2')
        {
            pipeline.addStage("sharpen", cuda_filter::FilterType::SHARPEN, 3, 1.0f);
            PLOG_INFO << "Added sharpen filter. Pipeline: " << pipeline.getPipelineDescription();
        }
        else if (key == '3')
        {
            pipeline.addStage("edge", cuda_filter::FilterType::EDGE_DETECTION, 3, 1.0f);
            PLOG_INFO << "Added edge detection filter. Pipeline: " << pipeline.getPipelineDescription();
        }
        else if (key == '4')
        {
            if (pipeline.getStageCount() > 0)
            {
                pipeline.removeStage(pipeline.getStageCount() - 1);
                PLOG_INFO << "Removed last filter. Pipeline: " << pipeline.getPipelineDescription();
            }
        }
        else if (key == 't' && options.enableTransition)
        {
            // Swap pipelines
            auto stagesA = pipelineA.getStages();
            pipelineA.clearStages();
            for (auto& s : pipelineB.getStages()) {
                pipelineA.addStage(s.name, s.filterType, s.kernelSize, s.intensity);
            }
            pipelineB.clearStages();
            for (auto& s : stagesA) {
                pipelineB.addStage(s.name, s.filterType, s.kernelSize, s.intensity);
            }
            
            transition.reset();
            transition.start();
            PLOG_INFO << "Triggered transition!";
        }
    }

    PLOG_INFO << "Application terminated";
    return 0;
}
