#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr std::string_view kDefaultEngine{
    "outputs/tensorrt/perception_rt_mit_b2_fp16.engine"};
constexpr std::string_view kDefaultOutputDirectory{"outputs/native_cpp"};
constexpr int kDefaultWarmupIterations{30};
constexpr int kDefaultMeasuredIterations{100};

using Shape = std::array<std::int64_t, 4>;

struct TensorSpec {
    std::string_view name;
    nvinfer1::TensorIOMode mode;
    Shape shape;
};

constexpr std::array<TensorSpec, 4> kTensorSpecs{{
    {"image", nvinfer1::TensorIOMode::kINPUT, {1, 3, 320, 640}},
    {"semantic_logits", nvinfer1::TensorIOMode::kOUTPUT, {1, 15, 320, 640}},
    {"log_depth", nvinfer1::TensorIOMode::kOUTPUT, {1, 1, 320, 640}},
    {"depth_log_scale", nvinfer1::TensorIOMode::kOUTPUT, {1, 1, 320, 640}},
}};

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, char const* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

[[noreturn]] void throwCudaError(cudaError_t status, std::string_view operation) {
    throw std::runtime_error(
        std::string{operation} + ": " + cudaGetErrorString(status));
}

void checkCuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throwCudaError(status, operation);
    }
}

class CudaStream final {
public:
    CudaStream() {
        checkCuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }

    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(CudaStream const&) = delete;
    CudaStream& operator=(CudaStream const&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_{nullptr};
};

class DeviceBuffer final {
public:
    explicit DeviceBuffer(std::size_t size) : size_{size} {
        checkCuda(cudaMalloc(&data_, size_), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    [[nodiscard]] void* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    void* data_{nullptr};
    std::size_t size_{};
};

struct Options {
    fs::path engine{kDefaultEngine};
    std::optional<fs::path> input;
    fs::path outputDirectory{kDefaultOutputDirectory};
    int warmupIterations{kDefaultWarmupIterations};
    int measuredIterations{kDefaultMeasuredIterations};
    bool showHelp{false};
};

void printUsage(std::ostream& output) {
    output
        << "Usage: perception_rt_native [options]\n\n"
        << "Options:\n"
        << "  --engine PATH       FP16 TensorRT engine\n"
        << "  --input PATH        Raw FP16 NCHW input; zeros when omitted\n"
        << "  --output-dir PATH   Directory for raw FP16 outputs\n"
        << "  --warmup N          Warmup iterations (default: 30)\n"
        << "  --iterations N      Measured iterations (default: 100)\n"
        << "  -h, --help           Show this message\n";
}

std::string requireValue(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(
            std::string{"Missing value for "} + argv[index]);
    }
    ++index;
    return argv[index];
}

int parseInteger(std::string const& value, std::string_view option) {
    std::size_t consumed{};
    int parsed{};
    try {
        parsed = std::stoi(value, &consumed);
    } catch (std::exception const&) {
        throw std::invalid_argument(
            std::string{option} + " must be an integer");
    }
    if (consumed != value.size()) {
        throw std::invalid_argument(
            std::string{option} + " must be an integer");
    }
    return parsed;
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        std::string const argument{argv[index]};
        if (argument == "-h" || argument == "--help") {
            options.showHelp = true;
        } else if (argument == "--engine") {
            options.engine = requireValue(index, argc, argv);
        } else if (argument == "--input") {
            options.input = requireValue(index, argc, argv);
        } else if (argument == "--output-dir") {
            options.outputDirectory = requireValue(index, argc, argv);
        } else if (argument == "--warmup") {
            options.warmupIterations = parseInteger(
                requireValue(index, argc, argv), argument);
        } else if (argument == "--iterations") {
            options.measuredIterations = parseInteger(
                requireValue(index, argc, argv), argument);
        } else {
            throw std::invalid_argument("Unknown argument: " + argument);
        }
    }

    if (options.warmupIterations < 0) {
        throw std::invalid_argument("--warmup must be nonnegative");
    }
    if (options.measuredIterations <= 0) {
        throw std::invalid_argument("--iterations must be positive");
    }
    return options;
}

std::vector<char> readBinaryFile(fs::path const& path) {
    std::ifstream input{path, std::ios::binary | std::ios::ate};
    if (!input) {
        throw std::runtime_error("Could not open " + path.string());
    }
    auto const end = input.tellg();
    if (end <= 0) {
        throw std::runtime_error("File is empty: " + path.string());
    }
    std::vector<char> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        throw std::runtime_error("Could not read " + path.string());
    }
    return bytes;
}

std::size_t elementCount(Shape const& shape) {
    return std::accumulate(
        shape.begin(), shape.end(), std::size_t{1},
        [](std::size_t product, std::int64_t dimension) {
            return product * static_cast<std::size_t>(dimension);
        });
}

std::string shapeString(Shape const& shape) {
    return "[" + std::to_string(shape[0]) + ", " +
        std::to_string(shape[1]) + ", " +
        std::to_string(shape[2]) + ", " +
        std::to_string(shape[3]) + "]";
}

void validateTensor(
    nvinfer1::ICudaEngine const& engine,
    TensorSpec const& expected,
    int index) {
    char const* actualName = engine.getIOTensorName(index);
    if (actualName == nullptr || actualName != expected.name) {
        throw std::runtime_error(
            "Unexpected tensor name at index " + std::to_string(index));
    }

    if (engine.getTensorIOMode(actualName) != expected.mode) {
        throw std::runtime_error(
            "Unexpected I/O mode for " + std::string{expected.name});
    }

    if (engine.getTensorDataType(actualName) != nvinfer1::DataType::kHALF) {
        throw std::runtime_error(
            "Expected FP16 tensor " + std::string{expected.name});
    }

    auto const dimensions = engine.getTensorShape(actualName);
    if (dimensions.nbDims != static_cast<int>(expected.shape.size())) {
        throw std::runtime_error(
            "Unexpected rank for " + std::string{expected.name});
    }
    for (int axis = 0; axis < dimensions.nbDims; ++axis) {
        if (dimensions.d[axis] != expected.shape[static_cast<std::size_t>(axis)]) {
            throw std::runtime_error(
                "Unexpected shape for " + std::string{expected.name});
        }
    }
}

void validateEngine(nvinfer1::ICudaEngine const& engine) {
    if (engine.getNbIOTensors() != static_cast<int>(kTensorSpecs.size())) {
        throw std::runtime_error("Engine must contain exactly four I/O tensors");
    }
    for (std::size_t index = 0; index < kTensorSpecs.size(); ++index) {
        validateTensor(engine, kTensorSpecs[index], static_cast<int>(index));
    }
}

std::vector<std::uint16_t> loadInput(
    std::optional<fs::path> const& path,
    std::size_t elementCountValue) {
    std::vector<std::uint16_t> values(elementCountValue, 0);
    if (!path) {
        return values;
    }

    std::ifstream input{*path, std::ios::binary | std::ios::ate};
    if (!input) {
        throw std::runtime_error("Could not open " + path->string());
    }
    auto const expectedBytes = values.size() * sizeof(std::uint16_t);
    if (input.tellg() != static_cast<std::streamoff>(expectedBytes)) {
        throw std::runtime_error(
            "Input must contain exactly " + std::to_string(expectedBytes) +
            " bytes");
    }
    input.seekg(0);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(expectedBytes));
    if (!input) {
        throw std::runtime_error("Could not read " + path->string());
    }
    return values;
}

bool allFinite(std::vector<std::uint16_t> const& values) {
    return std::all_of(values.begin(), values.end(), [](std::uint16_t bits) {
        return (bits & 0x7C00U) != 0x7C00U;
    });
}

void writeTensor(
    fs::path const& directory,
    TensorSpec const& spec,
    std::vector<std::uint16_t> const& values) {
    fs::create_directories(directory);
    fs::path const path = directory / (std::string{spec.name} + ".fp16.bin");
    std::ofstream output{path, std::ios::binary};
    output.write(
        reinterpret_cast<char const*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(std::uint16_t)));
    if (!output) {
        throw std::runtime_error("Could not write " + path.string());
    }
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    auto const rank = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size())));
    return values[std::max<std::size_t>(1, rank) - 1];
}

void execute(
    nvinfer1::IExecutionContext& context,
    cudaStream_t stream) {
    if (!context.enqueueV3(stream)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
}

int run(Options const& options) {
    if (!fs::is_regular_file(options.engine)) {
        throw std::runtime_error("Engine not found: " + options.engine.string());
    }

    int deviceCount{};
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount < 1) {
        throw std::runtime_error("No CUDA device is available");
    }
    checkCuda(cudaSetDevice(0), "cudaSetDevice");

    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");

    Logger logger;
    auto const engineBytes = readBinaryFile(options.engine);
    std::unique_ptr<nvinfer1::IRuntime> runtime{
        nvinfer1::createInferRuntime(logger)};
    if (!runtime) {
        throw std::runtime_error("Could not create TensorRT runtime");
    }

    std::unique_ptr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size())};
    if (!engine) {
        throw std::runtime_error("Could not deserialize TensorRT engine");
    }
    validateEngine(*engine);

    std::unique_ptr<nvinfer1::IExecutionContext> context{
        engine->createExecutionContext()};
    if (!context) {
        throw std::runtime_error("Could not create TensorRT execution context");
    }

    std::array<std::unique_ptr<DeviceBuffer>, kTensorSpecs.size()> deviceBuffers;
    std::array<std::vector<std::uint16_t>, kTensorSpecs.size()> hostTensors;

    hostTensors[0] = loadInput(
        options.input, elementCount(kTensorSpecs[0].shape));

    for (std::size_t index = 0; index < kTensorSpecs.size(); ++index) {
        auto const elements = elementCount(kTensorSpecs[index].shape);
        auto const bytes = elements * sizeof(std::uint16_t);
        if (index != 0) {
            hostTensors[index].resize(elements);
        }
        deviceBuffers[index] = std::make_unique<DeviceBuffer>(bytes);
        if (!context->setTensorAddress(
                kTensorSpecs[index].name.data(), deviceBuffers[index]->data())) {
            throw std::runtime_error(
                "Could not bind " + std::string{kTensorSpecs[index].name});
        }
    }

    CudaStream stream;
    checkCuda(
        cudaMemcpyAsync(
            deviceBuffers[0]->data(), hostTensors[0].data(),
            deviceBuffers[0]->size(), cudaMemcpyHostToDevice, stream.get()),
        "cudaMemcpyAsync input");

    for (int iteration = 0; iteration < options.warmupIterations; ++iteration) {
        execute(*context, stream.get());
    }
    checkCuda(cudaStreamSynchronize(stream.get()), "warmup synchronization");

    std::vector<double> latencies;
    latencies.reserve(static_cast<std::size_t>(options.measuredIterations));
    for (int iteration = 0; iteration < options.measuredIterations; ++iteration) {
        auto const start = std::chrono::steady_clock::now();
        execute(*context, stream.get());
        checkCuda(cudaStreamSynchronize(stream.get()), "inference synchronization");
        auto const stop = std::chrono::steady_clock::now();
        latencies.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
    }

    for (std::size_t index = 1; index < kTensorSpecs.size(); ++index) {
        checkCuda(
            cudaMemcpyAsync(
                hostTensors[index].data(), deviceBuffers[index]->data(),
                deviceBuffers[index]->size(), cudaMemcpyDeviceToHost, stream.get()),
            "cudaMemcpyAsync output");
    }
    checkCuda(cudaStreamSynchronize(stream.get()), "output synchronization");

    for (std::size_t index = 1; index < kTensorSpecs.size(); ++index) {
        if (!allFinite(hostTensors[index])) {
            throw std::runtime_error(
                "Non-finite values in " + std::string{kTensorSpecs[index].name});
        }
        writeTensor(
            options.outputDirectory, kTensorSpecs[index], hostTensors[index]);
    }

    auto const mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) /
        static_cast<double>(latencies.size());

    std::cout << "TensorRT runtime: " << getInferLibVersion() << '\n';
    std::cout << "GPU: " << properties.name << '\n';
    std::cout << "Precision: FP16\n";
    for (auto const& spec : kTensorSpecs) {
        std::cout << spec.name << ' ' << shapeString(spec.shape) << '\n';
    }
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Mean latency: " << mean << " ms\n";
    std::cout << "P95 latency: " << percentile(latencies, 0.95) << " ms\n";
    std::cout << "Throughput: " << 1000.0 / mean << " FPS\n";
    std::cout << "Outputs: " << options.outputDirectory << '\n';
    std::cout << "All outputs finite: true\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto const options = parseOptions(argc, argv);
        if (options.showHelp) {
            printUsage(std::cout);
            return 0;
        }
        return run(options);
    } catch (std::exception const& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
