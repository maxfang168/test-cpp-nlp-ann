#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <stdexcept>

// Random number generator (global for reuse)
std::mt19937 rng(std::random_device{}());

// Function to generate a random number within a range
int randomNumber(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

// Function to run the artificial neural network (ANN)
void runANN(const std::vector<long double>& inputWeights,
            const std::vector<long double>& inputBiases,
            const std::vector<long double>& outputWeights,
            const std::vector<long double>& outputBiases,
            size_t vocabSize,
            const std::vector<int>& contextIndices,
            std::vector<long double>& layer1Values,
            std::vector<long double>& outputValues) {
    // Ensure vectors are properly sized beforehand
    size_t inputLayerSize = vocabSize * 3;
    size_t outputLayerSize = vocabSize;

    // Initialize input values
    std::vector<long double> inputValues(inputLayerSize, 0.0);
    for (size_t i = 0; i < contextIndices.size(); ++i) {
        size_t idx = contextIndices[i] + i * vocabSize;
        if (idx < inputLayerSize) {
            inputValues[idx] = 1.0; // Marker value
        }
    }

    // Compute layer 1 values
    #pragma omp parallel for
    for (size_t i = 0; i < inputLayerSize; ++i) {
        layer1Values[i] = inputValues[i] * inputWeights[i] + inputBiases[i];
        // Apply ReLU activation
        if (layer1Values[i] < 0) {
            layer1Values[i] = 0;
        }
    }

    // Compute output layer values
    #pragma omp parallel for
    for (size_t i = 0; i < outputLayerSize; ++i) {
        outputValues[i] = 0.0;
        for (size_t j = 0; j < inputLayerSize; ++j) {
            outputValues[i] += layer1Values[j] * outputWeights[i * inputLayerSize + j];
        }
        outputValues[i] += outputBiases[i];
        // Apply ReLU activation
        if (outputValues[i] < 0) {
            outputValues[i] = 0;
        }
    }

    // Find the maximum output value and its index
    auto maxElementIt = std::max_element(outputValues.begin(), outputValues.end());
    size_t maxElementIndex = std::distance(outputValues.begin(), maxElementIt);

    // Print the result
    std::cout << "Predicted token index: " << maxElementIndex << " with value: " << *maxElementIt << std::endl;
}

int main() {
    size_t vocabSize = 20000; // Vocabulary size

    // Initialize weights and biases
    std::vector<long double> inputWeights(vocabSize * 3, 0.5); // Example initialization
    std::vector<long double> inputBiases(vocabSize * 3, 0.1);  // Example initialization
    std::vector<long double> outputWeights(vocabSize * vocabSize * 3, 0.01); // Example initialization
    std::vector<long double> outputBiases(vocabSize, 0.2); // Example initialization

    // Context token indices (example)
    std::vector<int> contextIndices = {randomNumber(0, vocabSize - 1), randomNumber(0, vocabSize - 1), randomNumber(0, vocabSize - 1)};

    // Pre-allocate layer values to avoid resizing during execution
    std::vector<long double> layer1Values(vocabSize * 3, 0.0);
    std::vector<long double> outputValues(vocabSize, 0.0);

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Run the ANN
    runANN(inputWeights, inputBiases, outputWeights, outputBiases, vocabSize, contextIndices, layer1Values, outputValues);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Execution time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    return 0;
}
