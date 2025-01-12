#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <functional>

std::vector<std::string> tokens;
std::string fileContent;
std::string processedFileContent;
const std::unordered_set<char> punctuation = {'.', ',', '!', '?', ';', ':', '(', ')', '-', '`', '\'', '\"', '[', ']', '{', '}', '<', '>', '/', '\\', '_', '@', '=', '+', '*', '%', '^', '~', '#', '$', '&', '\n', '\t'};
std::string trainingFilePath = "trainingData.txt";
int vocabSize = 0;
std::vector<long double> inputWeights;
std::vector<long double> inputBiases;
std::vector<long double> inputValues;
std::vector<std::vector<long double>> layerWeights;
std::vector<std::vector<long double>> layerBiases;
std::vector<std::vector<long double>> layerValues;
std::vector<long double> outputBiases;
std::vector<long double> outputValues;

void clearVectors() {
    inputWeights.clear();
    inputBiases.clear();
    inputValues.clear();
    layerWeights.clear();
    layerBiases.clear();
    layerValues.clear();
    outputBiases.clear();
    outputValues.clear();
    inputValues.shrink_to_fit();
    inputBiases.shrink_to_fit();
    inputWeights.shrink_to_fit();
    outputBiases.shrink_to_fit();
    outputValues.shrink_to_fit();
}

bool isPunctuation(char character) {
    return punctuation.find(character) != punctuation.end();
}

void preprocessText() {
    std::cout << "Preprocessing text..." << std::endl;
    processedFileContent = fileContent;
    std::transform(processedFileContent.begin(), processedFileContent.end(), processedFileContent.begin(), ::tolower);
    processedFileContent.erase(
        std::remove_if(processedFileContent.begin(), processedFileContent.end(), isPunctuation),
        processedFileContent.end());
    std::cout << "Text preprocessing complete." << std::endl;
}

void tokenizeText() {
    std::cout << "Tokenizing text..." << std::endl;
    std::istringstream inputStream(processedFileContent);
    std::string segment;
    while (inputStream >> segment) {
        tokens.push_back(segment);
    }
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    vocabSize = tokens.size();
    std::cout << "Tokenization complete. Vocabulary size: " << vocabSize << std::endl;
}

void initializeWeights(int numLayers, int contextWindow) {
    std::cout << "Initializing weights and biases..." << std::endl;
    layerWeights.resize(numLayers, std::vector<long double>(contextWindow * vocabSize));
    layerBiases.resize(numLayers, std::vector<long double>(contextWindow * vocabSize));
    layerValues.resize(numLayers, std::vector<long double>(contextWindow * vocabSize));
    outputBiases.resize(vocabSize);
    outputValues.resize(vocabSize);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<long double> dis(-1.0, 1.0);

    for (int layer = 0; layer < numLayers; ++layer) {
        for (size_t i = 0; i < contextWindow * vocabSize; ++i) {
            layerWeights[layer][i] = dis(gen);
            layerBiases[layer][i] = dis(gen);
        }
    }
    for (size_t i = 0; i < vocabSize; ++i) {
        outputBiases[i] = dis(gen);
    }
    std::cout << "Weights and biases initialized." << std::endl;
}

void runANN(const std::vector<long double>& input) {
    if (input.size() != layerWeights[0].size()) {
        throw std::invalid_argument("Input size does not match ANN configuration.");
    }

    layerValues[0] = input;
    for (size_t layer = 1; layer < layerWeights.size(); ++layer) {
        for (size_t i = 0; i < layerValues[layer].size(); ++i) {
            long double sum = layerBiases[layer][i];
            for (size_t j = 0; j < layerValues[layer - 1].size(); ++j) {
                sum += layerWeights[layer][j] * layerValues[layer - 1][j];
            }
            layerValues[layer][i] = std::max(static_cast<long double>(0), sum); // ReLU activation
        }
    }

    for (size_t i = 0; i < outputValues.size(); ++i) {
        long double sum = outputBiases[i];
        for (size_t j = 0; j < layerValues.back().size(); ++j) {
            sum += layerWeights.back()[j] * layerValues.back()[j];
        }
        outputValues[i] = sum;
    }

    std::cout << "ANN run complete." << std::endl;
}

std::string runSentence(const std::string& prompt, int numTokens, int contextWindow) {
    std::cout << "Generating sentence based on prompt: \"" << prompt << "\"" << std::endl;

    std::vector<long double> context(contextWindow * vocabSize, 0.0);
    std::string generatedText = prompt;

    for (int i = 0; i < numTokens; ++i) {
        runANN(context);

        size_t maxIndex = std::distance(outputValues.begin(), std::max_element(outputValues.begin(), outputValues.end()));
        if (maxIndex < tokens.size()) {
            generatedText += " " + tokens[maxIndex];
        }

        std::rotate(context.begin(), context.begin() + vocabSize, context.end());
        std::fill(context.end() - vocabSize, context.end(), 0.0);
        context[maxIndex] = 1.0; // Update context window with new token
    }

    return generatedText;
}

void loadFileContent(const std::string& filePath) {
    std::ifstream file(filePath);
    if (file.is_open()) {
        std::ostringstream buffer;
        buffer << file.rdbuf();
        fileContent = buffer.str();
        file.close();
    } else {
        throw std::runtime_error("Unable to open file: " + filePath);
    }
}

int main() {
    try {
        std::cout << "Loading training data..." << std::endl;
        loadFileContent(trainingFilePath);
        std::cout << "Training data loaded successfully." << std::endl;

        preprocessText();
        tokenizeText();
        const int numLayers = 12;
        const int contextWindow = 50;
        initializeWeights(numLayers, contextWindow);

        std::cout << "Setup complete. Ready to run the ANN." << std::endl;

        std::string prompt = "Once upon a time";
        std::string generatedText = runSentence(prompt, 20, contextWindow);
        std::cout << "Generated text: " << generatedText << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
    }

    return 0;
}
