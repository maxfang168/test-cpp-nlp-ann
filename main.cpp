#include <iostream> //possible memory lesak with higher vocab sizes; troubleshoot
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <unordered_set>
#include <random>
#include <chrono>
#include <thread>
#include <algorithm>
#include <functional>
#include <iomanip>
std::vector<std::string> tokens;
std::string fileContent;
std::string processedFileContent;
const std::unordered_set<char> punctuation = {'.', ',', '!', '?', ';', ':', '(', ')', '-', '`', '\'', '\"', '[', ']', '{', '}', '<', '>', '/', '\\', '_', '@', '=', '+', '*', '%', '^', '~', '#', '$', '&', '\n', '\t', static_cast<char>(0x92)};
std::string trainingFilePath = "C:\\Users\\RLS\\Documents\\GitHub\\test-cpp-nlp-ann\\simplified.txt";
std::string prompt;
std::vector<std::string> promptTokens;
int vocabSize = 0;
int contextSize = 1;
std::vector<long long> inputWeights;
std::vector<long long> inputBiases;
std::vector<long long> inputValues;
std::vector<long long> layer1Weights;
std::vector<long long> layer1Biases;
std::vector<long long> layer1Values;
std::vector<long long> outputBiases;
std::vector<long long> outputValues;

void clearVectors() {
    inputWeights.clear();
    inputBiases.clear();
    inputValues.clear();
    layer1Weights.clear();
    layer1Biases.clear();
    layer1Values.clear();
    outputBiases.clear();
    outputValues.clear();
}
std::vector<int> tokenizedText; // Store prompt tokens as indicies
std::vector<int> phraseTokens;
double randomNumber(double minValue, double maxValue)
{
    std::random_device rd;                                                    // Seed for randomness
    std::mt19937 gen(rd());                                                   // Random number generator
    std::uniform_real_distribution<double> dis(minValue * 10, maxValue * 10); // Real number distribution in range [minValue, maxValue]
    double result = dis(gen);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(10) << result;
    result = std::stod(oss.str());
    return result;
}
std::string mapTokentoString(int tokenIndex)
{
    if (tokenIndex >= 0 && tokenIndex < tokens.size())
    {
        return tokens[tokenIndex];
    }
    else
    {
        return "[UNKNOWN]";
    }
}
int toLowercase(std::string &text)
{
    std::transform(text.begin(), text.end(), text.begin(), ::tolower);
    clearVectors();
    return 0;
}
bool isPunctuation(char character)
{
    return punctuation.find(character) != punctuation.end();
}

void clearPromptTokens()
{
    promptTokens.clear();
}
int preprocessText()
{
    std::cout << "Preprocessing text" << std::endl;
    processedFileContent = fileContent;
    toLowercase(processedFileContent);

    processedFileContent.erase(
        std::remove_if(processedFileContent.begin(), processedFileContent.end(), isPunctuation),
        processedFileContent.end());
    std::cout << "Preprocessing completed." << std::endl;
    return 0;
}

int tokenizeText()
{
    std::cout << "Tokenizing text" << std::endl;

    std::istringstream inputStream(processedFileContent);
    std::string segment;
    while (inputStream >> segment)
    {
        tokens.push_back(segment);
    }
    size_t numSegments = tokens.size();
    std::cout << "Number of segments (Duplicates included): " << numSegments << std::endl;
    // Sort and remove duplicates
    std::sort(tokens.begin(), tokens.end());
    // Use unordered_set to remove duplicates
    std::unordered_set<std::string> uniqueTokens(tokens.begin(), tokens.end());
    tokens.assign(uniqueTokens.begin(), uniqueTokens.end());
    std::cout << "Tokenizing completed." << std::endl;

    return 0;
}
void trimTokens()
{
    tokens.erase(
        std::remove_if(tokens.begin(), tokens.end(), [](const std::string &token)
                       { return token.empty(); }),
        tokens.end());
}
std::string printText(std::string textContent)
{
    std::cout << textContent << std::endl;
    return textContent;
}
int segmentPrompt(const std::string &textContent)
{
    int countUnknown = 0;
    std::cout << "Segmenting text" << std::endl;
    std::istringstream inputStream(textContent);
    std::string word;
    std::ostringstream outputStream;
    while (inputStream >> word)
    {
        auto it = std::find(tokens.begin(), tokens.end(), word);
        if (it != tokens.end())
        {
            size_t index = std::distance(tokens.begin(), it);
            outputStream << tokens[index] << "\nTrue.\n";
            tokenizedText.push_back(index);
        }
        else
        {
            outputStream << "False.\n";
            countUnknown += 1;
        }
    }
    std::cout << outputStream.str();
    std::cout << "Segmenting completed." << std::endl;
    return countUnknown;
}
void removeDuplicates(std::vector<std::string> &tokens)
{
    std::unordered_set<std::string> seen;
    auto new_end = std::remove_if(tokens.begin(), tokens.end(), [&seen](const std::string &token)
                                  {
        if (seen.find(token) != seen.end()) {
            return true;
        } else {
            seen.insert(token);
            return false;
        } });
    tokens.erase(new_end, tokens.end());
}
int runANN(int contextTokenIndex1, int contextTokenIndex2, int contextTokenIndex3)
{
    std::cout << "Running ANN" << std::endl;
    layer1Values.clear();
    outputValues.clear();
    inputValues.clear();
    layer1Values.resize(vocabSize*3);
    outputValues.resize(vocabSize*3);
    inputValues.resize(vocabSize*3);
int initMarker = 1; // Marker dictates starting amount in one-hot encoding.
    inputValues[contextTokenIndex1] = initMarker;
    inputValues[(contextTokenIndex2 + vocabSize)] = initMarker;             // mirror to other side of context
    inputValues[(contextTokenIndex3 + vocabSize + vocabSize)] = initMarker; // mirror to other side of context
    for (size_t i = 0; i < vocabSize * 3; ++i)
    {
        inputValues[i] = inputValues[i] * inputWeights[i] + inputBiases[i];
    }
    for (size_t i = 0; i < vocabSize * 3; ++i)
    {
        layer1Values[i] = inputValues[i] * inputWeights[i] + inputBiases[i];
    }
    for (size_t i = 0; i < vocabSize * 3; ++i)
    {
        outputValues[i] = layer1Values[i] * layer1Weights[i] + layer1Biases[i];
    }
    for (size_t j = 0; j < vocabSize * 3; ++j)
    {
        outputValues[j] += layer1Values[j] * layer1Weights[j] + layer1Biases[j];
    }
    for (size_t j = 0; j < vocabSize * 3; ++j)
    {
        outputValues[j] -= outputBiases[j];
    }
    // std::cout << "Unsorted output values:" << std::endl;

    // for (size_t i = 0; i < outputValues.size(); ++i)
    //{
    //     std::cout << "Output Value Index: " << std::to_string(i) << " | " << /outputValues[i] << std::endl;
    // Apply softmax to outputValues
    double sumExp = 0.0;
    for (auto &val : outputValues)
    {
        val = std::exp(val);
        sumExp += val;
    }
    for (auto &val : outputValues)
    {
        val /= sumExp;
    }

    auto maxElementIter = std::max_element(outputValues.begin(), outputValues.end());
    int maxElementIndex = std::distance(outputValues.begin(), maxElementIter);
    return maxElementIndex;
}
std::string runSentence(int firstToken, int secondToken, int thirdToken, int wordLimit)
{
    int output = runANN(firstToken, secondToken, thirdToken);
    phraseTokens.push_back(output);
    for (int i = 2; i < wordLimit; i++)
    {
        output = runANN(phraseTokens[i - 2], phraseTokens[i - 1], phraseTokens[i]);
        phraseTokens.push_back(output);
    }
    std::string outputString;
    for (size_t i = 0; i < wordLimit; ++i)
    {
        outputString += (" " + mapTokentoString(phraseTokens[i]));
    }
    return outputString;
}
int runPrompt()
{
    promptTokens.clear();
    prompt = "";
    std::cout << "Please enter the prompt text: ";
    std::getline(std::cin, prompt);
    std::cout << "Preprocessing prompt text" << std::endl;
    toLowercase(prompt);
    printText(prompt);
    std::string newPrompt = prompt;
    std::transform(newPrompt.begin(), newPrompt.end(), newPrompt.begin(), [](char c) {
        return isPunctuation(c) ? ' ' : c;
    });

    prompt = newPrompt;
    printText(prompt);
    std::cout << "Preprocessing completed." << std::endl;

    std::cout << "Tokenizing prompt text" << std::endl;

    std::istringstream inputStream(prompt);
    std::string segment;
    while (inputStream >> segment)
    {
        promptTokens.push_back(segment);
    }
    size_t numSegments = promptTokens.size();
    std::cout << "Number of segments in prompt (Duplicates included): " << numSegments << std::endl;

    removeDuplicates(promptTokens);
    std::cout << "Tokenizing completed." << std::endl;
    promptTokens.erase(
        std::remove_if(promptTokens.begin(), promptTokens.end(), [](const std::string &token)
                       { return token.empty(); }),
        promptTokens.end());
    // for (size_t i = 0; i < promptTokens.size(); ++i)
    //{
    //     std::cout << "Prompt token Index: " << std::to_string(i) << " | " << promptTokens[i] << std::endl;
    // }
    segmentPrompt(prompt);
    std::cout << "Prompt to tokens completed." << std::endl;
    for (size_t j = 0; j < tokenizedText.size(); ++j)
    {
        std::cout << "Prompt to tokens Index: " << std::to_string(j) << " | " << tokenizedText[j] << std::endl;
    }
    return 0;
}

int main()
{
    std::cout << "Copyright maxfang168, 2024-2025. Prerelease version.";

    std::cout << "Main funct init now." << std::endl;
    std::ifstream file(trainingFilePath); // Open the file
    // initializeWeights();
    if (file.is_open())
    { // Check if the file is open
        std::string line;
        while (std::getline(file, line))
        {                        // Read each line of the file
            fileContent += line; // Add the line to the content string
            fileContent += "\n"; // Add a newline character after each line
        }
        file.close(); // Close the file
    }
    else
    {
        std::cout << "Unable to open file" << std::endl;
        return 1;
    }
    preprocessText();

    tokenizeText();
    trimTokens();
    /*
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        std::cout << "Token Index: " << std::to_string(i) << " | " << tokens[i] << std::endl;
    }
    */
    size_t numSegments = tokens.size();
    std::cout << "Number of segments (No duplicates): " << numSegments << std::endl;
    vocabSize = numSegments; // Set the vocabulary size
    for (size_t i = 0; i < vocabSize; ++i)
    {
        inputWeights.push_back(randomNumber(-1, 1));
        inputBiases.push_back(randomNumber(-1, 1));
        layer1Weights.push_back(randomNumber(-1, 1));
        layer1Biases.push_back(randomNumber(-1, 1));
        outputBiases.push_back(randomNumber(-1, 1));
    }
    std::cout << "Tokenizing completed." << std::endl;

    inputWeights.resize(vocabSize);
    inputBiases.resize(vocabSize);
    inputValues.resize(vocabSize);
    layer1Weights.resize(vocabSize);
    layer1Biases.resize(vocabSize);
    layer1Values.resize(vocabSize);
    outputBiases.resize(vocabSize);
    outputValues.resize(vocabSize);
    bool continueRunning = true;
    while (continueRunning)
    {
        {
            /*
                for (size_t i = 0; i < vocabSize; ++i)
                {
                    inputValues[i] = randomNumber(-1, 1);
                    std::cout << "Input Value: " << std::to_string(i) << " | " << std::to_string(inputValues[i]) << std::endl;
                    layer1Values[i] = randomNumber(-1, 1);
                    std::cout << "Layer 1 Value: " << std::to_string(i) << " | " << std::to_string(layer1Values[i]) << std::endl;
                    outputValues[i] = randomNumber(-1, 1);
                    std::cout << "Output Value: " << std::to_string(i) << " | " << std::to_string(outputValues[i]) << std::endl;
                }
                */
            std::cout << runSentence(0, 1, 2, 400);
            break; // implement better solution later
        }
    }
    return 0;
}
/* DEPRECATED
int mainWithPrompt()
{
    
    Optional boot function. It runs the prompt (single context). Note that this function may not work with more recent updates.
    
    std::cout << "Copyright maxfang168, 2024-2025. Prerelease version.";

    std::cout << "Main funct init now." << std::endl;
    std::ifstream file(trainingFilePath); // Open the file
    // initializeWeights();
    if (file.is_open())
    { // Check if the file is open
        std::string line;
        while (std::getline(file, line))
        {                        // Read each line of the file
            fileContent += line; // Add the line to the content string
            fileContent += "\n"; // Add a newline character after each line
        }
        file.close(); // Close the file
    }
    else
    {
        std::cout << "Unable to open file" << std::endl;
        return 1;
    }
    preprocessText();

    tokenizeText();
    trimTokens();
    
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        std::cout << "Token Index: " << std::to_string(i) << " | " << tokens[i] << std::endl;
    }
    
    size_t numSegments = tokens.size();
    std::cout << "Number of segments (No duplicates): " << numSegments << std::endl;
    vocabSize = numSegments; // Set the vocabulary size
    for (size_t i = 0; i < vocabSize; ++i)
    {
        inputWeights.push_back(randomNumber(-1, 1));
        inputBiases.push_back(randomNumber(-1, 1));
        layer1Weights.push_back(randomNumber(-1, 1));
        layer1Biases.push_back(randomNumber(-1, 1));
        outputBiases.push_back(randomNumber(-1, 1));
    }
    std::cout << "Tokenizing completed." << std::endl;

    inputWeights.resize(vocabSize);
    inputBiases.resize(vocabSize);
    inputValues.resize(vocabSize);
    layer1Weights.resize(vocabSize);
    layer1Biases.resize(vocabSize);
    layer1Values.resize(vocabSize);
    outputBiases.resize(vocabSize);
    outputValues.resize(vocabSize);
    bool continueRunning = true;
    while (continueRunning)
    {
        {
            
                for (size_t i = 0; i < vocabSize; ++i)
                {
                    inputValues[i] = randomNumber(-1, 1);
                    std::cout << "Input Value: " << std::to_string(i) << " | " << std::to_string(inputValues[i]) << std::endl;
                    layer1Values[i] = randomNumber(-1, 1);
                    std::cout << "Layer 1 Value: " << std::to_string(i) << " | " << std::to_string(layer1Values[i]) << std::endl;
                    outputValues[i] = randomNumber(-1, 1);
                    std::cout << "Output Value: " << std::to_string(i) << " | " << std::to_string(outputValues[i]) << std::endl;
                }
                
            tokenizedText.clear();
            runPrompt();
            std::cout << "Done running runPrompt()" << std::endl;
            int result = runANN(tokenizedText[tokenizedText.size() - 1]);
            if (!tokenizedText.empty())
            {

                std::cout << result << std::endl;
                std::cout << "Result: " << mapTokentoString(tokenizedText[tokenizedText.size() - 1]) << " " << mapTokentoString(result) << std::endl;
                std::cout << std::endl
                          << "Done running ANN." << std::endl;
            }
            else
            {
                std::cout << "Tokenized text is empty. Skipping ANN run." << std::endl;
            }
            tokenizedText.shrink_to_fit();
        }
    }
    return 0;
}
*/