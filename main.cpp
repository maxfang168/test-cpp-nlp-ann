#include <iostream> //possible memory leak with higher vocab sizes; troubleshoot
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
std::string trainingFilePath = "C:\\Users\\RLS\\Documents\\GitHub\\test-cpp-nlp-ann\\trainingData.txt";
std::string prompt;
std::vector<std::string> promptTokens;
int vocabSize = 0;
int contextSize = 1;
std::vector<long double> inputWeights;
std::vector<long double> inputBiases;
std::vector<long double> inputValues;
std::vector<long double> layer1Weights;
std::vector<long double> layer1Biases;
std::vector<long double> layer1Values;
std::vector<long double> outputBiases;
std::vector<long double> outputValues;

void clearVectors() {
    inputWeights.clear();
    inputBiases.clear();
    inputValues.clear();
    layer1Weights.clear();
    layer1Biases.clear();
    layer1Values.clear();
    outputBiases.clear();
    outputValues.clear();
    inputValues.shrink_to_fit();
    inputBiases.shrink_to_fit();
    inputWeights.shrink_to_fit();
    layer1Biases.shrink_to_fit();
    layer1Values.shrink_to_fit();
    layer1Weights.shrink_to_fit();
    outputBiases.shrink_to_fit();
    outputValues.shrink_to_fit();
}
std::vector<int> tokenizedText; // Store prompt tokens as indicies
std::vector<int> phraseTokens;
double randomNumber(long double minValue, long double maxValue)
{
    std::random_device rd;                                                    // Seed for randomness
    std::mt19937 gen(rd());                                                   // Random number generator
    std::uniform_real_distribution<long double> dis(minValue * 10, maxValue * 10); // Real number distribution in range [minValue, maxValue]
    long double result = dis(gen);
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
    return 0;
}
bool isPunctuation(char character)
{
    return punctuation.find(character) != punctuation.end();
}

void clearPromptTokens()
{
    promptTokens.clear();
    promptTokens.shrink_to_fit();
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
    std::cout << std::endl << std::endl << "Running ANN" << std::endl;

    int initMarker = 1; // Marker dictates starting amount in one-hot encoding.
    inputValues[contextTokenIndex1] = initMarker;
    inputValues[contextTokenIndex2 + vocabSize] = initMarker;             // mirror to other side of context
    inputValues[contextTokenIndex3 + 2 * vocabSize] = initMarker; // mirror to other side of context

    std::cout << "Input values initialized" << std::endl;

        // Combine input and layer1 calculations
        try
    {
        std::cout << "Calculating layer 1 values" << std::endl;
        std::cout << "vocabSize: " << vocabSize << std::endl;
        std::cout << "inputValues size: " << inputValues.size() << std::endl;
        std::cout << "inputWeights size: " << inputWeights.size() << std::endl;
        std::cout << "inputBiases size: " << inputBiases.size() << std::endl;
        std::cout << "layer1Values size: " << layer1Values.size() << std::endl;
    
        for (size_t i = 0; i < ((vocabSize) * 3); ++i) //causing error
        {
            layer1Values[i] = inputValues[i] * inputWeights[i] + inputBiases[i];
        }
        std::cout << "Layer 1 values calculated successfully" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error during layer 1 calculations: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown error during layer 1 calculations" << std::endl;
    }

    // Calculate output values
    for (size_t i = 0; i < vocabSize * 3; ++i)
    {
        outputValues[i] = layer1Values[i] * layer1Weights[i] + layer1Biases[i];
        outputValues[i] += layer1Values[i] * layer1Weights[i] + layer1Biases[i];
    }

    std::cout << "Output values calculated" << std::endl;

    // Sort outputValues and return the largest value
    std::sort(outputValues.begin(), outputValues.end(), std::greater<long double>());

    std::cout << "Output values sorted" << std::endl;

    int maxElementIndex = 0; // Since the largest value is now at the beginning
    std::cout << "Max element index: " << maxElementIndex << std::endl;
    return maxElementIndex;
}

std::string runSentence(int firstToken, int secondToken, int thirdToken, int wordLimit)
{
    std::cout << "Running sentence generation" << std::endl;
    int output = runANN(firstToken, secondToken, thirdToken);
    phraseTokens.push_back(output);
    std::cout << "Initial tokens: " << firstToken << ", " << secondToken << ", " << thirdToken << std::endl;
    std::cout << "Output token 1: " << output << std::endl;
    for (int i = 1; i < wordLimit; i++)
    {
        output = runANN(phraseTokens[i - 1], phraseTokens[i], phraseTokens[i-1]);
        phraseTokens.push_back(output);
        std::cout << "Generated token " << i << ": " << output << std::endl;
    }
    std::string outputString;
    for (size_t i = 0; i < wordLimit; ++i)
    {
        outputString += (" " + mapTokentoString(phraseTokens[i]));
    }
    std::cout << "Generated sentence: " << outputString << std::endl;
    return outputString;
}
int runPrompt()
{
    promptTokens.clear();
    promptTokens.shrink_to_fit();
    prompt = "";
    std::cout << "Please enter the prompt text: ";
    std::getline(std::cin, prompt);
    std::cout << "Preprocessing prompt text" << std::endl;
    toLowercase(prompt);
    printText(prompt);
    std::string newPrompt = prompt;
    std::transform(newPrompt.begin(), newPrompt.end(), newPrompt.begin(), [](char c) {
        return isspace(c) ? ' ' : c;
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

    std::cout << std::endl << "Main funct init now." << std::endl;
    layer1Values.clear();
    outputValues.clear();
    inputValues.clear();
    layer1Values.resize(vocabSize * 3);
    outputValues.resize(vocabSize * 3);
    inputValues.resize(vocabSize * 3);
    inputWeights.resize(vocabSize * 3);
    inputBiases.resize(vocabSize * 3);
    layer1Weights.resize(vocabSize * 3);
    layer1Biases.resize(vocabSize * 3);
    outputBiases.resize(vocabSize * 3);
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
    std::cout << "Number of segments (No duplicates are included): " << numSegments << std::endl;
    vocabSize = numSegments; // Set the vocabulary size
    inputWeights.resize(vocabSize*3);
    inputBiases.resize(vocabSize*3);
    inputValues.resize(vocabSize*3);
    layer1Weights.resize(vocabSize*3);
    layer1Biases.resize(vocabSize*3);
    layer1Values.resize(vocabSize*3);
    outputBiases.resize(vocabSize*3);
    outputValues.resize(vocabSize*3);

    for (size_t i = 0; i < vocabSize*3; ++i)
    {
        inputWeights[i] = randomNumber(-1, 1);
        inputBiases[i] = randomNumber(-1, 1);
        layer1Weights[i] = randomNumber(-1, 1);
        layer1Biases[i] = randomNumber(-1, 1);
        outputBiases[i] = randomNumber(-1, 1);
    }
    std::cout << "Tokenizing completed." << std::endl;
    bool continueRunning = true;
    while (continueRunning)
    {
        std::cout << runSentence(0, 1, 2, 100);
        std::cout << "stop";

        std::string userInput;
        std::cout << "Do you want to continue? (yes/no): ";
        std::getline(std::cin, userInput);
        if (userInput != "yes")
        {
            continueRunning = false;
        }
    }
    return 0;
}