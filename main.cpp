#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <vector>
#include <sstream>
std::vector<std::string> tokens;
std::string fileContent;
std::string processedFileContent;
std::string punctuation[34] = {".", ",", "!", "?", ";", ":", "(", ")", "-", "`", "'", "\"", "[", "]", "{", "}", "<", ">", "/", "\\", "_", "@", "=", "+", "*", "%", "^", "~", "#", "$", "&", "\n", "\t", "’"};
std::string trainingFilePath = "C:\\Users\\RLS\\Documents\\GitHub\\test-cpp-nlp-ann\\data.txt";
std::string prompt;
std::vector<std::string> promptTokens;
int vocabSize;
std::vector<long long> layer1Weights;
std::vector<long long> layer1Biases;
std::vector<long long> layer1Values;

int toLowercase(std::string &text)
{
    std::transform(text.begin(), text.end(), text.begin(), ::tolower);
    return 0;
}
bool isPunctuation(char character)
{
    return std::find(std::begin(punctuation), std::end(punctuation), std::string(1, character).c_str()) != std::end(punctuation);
}

int preprocessText()
{
    std::cout << "Preprocessing text" <<std::endl;
    processedFileContent = fileContent;
    toLowercase(processedFileContent);
    
    for (size_t i = 0; i < processedFileContent.size(); ++i) {
        if (isPunctuation(processedFileContent[i])) {
            processedFileContent.erase(i, 1);
            --i;
        }
    }
    std::replace_if(processedFileContent.begin(), processedFileContent.end(), [](char c) {
        return isPunctuation(c);
    }, ' ');
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
    auto last = std::unique(tokens.begin(), tokens.end());
    tokens.erase(last, tokens.end());

    numSegments = tokens.size();
    std::cout << "Tokenizing completed." << std::endl;

    return 0;
}
void trimTokens() 
{
    tokens.erase(
        std::remove_if(tokens.begin(), tokens.end(), [](const std::string& token) {
            return token.empty();
        }),
        tokens.end()
    );
}
std::string printText(std::string textContent)
{
    std::cout << textContent << std::endl;
    return textContent;
}
int runPrompt()
{
    std::cout << "Please enter the prompt text: ";
    std::getline(std::cin, prompt);
    std::cout << "Preprocessing prompt text" <<std::endl;
    toLowercase(prompt);
    printText(prompt);
    std::string newPrompt;
    for (char c : prompt)
    {
        if (isPunctuation(c))
        {
            newPrompt += " ";
        }
        else
        {
            newPrompt += c;
        }
    }
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

    // Sort and remove duplicates
    std::sort(promptTokens.begin(), promptTokens.end());
    auto last = std::unique(promptTokens.begin(), promptTokens.end());
    promptTokens.erase(last, promptTokens.end());
    std::cout << "Tokenizing completed." << std::endl;
    promptTokens.erase(
        std::remove_if(promptTokens.begin(), promptTokens.end(), [](const std::string& token) {
            return token.empty();
        }),
        promptTokens.end()
    );
    for (size_t i = 0; i < promptTokens.size(); ++i)
{
    std::cout << "Prompt token Index: " << std::to_string(i) << " | " << promptTokens[i] << std::endl;
}
    return 0;
}

int main()
{
    std::cout << "Main funct init." << std::endl;
    std::ifstream file(trainingFilePath); // Open the file

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
    fileContent.clear();  // Clear the content
    fileContent.shrink_to_fit();  // Shrink the capacity to release allocated memory
    std::cout << "fileContent has been unloaded from memory." << std::endl;
    processedFileContent.clear();  // Clear the content
    processedFileContent.shrink_to_fit();  // Shrink the capacity to release allocated memory
    std::cout << "processedFileContent has been unloaded from memory." << std::endl;

    for (size_t i = 0; i < tokens.size(); ++i)
{
    std::cout << "Token Index: " << std::to_string(i) << " | " << tokens[i] << std::endl;
}
    size_t numSegments = tokens.size();
    std::cout << "Number of segments (No duplicates): " << numSegments << std::endl;
    vocabSize = numSegments; // Set the vocabulary size
    std::cout << "Tokenizing completed." << std::endl;
    runPrompt();
    return 0;
}