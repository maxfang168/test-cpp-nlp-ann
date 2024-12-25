#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <vector>
#include <sstream>
std::vector<std::string> tokens;
std::string fileName = "trainingData.txt";
std::string fileContent;
std::string processedFileContent;
std::string punctuation[34] = {".", ",", "!", "?", ";", ":", "(", ")", "-", "`", "'", "\"", "[", "]", "{", "}", "<", ">", "/", "\\", "_", "@", "=", "+", "*", "%", "^", "~", "#", "$", "&", "\n", "\t", "’"};
std::string trainingFilePath = "C:\\Users\\RLS\\Documents\\GitHub\\test-cpp-nlp-ann\\trainingData.txt";
int vocabSize;
std::vector<long long> layer1Weights;
std::vector<long long> layer1Biases;
std::vector<long long> layer1Values;

std::string temporaryMemory; // Used in the preprocessing stage

int lowercase(std::string &text)
{
    std::transform(text.begin(), text.end(), text.begin(), ::tolower);
    return 0;
}
bool isPunctuation(std::string character)
{
    return std::find(std::begin(punctuation), std::end(punctuation), character) != std::end(punctuation);
}

int preprocesstext()
{
    std::cout << "Preprocessing text" <<std::endl;
    processedFileContent = fileContent;
    lowercase(processedFileContent);
    
    for (int i = 0; i < processedFileContent.length(); i++)
    {
        if (isPunctuation(std::string(1, processedFileContent[i])))
        {
            processedFileContent.erase(i, 1);
            processedFileContent.insert(i, " ");
        
        }
    }
    std::cout << "Preprocessing completed." << std::endl;
    return 0;
}

int tokenize()
{
    std::cout << "Tokenizing text" << std::endl;

    std::istringstream iss(processedFileContent);
    std::string segment;
    while (iss >> segment)
    {
        tokens.push_back(segment);
    }
    size_t numSegments = tokens.size();
    std::cout << "Number of segments (Duplicates included): " << numSegments << std::endl;
    std::sort(tokens.begin(), tokens.end());
    numSegments = std::unique(tokens.begin(), tokens.end()) - tokens.begin();
    std::cout << "Number of segments (No duplicates): " << numSegments << std::endl;
    vocabSize = numSegments; // Set the vocabulary size
    std::cout << "Tokenizing completed." << std::endl;

   return 0;
}

int main()
{
    std::cout << "Main funct init.";
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
    preprocesstext();

    tokenize();
    return 0;
}