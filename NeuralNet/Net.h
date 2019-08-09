#pragma once

#include <vector>
#include <random>
#include <ctime>


 class NeuralNet
{

    //SECTION: DATA

private: std::vector<std::vector<std::vector<double>>> *nodesWeights;

private: std::vector<std::vector<double>> *nodesValues;

private: double learningRate;

private: double nodeInputCount;

private: double nodeHiddenCount;

private: double nodeOutputCount;

private: double layerHiddenCount;
    //SECTION: FUNCTIONS

public: NeuralNet(size_t inputNodesCount, size_t hiddenNodesCount, size_t outputNodesCount, size_t hiddenLayersCount);

private: int ProduceResult(double *inputData, double *expectedResult);

private: double ActivationFunction(double &value);

};

