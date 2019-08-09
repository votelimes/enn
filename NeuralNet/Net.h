#pragma once

#include <vector>
#include <random>
#include <ctime>
#include <iostream>
#include "Windows.h"


 class NeuralNet
{

    //SECTION: DATA

private: std::vector<std::vector<std::vector<double>>> *nodesWeights;

private: std::vector<std::vector<double>> *nodesValues;

private: double learningRate;

private: size_t nodeInputCount;

private: size_t nodeHiddenCount;

private: size_t nodeOutputCount;

private: size_t layerHiddenCount;
    //SECTION: FUNCTIONS

public: NeuralNet(size_t inputNodesCount, size_t hiddenNodesCount, size_t outputNodesCount, size_t hiddenLayersCount);

private: int ProduceResult(double *inputData, double *expectedResult);

private: double ActivationFunction(double &value);

};

 double RandomFunc();

