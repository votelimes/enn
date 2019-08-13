#pragma once

#include <vector>
#include <random>
#include <ctime>
#include <iostream>
#include "Windows.h"


 class NeuralNet
{

    //SECTION: DATA

private: std::vector<std::vector<std::vector<double>>>* nodesWeights;

private: std::vector<std::vector<double>>* nodesValues;

private: std::vector<std::vector<double>>* nodesErrorValues;

private: double learningRate;

private: size_t nodeInputCount;

private: size_t nodeHiddenCount;

private: size_t nodeOutputCount;

private: size_t layerHiddenCount;
    
	//SECTION: FUNCTIONS

public: NeuralNet(size_t inputNodesCount, size_t hiddenNodesCount, size_t outputNodesCount, size_t hiddenLayersCount);

private: void forwardPropogation();

private: int backPropogation(std::vector<double>& expectedValues, bool ignoreWarnings);

public: void studyNetwork(std::vector<std::vector<double>>& examplesSet, std::vector<std::vector<double>>& expectedValueslesSet);

private: int setData(std::vector<double>& inputData, bool ignoreWarnings); // Return value is difference between network input layer size() and input data size();

private: double activationFunction(double &value, bool returnDerivativereturnDerivativeValueInstead);

};

 double RandomFunc();

