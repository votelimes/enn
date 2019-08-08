#pragma once

#include <vector>
#include <random>
#include <ctime>


 class NeuralNet
{

    private : std::vector<std::vector<double>> nodesWeights;

	private: std::vector<std::vector<double>> nodesValues;
	
    public : NeuralNet(size_t inputNodesCount, size_t hiddenNodesCount, size_t outputNodesCount, size_t hiddenLayersCount);



};

