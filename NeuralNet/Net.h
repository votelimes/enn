#pragma once

#include <vector>
#include <random>
#include <ctime>


 class NeuralNet
{

    public : std::vector<std::vector<double>> nodesWeights;
	
    public : NeuralNet(size_t inputNodesCount, size_t hiddenNodesCount, size_t outputNodesCount, size_t hiddenLayersCount);



};

