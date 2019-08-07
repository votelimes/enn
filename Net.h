#pragma once

#include <vector>


class NeuralNet
{

	std::vector<std::vector<double>> weight;
	
	NeuralNet(size_t inputNodesCount, size_t hiddenNodesCount, size_t outputNodesCount, size_t hiddenLayersCount);



};

