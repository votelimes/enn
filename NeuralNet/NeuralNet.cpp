// NeuralNet.cpp : Main function file.
//

#include "NeuralNet.h"
#include "Net.h"

int main()
{
	std::vector<std::vector<double>> examples(10000, std::vector<double>(2, 0));
	NeuralNet newNet(inputNodes, hiddenNodes, outputNodes, hiddenNodesLayers);


}
