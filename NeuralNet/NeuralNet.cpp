// NeuralNet.cpp : Main function file.
//

#include "NeuralNet.h"
#include "Net.h"

int main()
{
	std::vector<double> inpData = { 1, 2, 5 };
	std::vector<double> expValues = { 1, 1, 0 };
	
	NeuralNet newNet(inputNodes, hiddenNodes, outputNodes, hiddenNodesLayers);
	
	std::cout << newNet.setData(inpData, 1) << std::endl;
	newNet.forwardPropogation();
}
