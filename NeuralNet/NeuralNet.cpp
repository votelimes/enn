// NeuralNet.cpp : Main function file.
//

#include "NeuralNet.h"
#include "Net.h"

int main()
{
	std::vector<std::vector<double>> examples(examplesSIZE, std::vector<double>(inputNodes, 0));
	std::vector<std::vector<double>> expecteds(examplesSIZE, std::vector<double>(outputNodes, 0));;
	
	NeuralNet newNet(inputNodes, hiddenNodes, outputNodes, hiddenNodesLayers);


}
