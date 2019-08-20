// NeuralNet.cpp : Main function file.
//

#include "NeuralNet.h"

int main()
{	
	nnet::NeuralNet net1(inputNodes, hiddenNodes, outputNodes, hiddenNodesLayers);
	nnet::dataMassiveMaker maker1;
	std::string dataMassiveEven = "dataMassiveEven.bin";

	maker1.evenNumbersMassive( net1.nodesCount.getInputNodesCount(), net1.nodesCount.getOutputNodesCount(), static_cast<size_t>(10000), dataMassiveEven);

	size_t var = 0;
	std::string command;
	nai::NetApplicationInterface int_1;
	int_1.doWork();

	
	/*for (auto i : examples)
	{
		for (auto j : i) 
		{

		}
	}*/
}
