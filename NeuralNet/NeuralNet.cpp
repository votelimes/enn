// NeuralNet.cpp : Main function file.
//

#include "NeuralNet.h"

int main()
{
	nnet::NeuralNet Net_1(inputNodes, hiddenNodes, outputNodes, hiddenNodesLayers);

	std::vector<std::vector<double>> examples(examplesSIZE, std::vector<double>(inputNodes, 0));
	std::vector<std::vector<double>> expecteds(examplesSIZE, std::vector<double>(outputNodes, 0));

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
