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
	commandsList.push_back("help");
	while (true)
	{
		system("clear");
		std::cin >> command;
		if (command == commandsList[0]) {
			for (size_t i = 0; i < commandsList.size(); i++)
			{
				std::cout << std::setw(15) << commandsList[i] << commandsDescription[i] << std::endl;
			}
			continue;
		}
	}


	
	/*for (auto i : examples)
	{
		for (auto j : i) 
		{

		}
	}*/
}
