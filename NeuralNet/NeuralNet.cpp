// NeuralNet.cpp : Main function file.
//

#include "NeuralNet.h"

int main()
{
	nnet::NeuralNet Net_1(inputNodes, hiddenNodes, outputNodes, hiddenNodesLayers);

	std::vector<std::vector<double>> examples(examplesSIZE, std::vector<double>(inputNodes, 0));
	std::vector<std::vector<double>> expecteds(examplesSIZE, std::vector<double>(outputNodes, 0));

	size_t var = 0;
	while (true)
	{
		
		std::cout << "Главное меню.\n\n\n";
		std::cout << "Выберите опцию: \n\n";
		std::cout << "1. Сохранить весы." << std::endl;
		std::cin >> var;
		if () {

		}
	}


	
	/*for (auto i : examples)
	{
		for (auto j : i) 
		{

		}
	}*/
}
