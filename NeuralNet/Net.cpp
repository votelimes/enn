#include "Net.h"

 NeuralNet::NeuralNet(size_t inputNodesCount, size_t hiddenNodesCount, size_t outputNodesCount, size_t hiddenLayersCount)
{
	std::uniform_real_distribution<double> unif(0, 1);
	
	std::default_random_engine re;
	
	re.seed(time(0));
	
	this->learningRate = 0.1;
	
	nodesWeights.resize(hiddenLayersCount + 2);
	nodesValues.resize(hiddenLayersCount + 2);
	
	nodesWeights[0].resize(inputNodesCount);
	nodesValues[0].resize(inputNodesCount);
	
	nodesWeights[hiddenLayersCount + 1].resize(outputNodesCount);
	nodesValues[hiddenLayersCount + 1].resize(outputNodesCount);


	for (size_t i = 0; i < hiddenLayersCount + 2; i++)
	{
		if (i > 0 && i < hiddenLayersCount + 1) {
			nodesWeights[i].resize(hiddenNodesCount);
			nodesValues[i].resize(hiddenNodesCount);
		}
		
		for (size_t j = 0; j < nodesWeights[i].size(); j++)
		{
			nodesWeights[i][j] = unif(re);
		}
	}
}

 int NeuralNet::ProduceResult(double* inputData, double* expectedResult)
 {
	 double tmp;

	 
	 
	 for (size_t i = 1; i < nodesWeights.size(); i++)
	 {
			
	 }
	 
	 return 0;
 }
