#include "Net.h"

 NeuralNet::NeuralNet(size_t inputNodesCount, size_t hiddenNodesCount, size_t outputNodesCount, size_t hiddenLayersCount) // hiddenNodesCount should be the largest
{

	this->learningRate = 0.1;
	this->nodeInputCount = inputNodesCount;	
	this->nodeHiddenCount = hiddenNodesCount;
	this->nodeOutputCount = outputNodesCount;
	this->layerHiddenCount = hiddenLayersCount;


	this->nodesWeights = new std::vector<std::vector<std::vector<double>>>(hiddenLayersCount + 1, std::vector<std::vector<double>>(hiddenNodesCount, std::vector<double>(hiddenNodesCount, 0)));
	this->nodesValues = new std::vector<std::vector<double>>(hiddenLayersCount+2, std::vector<double>(hiddenNodesCount, 0));

	//Weights initialization(random values) cicles
	for (size_t i = 0; i < (*nodesWeights).size(); i++)
	{
		for (size_t k = 0; k < (*nodesWeights)[i].size(); k++)
		{
			for (size_t j = 0; j < (*nodesWeights)[i][k].size(); j++)
			{
				(*nodesWeights)[i][k][j] = RandomFunc();
			}
		}
	}
		
	(*nodesValues)[0].resize(inputNodesCount);
	(*nodesValues)[0].shrink_to_fit();
	(*nodesValues)[nodesValues->size() - 1].resize(outputNodesCount);
	(*nodesValues)[nodesValues->size() - 1].shrink_to_fit();

	(*nodesWeights)[0].resize(inputNodesCount);
	(*nodesWeights)[0].shrink_to_fit();
	(*nodesWeights)[nodesWeights->size() - 1].resize(outputNodesCount);
	(*nodesWeights)[nodesWeights->size() - 1].shrink_to_fit();

	for (size_t i = 0; i < hiddenLayersCount; i++)
	{	
		(*nodesWeights)[0][i].resize(outputNodesCount);
		(*nodesWeights)[0][i].shrink_to_fit();
		(*nodesWeights)[nodesWeights->size() - 1][i].resize(outputNodesCount);
		(*nodesWeights)[nodesWeights->size() - 1][i].shrink_to_fit();
	}

}

 int NeuralNet::ProduceResult(double* inputData, double* expectedResult)
 {
	 double tmp = 0;
	 
	 
	/* for (size_t i = 0; i < nodesValues.size() - 1; i++)
	 {
		 for (size_t j = 0; j < nodesValues[i].size(); j++)
		 {
			 tmp = tmp + (nodesWeights[i][j] * nodesValues[i][j]);
		 }
	 }*/
	
	 
	 return 0;
 }
 double NeuralNet::ActivationFunction(double &value)
 {
	 return tanh(value);
 }

 double RandomFunc()
 {
	 std::uniform_real_distribution<double> unif(0, 1);

	 std::default_random_engine re;

	 re.seed(time(0));
	 
	 return unif(re);
 }
