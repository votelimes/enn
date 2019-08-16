#include "Net.h"

	nnet::NeuralNet::NeuralNet(size_t inputNodesCount, size_t hiddenNodesCount, size_t outputNodesCount, size_t hiddenLayersCount) // hiddenNodesCount should be the largest
	{
		//Fill variables:
		
		this->nodesCount.setInputNodesCount(inputNodesCount);
		this->nodesCount.setHiddenNodesCount(hiddenNodesCount);
		this->nodesCount.setOutputNodesCount(outputNodesCount);
		this->nodesCount.setHiddenLayersCount(hiddenLayersCount);
		this->learningRate = 0.1;
		
		//
		//Create and normalize arrays
		this->nodesWeights = new std::vector<std::vector<std::vector<double>>>(hiddenLayersCount + 1, std::vector<std::vector<double>>(hiddenNodesCount, std::vector<double>(hiddenNodesCount, 0)));
		this->nodesValues = new std::vector<std::vector<double>>(hiddenLayersCount + 2, std::vector<double>(hiddenNodesCount, 0));
		this->nodesErrorValues = new std::vector<std::vector<double>>(hiddenLayersCount + 1, std::vector<double>(hiddenNodesCount, 0));

		(*nodesValues)[0].resize(inputNodesCount, 0);
		(*nodesValues)[0].shrink_to_fit();
		(*nodesValues)[nodesValues->size() - 1].resize(outputNodesCount, 0);
		(*nodesValues)[nodesValues->size() - 1].shrink_to_fit();

		(*nodesErrorValues)[nodesErrorValues->size() - 1].resize(outputNodesCount, 0);
		(*nodesErrorValues)[nodesErrorValues->size() - 1].shrink_to_fit();

		(*nodesWeights)[0].resize(inputNodesCount);
		(*nodesWeights)[0].shrink_to_fit();
		//
		for (size_t i = 0; i < (*nodesWeights)[nodesWeights->size() - 1].size(); i++)
		{
			(*nodesWeights)[nodesWeights->size() - 1][i].resize(outputNodesCount, afunctions::RandomFunc());
			(*nodesWeights)[nodesWeights->size() - 1][i].shrink_to_fit();
		}
		//Weights initialization(random values) cicles:
		for (size_t i = 0; i < (*nodesWeights).size(); i++)
		{
			for (size_t k = 0; k < (*nodesWeights)[i].size(); k++)
			{
				for (size_t j = 0; j < (*nodesWeights)[i][k].size(); j++)
				{
					(*nodesWeights)[i][k][j] = afunctions::RandomFunc();
				}
			}
		}
		//
	}

	int nnet::NeuralNet::readWeightsFromFile(std::string weightsStorageFileName)
	{
		std::ifstream ifs;
		nnet:nodesCountStorage rww;
		std::vector<double> vec;
		ifs.open(weightsStorageFileName, std::ios::binary);
		ifs.read((char*)&rww, sizeof(rww));
		
		return 0;
	}

	int nnet::NeuralNet::writeWeightsToFile(std::string weightsStorageFileName)
	{
		std::ofstream ofs;
		std::vector<double> vec = { 0.123456789, 0.0987654321, 0.3314, 0.015, 0.043014 };

		ofs.open(weightsStorageFileName, std::ios::binary | std::ios::app);
		ofs.write((char*)& this->nodesCount, sizeof(nodesCount));

		/*for (size_t i = 0; i < vec.size(); i++)
		{
			ofs.write((char*)& vec[i], sizeof(double));
		}*/
		ofs.close();
		return 0;
	}

	void nnet::NeuralNet::studyNetwork(std::vector<std::vector<double>>& examplesSet, std::vector<std::vector<double>>& expectedValueslesSet)
	{
		for (size_t i = 0; i < examplesSet.size(); i++)
		{
			setData(examplesSet[i], true);
			forwardPropogation();
			backPropogation(expectedValueslesSet[i], true);
		}
	}

	void nnet::NeuralNet::forwardPropogation()
	{
		double tmp = 0;

		for (size_t i = 0; i < (*nodesWeights).size(); i++)
		{
			for (size_t j = 0; j < (*nodesValues)[i + 1].size(); j++)
			{
				for (size_t k = 0; k < (*nodesWeights)[i].size(); k++)
				{
					tmp = tmp + ((*nodesWeights)[i][k][j] * (*nodesValues)[i][k]);
				}

				(*nodesValues)[i + 1][j] = activationFunction(tmp, false);

				tmp = 0;
			}
		}
	}

	__int64 nnet::NeuralNet::backPropogation(std::vector<double>& expectedValues, bool ignoreWarnings)
	{
		//Options:

		bool flag1 = true;
		if (expectedValues.size() != (*nodesValues)[(*nodesErrorValues).size() - 1].size()) {
			if (!ignoreWarnings) {
				std::cout << "\n Warning! Storage size and expected values data size does not match, errors are possible. " << std::endl;
				std::cin.get();
			}
			flag1 = false;
		}

		//Core part:

		double tmp = 0;

		//Calculate error procent for output layer:

		for (size_t i = 0; i < (*nodesValues)[(*nodesValues).size() - 1].size() && i < expectedValues.size(); i++)
		{
			(*nodesErrorValues)[(*nodesErrorValues).size() - 1][i] = expectedValues[i] - (*nodesValues)[(*nodesValues).size() - 1][i];
		}

		for (size_t i = (*nodesValues).size() - 2; i > 0; i--)
		{
			for (size_t j = 0; j < (*nodesValues)[i].size(); j++)
			{
				for (size_t k = 0; k < (*nodesValues)[i + 1].size(); k++)
				{
					tmp = tmp + ((*nodesWeights)[i][j][k] * (*nodesErrorValues)[i][k]);
				}

				(*nodesErrorValues)[i - 1][j] = tmp;
				tmp = 0;
			}
		} // tmp = tmp + ((*nodesWeights)[i][j][k] * (*nodesErrorValues)[i + 1][k]);

		//Adjust weights:

		for (size_t i = 0; i < (*nodesWeights).size(); i++)
		{
			for (size_t j = 0; j < (*nodesValues)[i + 1].size(); j++)
			{
				for (size_t k = 0; k < (*nodesValues)[i].size(); k++)
				{
					(*nodesWeights)[i][k][j] = (*nodesWeights)[i][k][j] + (learningRate * (*nodesErrorValues)[i][j] * activationFunction((*nodesValues)[i + 1][j], true) * (*nodesValues)[i][k]);
				}
			}
		}//(*nodesWeights)[i][k][j] = (*nodesWeights)[i][k][j] + (learningRate * (*nodesErrorValues)[i][j] * activationFunction((*nodesValues)[i + 1][j], true) * (*nodesValues)[i][j]);

	  //return a differece between expected values collection size and output layer size
		return flag1 ? 0 : (*nodesValues)[0].size() - expectedValues.size();
	}

	__int64 nnet::NeuralNet::setData(std::vector<double>& inputData, bool ignoreWarnings) // Return value is difference between network input layer size() and input data size();
	{
		//Options:
		bool flag1 = true;
		if (inputData.size() != (*nodesValues)[0].size()) {
			if (!ignoreWarnings) {
				std::cout << "\n Warning! Storage size and input data size does not match, errors are possible. " << std::endl;
				std::cin.get();
			}
			flag1 = false;
		}

		//Core part:
		for (size_t i = 0; i < (*nodesValues)[0].size() && i < inputData.size(); i++)
		{
			(*nodesValues)[0][i] = inputData[i];
		}
		return flag1 ? 0 : (*nodesValues)[0].size() - inputData.size();
	}

	template <class T>
	T nnet::NeuralNet::activationFunction(T value, bool returnDerivativeValueInstead)
	{
		if (returnDerivativeValueInstead) {

			return 1 - pow(tanh(value), 2);
		}

		return tanh(value);
	}
 
	
	
	nnet::nodesCountStorage::nodesCountStorage()
	{
		this->inputNodesCount = 0;
		this->hiddenNodesCount = 0;
		this->outputNodesCount = 0;
		this->hiddenLayersCount = 0;
		this->classFlag = 1.55;
	}

	size_t nnet::nodesCountStorage::getInputNodesCount()
	{
		return this->inputNodesCount;
	}

	size_t nnet::nodesCountStorage::getHiddenNodesCount()
	{
		return this->hiddenNodesCount;
	}

	size_t nnet::nodesCountStorage::getOutputNodesCount()
	{
		return this->outputNodesCount;
	}

	size_t nnet::nodesCountStorage::getHiddenLayersCount()
	{
		return this->hiddenLayersCount;
	}

	
	void nnet::nodesCountStorage::setInputNodesCount(size_t value)
	{
		this->inputNodesCount = value;
	}

	void nnet::nodesCountStorage::setHiddenNodesCount(size_t value)
	{
		this->hiddenNodesCount = value;
	}

	void nnet::nodesCountStorage::setOutputNodesCount(size_t value)
	{
		this->outputNodesCount = value;
	}

	void nnet::nodesCountStorage::setHiddenLayersCount(size_t value)
	{
		this->hiddenLayersCount = value;
	}
	
	
	
	double afunctions::RandomFunc()
 {
	 double rv;
	 std::random_device rd;
	 std::mt19937 gen(rd());
	 std::uniform_real_distribution<double> uid(0.0, 1.0);
	 rv = uid(gen);
	 return rv > 0 ? rv = uid(gen) : rv;
 }


