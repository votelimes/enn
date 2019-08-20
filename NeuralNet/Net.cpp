#include "Net.h"

	
	//Kernel class:
	nnet::NeuralNet::NeuralNet(size_t inputNodesCount, size_t hiddenNodesCount, size_t outputNodesCount, size_t hiddenLayersCount) // //Kernel constructor, hiddenNodesCount should be the largest
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
			(*nodesWeights)[nodesWeights->size() - 1][i].resize(outputNodesCount, afunctions::RandomFunc(0.0, 1.0));
			(*nodesWeights)[nodesWeights->size() - 1][i].shrink_to_fit();
		}
		//Weights initialization(random values) cicles:
		for (size_t i = 0; i < (*nodesWeights).size(); i++)
		{
			for (size_t k = 0; k < (*nodesWeights)[i].size(); k++)
			{
				for (size_t j = 0; j < (*nodesWeights)[i][k].size(); j++)
				{
					(*nodesWeights)[i][k][j] = afunctions::RandomFunc(0.0, 1.0);
				}
			}
		}
		//
	} // //Kernel constructor

	__int64 nnet::NeuralNet::readWeightsFromFile(std::string weightsStorageFileName) //returns 2 if layers nodes count does not match, returns 1 if file can not be open, 0 if it opens
	{
		std::ifstream ifs;
		nodesCountStorage rww;

		ifs.open(weightsStorageFileName, std::ios::binary);
		if (!ifs.is_open()) {
			return 1;
		}
		
		ifs.read((char*)&rww, sizeof(rww));
		if (rww != this->nodesCount) {
			return 2;
		}

		for (size_t i = 0; i < this->nodesCount.getHiddenLayersCount() + 1; i++)
		{
			for (size_t j = 0; j < (*nodesValues)[i].size(); j++)
			{
				for (size_t k = 0; k < (*nodesValues)[i + 1].size(); k++)
				{
					ifs.read((char*) & (*nodesWeights)[i][j][k], sizeof(double));
				}
			}
		}
		
		return 0;
	}
	__int64 nnet::NeuralNet::writeWeightsToFile(std::string weightsStorageFileName) const // returns 1 if file can not be open, 0 if it opens
	{
		std::ofstream ofs;

		ofs.open(weightsStorageFileName, std::ios::binary);
		
		if (!ofs.is_open()) {
			return 1;
		}
		
		ofs.write((char*)& this->nodesCount, sizeof(this->nodesCount));
		
		for (size_t i = 0; i < this->nodesCount.getHiddenLayersCount() + 1; i++)
		{
			for (size_t j = 0; j < (*nodesValues)[i].size(); j++)
			{
				for (size_t k = 0; k < (*nodesValues)[i + 1].size(); k++)
				{
					ofs.write((char*)& (*nodesWeights)[i][j][k], sizeof(double));
				}	
			}
		}
		
		ofs.close();
		return 0;
	}
	
	void nnet::NeuralNet::studyNetworkManual(std::vector<std::vector<double>>& examplesSet, std::vector<std::vector<double>>& expectedValueslesSet)
	{
		for (size_t i = 0; i < examplesSet.size(); i++)
		{
			setData(examplesSet[i], true);
			forwardPropogationManual();
			backPropogationManual(expectedValueslesSet[i], true);
		}
	}
	__int64 nnet::NeuralNet::studyNetworkAuto(std::string& fileName)
	{
		std::ifstream ifs;
		nodesCountStorage rww;
		size_t dataMassiveSize;
		ifs.open(fileName, std::ios::binary);
		if (!ifs.is_open()) { return 1; }

		ifs.read((char*)& dataMassiveSize, sizeof(size_t)); //1st line read count of examples

		ifs.read((char*)& rww, sizeof(rww)); //2nd line read network count storage class
		if (rww.getInputNodesCount() != this->nodesCount.getInputNodesCount() && rww.getOutputNodesCount() != this->nodesCount.getOutputNodesCount()) { return 2; }

		for (size_t y = 0; y < dataMassiveSize; y++)
		{
			double tmp;
			//input layer initialization
			for (size_t u = 0; u < rww.getInputNodesCount(); u++)
			{
				ifs.read((char*)& tmp, sizeof(double));
				(*nodesValues)[0][u] = tmp;
			}
			tmp = 0;
			//forward propogation:
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
			////Calculate error procent for output layer:
			tmp = 0;
			for (size_t i = 0; i < (*nodesValues)[(*nodesValues).size() - 1].size(); i++)
			{
				ifs.read((char*)& tmp, sizeof(double));
				(*nodesErrorValues)[(*nodesErrorValues).size() - 1][i] = tmp - (*nodesValues)[(*nodesValues).size() - 1][i];
			}
			tmp = 0;
			//Calculate error procent for all other layers:
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
			}
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
			}

		}


		ifs.close();
		return 0;

	}
	void nnet::NeuralNet::forwardPropogationManual()
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
	__int64 nnet::NeuralNet::backPropogationManual(std::vector<double>& expectedValues, bool ignoreWarnings)
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

		//Calculate error procent for all other layers
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
	void nnet::NeuralNet::setWeights(double value)
	{
		for (size_t i = 0; i < this->nodesCount.getHiddenLayersCount() + 1; i++)
		{
			for (size_t j = 0; j < (*nodesValues)[i].size(); j++)
			{
				for (size_t k = 0; k < (*nodesValues)[i + 1].size(); k++)
				{
					(*nodesWeights)[i][j][k] = value;
				}
			}
		}
	}
	void nnet::NeuralNet::setLearningRate(double value)
	{
		this->learningRate = value;
	}

	double nnet::NeuralNet::getLearningRate() const
	{
		return this->learningRate;
	}

	void nnet::NeuralNet::reinitializeWeights(double lowerLimit, double upperLimit)
	{
		for (size_t i = 0; i < this->nodesCount.getHiddenLayersCount() + 1; i++)
		{
			for (size_t j = 0; j < (*nodesValues)[i].size(); j++)
			{
				for (size_t k = 0; k < (*nodesValues)[i + 1].size(); k++)
				{
					(*nodesWeights)[i][j][k] = afunctions::RandomFunc(lowerLimit, upperLimit);
				}
			}
		}
	}

	template <class T>
	T nnet::NeuralNet::activationFunction(T value, bool returnDerivativeValueInstead) const
	{
		if (returnDerivativeValueInstead) {

			return 1 - pow(tanh(value), 2);
		}

		return tanh(value);
	}
 
	
	

	//Additional classes: 
	nnet::nodesCountStorage::nodesCountStorage()
	{
		this->inputNodesCount = 0;
		this->hiddenNodesCount = 0;
		this->outputNodesCount = 0;
		this->hiddenLayersCount = 0;
	}

	auto nnet::nodesCountStorage::operator==(const nodesCountStorage& ex) const
	{
		return (inputNodesCount == ex.inputNodesCount) && (hiddenNodesCount == ex.hiddenNodesCount) && (outputNodesCount == ex.outputNodesCount) && (hiddenLayersCount == ex.hiddenLayersCount);
	}
	bool nnet::nodesCountStorage::operator!=(const nodesCountStorage& ex) const
	{
		return (inputNodesCount != ex.inputNodesCount) && (hiddenNodesCount != ex.hiddenNodesCount) && (outputNodesCount != ex.outputNodesCount) && (hiddenLayersCount != ex.hiddenLayersCount);
	}
	
	size_t nnet::nodesCountStorage::getInputNodesCount() const
	{
		return this->inputNodesCount;
	}
	size_t nnet::nodesCountStorage::getHiddenNodesCount() const
	{
		return this->hiddenNodesCount;
	}
	size_t nnet::nodesCountStorage::getOutputNodesCount() const
	{
		return this->outputNodesCount;
	}
	size_t nnet::nodesCountStorage::getHiddenLayersCount() const
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
	
	nnet::dataMassiveMaker::dataMassiveMaker()
	{
		this->massiveSize = 0;
		this->inputDataSize = 0;
		this->expectedValuesSize = 0;
	}

	__int64 nnet::dataMassiveMaker::evenNumbersMassive(size_t inputDataSize, size_t outputDataSize, size_t massiveSize, std::string &fileName)
	{
		std::ofstream ofs;
		std::vector<double> range;
		nnet::nodesCountStorage ncs;
		__int64 var = 0;

		ncs.setInputNodesCount(inputDataSize);
		ncs.setOutputNodesCount(outputDataSize);
		
		ofs.open(fileName, std::ios::binary);
		if (!ofs.is_open()) { return 1; }

		ofs.write((char*)& massiveSize, sizeof(size_t));
		ofs.write((char*)& ncs, sizeof(ncs));
		
		for (size_t i = 0; i < massiveSize; i++)
		{
			var = afunctions::RandomFunc(static_cast<__int64>(1), static_cast<__int64>(1000000));
			ofs.write((char*)& var, sizeof(double));
			if (var % 2 == 0) { 
				double if1 = 1.0;
				ofs.write((char*)& (if1), sizeof(double)); 
			}
			else {
				double if1 = 0.5;
				ofs.write((char*) & (if1), sizeof(double));
			}
		}
		
		
		return 0;
	}
	
	//Additional functions:
	inline double afunctions::RandomFunc(double lowerLimit, double upperLimit)
 {
	 double rv;
	 std::random_device rd;
	 std::mt19937 gen(rd());
	 std::uniform_real_distribution<double> uid(lowerLimit, upperLimit);
	 rv = uid(gen);
	 return rv > 0 ? rv = uid(gen) : rv;
 }
	inline __int64 afunctions::RandomFunc(__int64 lowerLimit, __int64 upperLimit)
 {
	__int64 rv;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<__int64> uid(lowerLimit, upperLimit);
	rv = uid(gen);
	return rv > 0 ? rv = uid(gen) : rv;
 }

