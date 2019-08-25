#include "Net.h"

	
	//Kernel class:
	nnet::NeuralNet::NeuralNet(const size_t inputNodesCount, const size_t hiddenNodesCount, const size_t outputNodesCount, const size_t hiddenLayersCount) // //Kernel constructor, hiddenNodesCount should be the largest
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

	__int64 nnet::NeuralNet::readWeightsFromFile(const std::string weightsStorageFileName) //returns 2 if layers nodes count does not match, returns 1 if file can not be open, 0 if it opens
	{
		std::ifstream ifs;

		ifs.open(weightsStorageFileName, std::ios::binary);
		if (!ifs.is_open()) {
			return 1;
		}

		nodesCountStorage rww;
		ifs.read((char*)&rww, sizeof(rww));
		if (rww != this->nodesCount) {
			return 2;
		}

		double var{};
		for (size_t i = 0; i < this->nodesCount.getHiddenLayersCount() + 1; i++)
		{
			for (size_t j = 0; j < (*nodesValues)[i].size(); j++)
			{
				for (size_t k = 0; k < (*nodesValues)[i + 1].size(); k++)
				{
					ifs.read((char*) & var, sizeof(double));
					(*nodesWeights)[i][j][k] =  var;
				}
			}
		}
		return 0;
	}
	__int64 nnet::NeuralNet::writeWeightsToFile(const std::string weightsStorageFileName) const // returns 1 if file can not be open, 0 if it opens
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
		return 0;
	}
	
	void nnet::NeuralNet::studyNetworkManual(const std::vector<std::vector<double>>& examplesSet, const std::vector<std::vector<double>>& expectedValueslesSet)
	{
		for (size_t i = 0; i < examplesSet.size(); i++)
		{
			setData(examplesSet[i]);
			forwardPropogationManual();
			backPropogationManual(expectedValueslesSet[i]);
		}
	}
	__int64 nnet::NeuralNet::studyNetworkAuto(const std::string& fileName)
	{
		std::ifstream ifs;
	
		ifs.open(fileName, std::ios::binary);
		if (!ifs.is_open()) { return 1; }
		
		size_t dataMassiveSize{};
		ifs.read((char*)& dataMassiveSize, sizeof(size_t)); //1st line read count of examples

		nodesCountStorage rww;
		ifs.read((char*)& rww, sizeof(rww)); //2nd line read network count storage class
		if (rww.getInputNodesCount() != this->nodesCount.getInputNodesCount() && rww.getOutputNodesCount() != this->nodesCount.getOutputNodesCount()) { return 2; }

		for (size_t y = 0; y < dataMassiveSize; y++)
		{
			double tmp{};
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
		return 0;
	}
	void nnet::NeuralNet::forwardPropogationManual()
	{
		double tmp{};

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
	__int64 nnet::NeuralNet::backPropogationManual(const std::vector<double>& expectedValues)
	{
		//:

		if (this->nodesCount.getOutputNodesCount() != expectedValues.size()) return static_cast<__int64>(this->nodesCount.getOutputNodesCount() - expectedValues.size());

		//Core part:

		

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
				double tmp{};

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
		return 0;
	}

	__int64 nnet::NeuralNet::setData(const std::vector<double>& inputData) // Return value is difference between network input layer size() and input data size();
	{
		//Options:
		
		if (inputData.size() != this->nodesCount.getInputNodesCount()) return static_cast<__int64>(this->nodesCount.getInputNodesCount() - inputData.size());

		//Core part:
		for (size_t i = 0; i < (*nodesValues)[0].size() && i < inputData.size(); i++)
		{
			(*nodesValues)[0][i] = inputData[i];
		}
		return 0;
	}
	__int64 nnet::NeuralNet::setData(const std::string fileName)
	{
		std::ifstream ifs;
		ifs.open(fileName, std::ios::binary);
		if (!ifs.is_open()) return 1;

		nnet::nodesCountStorage ncs;
		ifs.read((char*)& ncs, sizeof(ncs));
		if (ncs.getInputNodesCount() != this->nodesCount.getInputNodesCount()) return static_cast<__int64>((this->nodesCount.getInputNodesCount()) - ncs.getInputNodesCount());

		for (size_t i = 0; i < this->nodesCount.getInputNodesCount() && ifs; i++)	
		{
			double tmp;
			ifs.read((char*)& tmp, sizeof(double));
			(*nodesValues)[0][i] = tmp;
		}
		return 0;
	}
	void nnet::NeuralNet::setWeights(const double value)
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
	void nnet::NeuralNet::setLearningRate(const double value)
	{
		this->learningRate = value;
	}

	double nnet::NeuralNet::getLearningRate() const
	{
		return this->learningRate;
	}

	void nnet::NeuralNet::printResult() const
	{
		std::cout << "_______________________________________________________" << std::endl;
		for (size_t i = 0; i < this->nodesCount.getOutputNodesCount(); i++)
		{
			std::cout << i + 1 << ". " << (*nodesValues)[this->nodesCount.getHiddenLayersCount() + 1][i] << std::endl;
		}
		std::cout << "_______________________________________________________" << std::endl;
	}

	void nnet::NeuralNet::reinitializeWeights(const double lowerLimit, const double upperLimit)
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
	inline T nnet::NeuralNet::activationFunction(const T value, const bool returnDerivativeValueInstead) const
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
	
	void nnet::nodesCountStorage::setInputNodesCount(const size_t value)
	{
		this->inputNodesCount = value;
	}
	void nnet::nodesCountStorage::setHiddenNodesCount(const size_t value)
	{
		this->hiddenNodesCount = value;
	}
	void nnet::nodesCountStorage::setOutputNodesCount(const size_t value)
	{
		this->outputNodesCount = value;
	}
	void nnet::nodesCountStorage::setHiddenLayersCount(const size_t value)
	{
		this->hiddenLayersCount = value;
	}

	void nnet::nodesCountStorage::print()
	{
		std::cout << "Input nodes: " << this->getInputNodesCount() << "  |  Hidden nodes: " << this->getHiddenNodesCount() << "  |  Output nodes: " << this->getOutputNodesCount() << "  |  Hidden layers: " << this->getHiddenLayersCount() << std::endl;
	}
	
	nnet::dataMassiveMaker::dataMassiveMaker()
	{
		this->massiveSize = 0;
		this->inputDataSize = 0;
		this->expectedValuesSize = 0;
	}

	__int64 nnet::dataMassiveMaker::printNumbersMassive(std::string fileName) const
	{
		std::ifstream ifs;
		ifs.open(fileName, std::ios::binary);
		
		size_t massiveSize;
		ifs.read((char*)& massiveSize, sizeof(size_t));

		nnet::nodesCountStorage ncs;
		ifs.read((char*)& ncs, sizeof(ncs));
		
		ncs.print();

		std::string str;
		double value;
		std::stringstream strst;
		strst << std::fixed << std::setprecision(15);

		for (size_t i = 0; i < massiveSize; i++)
		{
			std::cout << "\nInput data: " << std::endl;
			for (size_t j = 0; j < ncs.getInputNodesCount(); j++)
			{
				ifs.read((char*)& value, sizeof(double));
				std::cout << std::setw(15) << std::left << value;
			}
			std::cout << "\nExpected value respectevly: " << std::endl;
			for (size_t j = 0; j < ncs.getOutputNodesCount(); j++)
			{
				ifs.read((char*)& value, sizeof(double));
				std::cout << std::setw(15) << std::left << value;
			}
			std::cout << "\n_______________" << std::endl;
		}


		return 0;
	}

	__int64 nnet::dataMassiveMaker::evenNumbersMassive(const size_t inputDataSize, const size_t outputDataSize, const size_t massiveSize, const std::string fileName) const
	{
		std::ofstream ofs;
		ofs.open(fileName, std::ios::binary);
		if (!ofs.is_open()) { return 1; }

		ofs.write((char*)& massiveSize, sizeof(size_t));
		
		nnet::nodesCountStorage ncs;
		ncs.setInputNodesCount(inputDataSize);
		ncs.setOutputNodesCount(outputDataSize);
		ofs.write((char*)& ncs, sizeof(ncs));
		
		for (size_t i = 0; i < massiveSize; i++)
		{
			__int64 varInt{};
			
			varInt = afunctions::RandomFunc(static_cast<__int64>(1), static_cast<__int64>(1000000));
			
			double if1{};
			if (varInt % 2 == 0) if1 = 1.0;
			else if1 = 0.5;
			
			double varDouble = static_cast<double>(varInt);
			ofs.write((char*)& (varDouble), sizeof(double));
			ofs.write((char*) & (if1), sizeof(double));
		}
		
		
		return 0;
	}
	
	//Additional functions:
	inline double afunctions::RandomFunc(const double lowerLimit, const double upperLimit)
 {
	 
	 std::random_device rd;
	 std::mt19937 gen(rd());
	 
	 std::uniform_real_distribution<double> uid(lowerLimit, upperLimit);
	 
	 double rv;
	 rv = uid(gen);
	 return rv > 0 ? rv = uid(gen) : rv;
 }
	inline __int64 afunctions::RandomFunc(const __int64 lowerLimit, const __int64 upperLimit)
 {
	
	std::random_device rd;
	std::mt19937 gen(rd());
	
	std::uniform_int_distribution<__int64> uid(lowerLimit, upperLimit);
	
	__int64 rv;
	rv = uid(gen);
	return rv > 0 ? rv = uid(gen) : rv;
 }

