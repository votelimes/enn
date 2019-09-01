#include "Net.h"

	
	//Kernel class:
	ann::NeuralNet::NeuralNet(const size_t inputNodesCount, const size_t hiddenNodesCount, const size_t outputNodesCount, const size_t hiddenLayersCount) // //Kernel constructor, hiddenNodesCount should be the largest
	{
		//Fill variables:
		
		this->nodesCount.setInputNodesCount(inputNodesCount);
		this->nodesCount.setHiddenNodesCount(hiddenNodesCount);
		this->nodesCount.setOutputNodesCount(outputNodesCount);
		this->nodesCount.setHiddenLayersCount(hiddenLayersCount);
		this->learningRate = 0.1;
		
		//Create and normalize arrays
		this->nodesWeights = new std::vector<std::vector<std::vector<double>>>(hiddenLayersCount + 1, std::vector<std::vector<double>>(hiddenNodesCount, std::vector<double>(hiddenNodesCount, 0)));
		this->nodesValues = new std::vector<std::vector<double>>(hiddenLayersCount + 2, std::vector<double>(hiddenNodesCount, 0));
		this->nodesErrorValues = new std::vector<std::vector<double>>(hiddenLayersCount + 1, std::vector<double>(hiddenNodesCount, 0));

		(*nodesValues)[0].resize(inputNodesCount, 0);
		(*nodesValues)[0].shrink_to_fit();
		(*nodesValues)[this->nodesCount.getHiddenLayersCount() + 1].resize(outputNodesCount, 0);
		(*nodesValues)[this->nodesCount.getHiddenLayersCount() + 1].shrink_to_fit();

		(*nodesErrorValues)[this->nodesCount.getHiddenLayersCount()].resize(outputNodesCount, 0);
		(*nodesErrorValues)[this->nodesCount.getHiddenLayersCount()].shrink_to_fit();

		(*nodesWeights)[0].resize(inputNodesCount);
		(*nodesWeights)[0].shrink_to_fit();
		//
		for (size_t i = 0; i < (*nodesWeights)[this->nodesCount.getHiddenLayersCount()].size(); i++)
		{
			(*nodesWeights)[this->nodesCount.getHiddenLayersCount()][i].resize(outputNodesCount, afunctions::RandomFunc(0.0, 1.0));
			(*nodesWeights)[this->nodesCount.getHiddenLayersCount()][i].shrink_to_fit();
		}
		//Weights initialization(random values) cicles:
		
		concurrency::parallel_for(size_t(0), this->nodesCount.getHiddenLayersCount() + 1, [&](size_t i)
			//for (size_t i = 0; i < this->nodesCount.getHiddenLayersCount() + 1; i++)
			{
				for (size_t k = 0; k < (*nodesWeights)[i].size(); k++)
				{
					for (size_t j = 0; j < (*nodesWeights)[i][k].size(); j++)
					{
						(*nodesWeights)[i][k][j] = afunctions::RandomFunc(0.0, 1.0);
					}
				}
			});
		//
	} // //Kernel constructor

	__int64 ann::NeuralNet::readWeightsFromFile(const std::string weightsStorageFileName) //returns 2 if layers nodes count does not match, returns 1 if file can not be open, 0 if it opens
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
	__int64 ann::NeuralNet::writeWeightsToFile(const std::string weightsStorageFileName) const // returns 1 if file can not be open, 0 if it opens
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
	
	__int64 ann::NeuralNet::studyNetworkAuto(const std::string& fileName)
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
			for (size_t i = 0; i < (*nodesValues)[(*nodesValues).size() - 1].size(); i++)
			{
				ifs.read((char*)& tmp, sizeof(double));
				(*nodesErrorValues)[(*nodesErrorValues).size() - 1][i] = tmp - (*nodesValues)[(*nodesValues).size() - 1][i];
			}
			tmp = 0;

			//Calculate error procent for all other layers:

			//concurrency::parallel_for(((*nodesValues).size() - 2), (size_t)0, [&](size_t i)
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
	__int64 ann::NeuralNet::studyNetworkFileMT(const std::string& fileName)
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
			//Get input data from file
			std::vector<double> inputData;
			for (size_t u = 0; u < rww.getInputNodesCount(); u++)
			{
				ifs.read((char*)& tmp, sizeof(double));
				inputData.push_back(tmp);
			}
			//Set input data to network
			this->setData(inputData);
			//Feed forward
			this->feedForward();
			//Get expected values from file
			std::vector<double> expectedValues;
			
			for (size_t i = 0; i < rww.getOutputNodesCount(); i++)
			{
				ifs.read((char*)& tmp, sizeof(double));
				expectedValues.push_back(tmp);
			}
			//Calculate error procent for all layers
			this->feedBack(expectedValues);

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

	std::vector<double>* ann::NeuralNet::produceResult(const std::vector<double>& inputValues)
	{
		this->setData(inputValues);
		this->feedForward();

		static std::vector<double>* outputValues = new std::vector<double>;

		for (size_t i = 0; i < (*nodesValues)[this->nodesCount.getTotalLayersCount() - 1].size(); i++)
		{
			outputValues->push_back((*nodesValues)[this->nodesCount.getTotalLayersCount() - 1][i]);
		}

		return outputValues;
	}
	
	void ann::NeuralNet::feedForward()
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
	void ann::NeuralNet::feedBack(const std::vector<double>& expValues)
	{
		//Calc output layer errors
		for (size_t i = 0; i < this->nodesCount.getOutputNodesCount(); i++)
		{
			(*nodesErrorValues)[this->nodesCount.getTotalLayersCount() - 1][i] = expValues[i] - (*nodesValues)[this->nodesCount.getTotalLayersCount() - 1][i];
		}
		//Calc all other layers errors

		for (size_t i = this->nodesCount.getTotalLayersCount() - 2; i > 0; i--)
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
		}

	}
	void ann::NeuralNet::weightsReadjustment()
	{
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

	__int64 ann::NeuralNet::setData(const std::vector<double> &inputData) // Return value is difference between network input layer size() and input data size();
	{
		
		if (inputData.size() != this->nodesCount.getInputNodesCount()) return (__int64)this->nodesCount.getInputNodesCount() - inputData.size();

		//Core part:
		for (size_t i = 0; i < (*nodesValues)[0].size() && i < inputData.size(); i++)
		{
			(*nodesValues)[0][i] = inputData[i];
		}
		return 0;
	}
	
	void ann::NeuralNet::reinitializeWeights(const double lowerLimit, const double upperLimit)
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
	void ann::NeuralNet::setWeights(const double value)
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
	void ann::NeuralNet::setLearningRate(const double value)
	{
		this->learningRate = value;
	}

	double ann::NeuralNet::getLearningRate() const
	{
		return this->learningRate;
	}
	
	void ann::NeuralNet::printResult() const
	{
		std::cout << "_______________________________________________________" << std::endl;
		for (size_t i = 0; i < this->nodesCount.getOutputNodesCount(); i++)
		{
			std::cout << i + 1 << ". " << (*nodesValues)[this->nodesCount.getHiddenLayersCount() + 1][i] << std::endl;
		}
		std::cout << "_______________________________________________________" << std::endl;
	}
	void ann::NeuralNet::printWeights() const
	{
		
		std::cout.setf(std::ios::fixed);
		for (size_t i = 0; i < (*nodesWeights).size(); i++)
		{
			std::cout << "________________LAYER " << i + 1 << " WEIGHTS________________" << std::endl << std::endl;
			for (size_t j = 0; j < (*nodesWeights)[i].size(); j++)
			{
				std::cout << "________NODE " << j + 1 << " WEIGHTS________" << std::endl << std::endl;
				for (size_t k = 0; k < (*nodesWeights)[i][j].size(); k++)
				{
					std::cout << k + 1 << ". ";
					std::cout << std::setw(15) << std::left << (*nodesWeights)[i][j][k];
				}
				std::cout << std::endl << std::endl;
			}
		}
	}
	

	

	template <class T>
	inline T ann::NeuralNet::activationFunction(const T value, const bool returnDerivativeValueInstead) const
	{
		const double e = 2.718281828459045235360287471352; //euler's number
		//Tanh:
		if (returnDerivativeValueInstead) { 

			return 1 - pow(tanh(value), 2);
		}
		
		return tanh(value);
		//////

		//Gauss:
		/*if (returnDerivativeValueInstead) { 

			return (-2 * (value * pow(e, -pow(value, 2))));
		}

		return pow(e, -pow(value,2));*/
		/////
		//Sin:
		/*if (returnDerivativeValueInstead) {

			return cos(value);
		}

		return sin(value);*/
	}
 
	
	

	//Additional classes: 
	ann::nodesCountStorage::nodesCountStorage()
	{
		this->inputNodesCount = 0;
		this->hiddenNodesCount = 0;
		this->outputNodesCount = 0;
		this->hiddenLayersCount = 0;
		this->totalLayersCount = 0;
	}

	auto ann::nodesCountStorage::operator==(const nodesCountStorage& ex) const
	{
		return (inputNodesCount == ex.inputNodesCount) && (hiddenNodesCount == ex.hiddenNodesCount) && (outputNodesCount == ex.outputNodesCount) && (hiddenLayersCount == ex.hiddenLayersCount);
	}
	bool ann::nodesCountStorage::operator!=(const nodesCountStorage& ex) const
	{
		return (inputNodesCount != ex.inputNodesCount) && (hiddenNodesCount != ex.hiddenNodesCount) && (outputNodesCount != ex.outputNodesCount) && (hiddenLayersCount != ex.hiddenLayersCount);
	}
	
	size_t ann::nodesCountStorage::getInputNodesCount() const
	{
		return this->inputNodesCount;
	}
	size_t ann::nodesCountStorage::getHiddenNodesCount() const
	{
		return this->hiddenNodesCount;
	}
	size_t ann::nodesCountStorage::getOutputNodesCount() const
	{
		return this->outputNodesCount;
	}
	size_t ann::nodesCountStorage::getHiddenLayersCount() const
	{
		return this->hiddenLayersCount;
	}
	size_t ann::nodesCountStorage::getTotalLayersCount() const
	{
		return this->totalLayersCount;
	}
	
	void ann::nodesCountStorage::setInputNodesCount(const size_t value)
	{
		this->inputNodesCount = value;
		this->totalLayersCount = this->hiddenLayersCount + (size_t) 2;
	}
	void ann::nodesCountStorage::setHiddenNodesCount(const size_t value)
	{
		this->hiddenNodesCount = value;
		this->totalLayersCount = this->hiddenLayersCount + (size_t)2;
	}
	void ann::nodesCountStorage::setOutputNodesCount(const size_t value)
	{
		this->outputNodesCount = value;
		this->totalLayersCount = this->hiddenLayersCount + (size_t)2;
	}
	void ann::nodesCountStorage::setHiddenLayersCount(const size_t value)
	{
		this->hiddenLayersCount = value;
		this->totalLayersCount = this->hiddenLayersCount + (size_t)2;
	}

	void ann::nodesCountStorage::print()
	{
		std::cout << "Input nodes: " << this->getInputNodesCount() << "  |  Hidden nodes: " << this->getHiddenNodesCount() << "  |  Output nodes: " << this->getOutputNodesCount() << "  |  Hidden layers: " << this->getHiddenLayersCount() << std::endl;
	}
	
	ann::dataMassiveMaker::dataMassiveMaker()
	{
		this->massiveSize = 0;
		this->inputDataSize = 0;
		this->expectedValuesSize = 0;
	}

	__int64 ann::dataMassiveMaker::printNumbersMassive(std::string fileName) const
	{
		std::ifstream ifs;
		ifs.open(fileName, std::ios::binary);
		
		size_t massiveSize;
		ifs.read((char*)& massiveSize, sizeof(size_t));

		ann::nodesCountStorage ncs;
		ifs.read((char*)& ncs, sizeof(ncs));
		
		ncs.print();

		double value;
		std::cout.setf(std::ios::fixed);
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

	__int64 ann::dataMassiveMaker::evenNumbersMassive(const size_t inputDataSize, const size_t outputDataSize, const size_t massiveSize, const std::string fileName, const __int64 lowerLimit, const __int64 upperLimit) const
	{
		std::ofstream ofs;
		ofs.open(fileName, std::ios::binary);
		if (!ofs.is_open()) { return 1; }

		ofs.write((char*)& massiveSize, sizeof(size_t));
		
		ann::nodesCountStorage ncs;
		ncs.setInputNodesCount(inputDataSize);
		ncs.setOutputNodesCount(outputDataSize);
		ofs.write((char*)& ncs, sizeof(ncs));
		
		for (size_t i = 0; i < massiveSize; i++)
		{
			__int64 varInt{};
			
			varInt = afunctions::RandomFunc(static_cast<__int64>(lowerLimit), static_cast<__int64>(upperLimit));
			
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

