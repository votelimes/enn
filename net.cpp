#include "net.h"

	
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
		this->nodes_weights = new std::vector<std::vector<std::vector<double>>>(hiddenLayersCount + 1, std::vector<std::vector<double>>(hiddenNodesCount, std::vector<double>(hiddenNodesCount, 0)));
		this->nodes_values = new std::vector<std::vector<double>>(hiddenLayersCount + 2, std::vector<double>(hiddenNodesCount, 0));
		this->nodes_error_values = new std::vector<std::vector<double>>(hiddenLayersCount + 1, std::vector<double>(hiddenNodesCount, 0));

		(*nodes_values)[0].resize(inputNodesCount, 0);
		(*nodes_values)[0].shrink_to_fit();
		(*nodes_values)[this->nodesCount.getHiddenLayersCount() + 1].resize(outputNodesCount, 0);
		(*nodes_values)[this->nodesCount.getHiddenLayersCount() + 1].shrink_to_fit();

		(*nodes_error_values)[this->nodesCount.getHiddenLayersCount()].resize(outputNodesCount, 0);
		(*nodes_error_values)[this->nodesCount.getHiddenLayersCount()].shrink_to_fit();

		(*nodes_weights)[0].resize(inputNodesCount);
		(*nodes_weights)[0].shrink_to_fit();
		//
		for (size_t i = 0; i < (*nodes_weights)[this->nodesCount.getHiddenLayersCount()].size(); i++)
		{
			(*nodes_weights)[this->nodesCount.getHiddenLayersCount()][i].resize(outputNodesCount, afunctions::RandomFunc(0.0, 1.0));
			(*nodes_weights)[this->nodesCount.getHiddenLayersCount()][i].shrink_to_fit();
		}
		//Weights initialization(random values) cicles:
		
		for (size_t i = 0; i < this->nodesCount.getHiddenLayersCount() + 1; i++)
		{
			for (size_t k = 0; k < (*nodes_weights)[i].size(); k++)
			{
				for (size_t j = 0; j < (*nodes_weights)[i][k].size(); j++)
				{
					(*nodes_weights)[i][k][j] = afunctions::RandomFunc(0.0, 1.0);
				}
			}
		}
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
			for (size_t j = 0; j < (*nodes_values)[i].size(); j++)
			{
				for (size_t k = 0; k < (*nodes_values)[i + 1].size(); k++)
				{
					ifs.read((char*) & var, sizeof(double));
					(*nodes_weights)[i][j][k] =  var;
				}
			}
		}
		return 0;
	}
	__int64 ann::NeuralNet::writeWeightsToFile(const std::string weightsStorageFileName) const // returns 1 if file can not be open, 0 if it opens
	{
		std::ofstream output_file_stream;
		output_file_stream.open(weightsStorageFileName, std::ios::binary);
		
		if (!output_file_stream.is_open()) {
			return 1;
		}
		
		output_file_stream.write((char*)& this->nodesCount, sizeof(this->nodesCount));
		
		for (size_t i = 0; i < this->nodesCount.getHiddenLayersCount() + 1; i++)
		{
			for (size_t j = 0; j < (*nodes_values)[i].size(); j++)
			{
				for (size_t k = 0; k < (*nodes_values)[i + 1].size(); k++)
				{
					output_file_stream.write((char*)& (*nodes_weights)[i][j][k], sizeof(double));
				}	
			}
		}
		return 0;
	}
	
	__int64 ann::NeuralNet::studyNetworkFile(const std::string& fileName)
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
				(*nodes_values)[0][u] = tmp;
			}
			tmp = 0;
			//forward propogation:
			
			for (size_t i = 0; i < (*nodes_weights).size(); i++)
			{
				for (size_t j = 0; j < (*nodes_values)[i + 1].size(); j++)
				{
					for (size_t k = 0; k < (*nodes_weights)[i].size(); k++)
					{
						tmp = tmp + ((*nodes_weights)[i][k][j] * (*nodes_values)[i][k]);
					}

					(*nodes_values)[i + 1][j] = activationFunction(tmp, false);

					tmp = 0;
				}
			}

			////Calculate error procent for output layer:
			for (size_t i = 0; i < (*nodes_values)[(*nodes_values).size() - 1].size(); i++)
			{
				ifs.read((char*)& tmp, sizeof(double));
				(*nodes_error_values)[(*nodes_error_values).size() - 1][i] = tmp - (*nodes_values)[(*nodes_values).size() - 1][i];
			}
			tmp = 0;

			//Calculate error procent for all other layers:

			//concurrency::parallel_for(((*nodes_values).size() - 2), (size_t)0, [&](size_t i)
			for (size_t i = (*nodes_values).size() - 2; i > 0; i--)
			{
				for (size_t j = 0; j < (*nodes_values)[i].size(); j++)
				{
					for (size_t k = 0; k < (*nodes_values)[i + 1].size(); k++)
					{
						tmp = tmp + ((*nodes_weights)[i][j][k] * (*nodes_error_values)[i][k]);
					}

					(*nodes_error_values)[i - 1][j] = tmp;
					tmp = 0;
				}
			}

			//Adjust weights:
			for (size_t i = 0; i < (*nodes_weights).size(); i++)
			{
				for (size_t j = 0; j < (*nodes_values)[i + 1].size(); j++)
				{
					for (size_t k = 0; k < (*nodes_values)[i].size(); k++)
					{
						(*nodes_weights)[i][k][j] = (*nodes_weights)[i][k][j] + (learningRate * (*nodes_error_values)[i][j] * activationFunction((*nodes_values)[i + 1][j], true) * (*nodes_values)[i][k]);
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
			for (size_t i = 0; i < (*nodes_weights).size(); i++)
			{
				for (size_t j = 0; j < (*nodes_values)[i + 1].size(); j++)
				{
					for (size_t k = 0; k < (*nodes_values)[i].size(); k++)
					{
						(*nodes_weights)[i][k][j] = (*nodes_weights)[i][k][j] + (learningRate * (*nodes_error_values)[i][j] * activationFunction((*nodes_values)[i + 1][j], true) * (*nodes_values)[i][k]);
					}
				}
			}

		}
		return 0;
	}
	__int8 ann::NeuralNet::StudyOnce(const std::vector<double> &input_data){
		
		return 0;
	}

	std::vector<double>* ann::NeuralNet::produceResult(const std::vector<double>& inputValues)
	{
		if (this->setData(inputValues)) { return nullptr; }
		
		this->feedForward();

		static std::vector<double>* outputValues = new std::vector<double>;
		for (size_t i = 0; i < (*nodes_values)[this->nodesCount.getTotalLayersCount() - 1].size(); i++)
		{
			outputValues->push_back((*nodes_values)[this->nodesCount.getTotalLayersCount() - 1][i]);
		}

		return outputValues;
	}
	__int64 ann::NeuralNet::produceResult(const std::string inputDataFileName, const std::string outputDataFileName)
	{
		std::ifstream ifs;

		ifs.open(inputDataFileName, std::ios::binary);
		if (!ifs.is_open()) { return 1; }

		size_t dataMassiveSize{};
		ifs.read((char*)& dataMassiveSize, sizeof(size_t)); //1st line read count of examples

		nodesCountStorage rww;
		ifs.read((char*)& rww, sizeof(rww)); //2nd line read network count storage class
		if (rww.getInputNodesCount() != this->nodesCount.getInputNodesCount() && rww.getOutputNodesCount() != this->nodesCount.getOutputNodesCount()) { return 2; }
		
		
		std::ofstream output_file_stream;
		output_file_stream.open(outputDataFileName, std::ios::binary);
		if (!ifs.is_open()) { return 3; }
		output_file_stream.write((char*)& dataMassiveSize, sizeof(size_t));
		output_file_stream.write((char*)& rww, sizeof(rww)); 
		for (size_t i = 0; i < dataMassiveSize; i++)
		{
			
			std::vector<double> inputValues;
			for (size_t j = 0; j < rww.getInputNodesCount(); i++)
			{
				inputValues.push_back(0);
				ifs.read((char*)& inputValues[i], sizeof(double));
			}
			this->setData(inputValues);
			this->feedForward();

			for (size_t j = 0; j < rww.getOutputNodesCount(); i++)
			{
				inputValues.push_back(0);
				output_file_stream.write((char*)& (*nodes_values)[this->nodesCount.getTotalLayersCount() - 1][j], sizeof(double));
			}
		}
		return 0;
	}
	
	void ann::NeuralNet::feedForward()
	{
		double tmp{};

		for (size_t i = 0; i < (*nodes_weights).size(); i++)
		{
			for (size_t j = 0; j < (*nodes_values)[i + 1].size(); j++)
			{
				for (size_t k = 0; k < (*nodes_weights)[i].size(); k++)
				{
					tmp = tmp + ((*nodes_weights)[i][k][j] * (*nodes_values)[i][k]);
				}

				(*nodes_values)[i + 1][j] = activationFunction(tmp, false);

				tmp = 0;
			}
		}
	}
	void ann::NeuralNet::feedBack(const std::vector<double>& expValues)
	{
		//Calc output layer errors
		for (size_t i = 0; i < this->nodesCount.getOutputNodesCount(); i++)
		{
			(*nodes_error_values)[this->nodesCount.getTotalLayersCount() - 2][i] = expValues[i] - (*nodes_values)[this->nodesCount.getTotalLayersCount() - 1][i];
		}
		
		//Calc all other layers errors
		for (size_t i = this->nodesCount.getTotalLayersCount() - 2; i > 0; i--)
		{
			for (size_t j = 0; j < (*nodes_values)[i].size(); j++)
			{
				double tmp{};
				for (size_t k = 0; k < (*nodes_values)[i + 1].size(); k++)
				{
					tmp = tmp + ((*nodes_weights)[i][j][k] * (*nodes_error_values)[i][k]);
				}

				(*nodes_error_values)[i - 1][j] = tmp;
			}
		}

	}
	void ann::NeuralNet::weightsReadjustment()
	{
		for (size_t i = 0; i < (*nodes_weights).size(); i++)
		{
			for (size_t j = 0; j < (*nodes_values)[i + 1].size(); j++)
			{
				for (size_t k = 0; k < (*nodes_values)[i].size(); k++)
				{
					(*nodes_weights)[i][k][j] = (*nodes_weights)[i][k][j] + (learningRate * (*nodes_error_values)[i][j] * activationFunction((*nodes_values)[i + 1][j], true) * (*nodes_values)[i][k]);
				}
			}
		}
	}

	__int64 ann::NeuralNet::setData(const std::vector<double> &inputData) // Return value is difference between network input layer size() and input data size();
	{
		
		if (inputData.size() != this->nodesCount.getInputNodesCount()) return (__int64)this->nodesCount.getInputNodesCount() - inputData.size();

		//Core part:
		for (size_t i = 0; i < (*nodes_values)[0].size() && i < inputData.size(); i++)
		{
			(*nodes_values)[0][i] = inputData[i];
		}
		return 0;
	}
	
	void ann::NeuralNet::reinitializeWeights(const double lowerLimit, const double upperLimit)
	{
		for (size_t i = 0; i < this->nodesCount.getHiddenLayersCount() + 1; i++)
		{
			for (size_t j = 0; j < (*nodes_values)[i].size(); j++)
			{
				for (size_t k = 0; k < (*nodes_values)[i + 1].size(); k++)
				{
					(*nodes_weights)[i][j][k] = afunctions::RandomFunc(lowerLimit, upperLimit);
				}
			}
		}
	}
	void ann::NeuralNet::setWeights(const double value)
	{
		for (size_t i = 0; i < this->nodesCount.getHiddenLayersCount() + 1; i++)
		{
			for (size_t j = 0; j < (*nodes_values)[i].size(); j++)
			{
				for (size_t k = 0; k < (*nodes_values)[i + 1].size(); k++)
				{
					(*nodes_weights)[i][j][k] = value;
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
			std::cout << i + 1 << ". " << (*nodes_values)[this->nodesCount.getHiddenLayersCount() + 1][i] << std::endl;
		}
		std::cout << "_______________________________________________________" << std::endl;
	}
	void ann::NeuralNet::printWeights() const
	{
		
		std::cout.setf(std::ios::fixed);
		for (size_t i = 0; i < (*nodes_weights).size(); i++)
		{
			std::cout << "________________LAYER " << i + 1 << " WEIGHTS________________" << std::endl << std::endl;
			for (size_t j = 0; j < (*nodes_weights)[i].size(); j++)
			{
				std::cout << "________NODE " << j + 1 << " WEIGHTS________" << std::endl << std::endl;
				for (size_t k = 0; k < (*nodes_weights)[i][j].size(); k++)
				{
					std::cout << k + 1 << ". ";
					std::cout << std::setw(15) << std::left << (*nodes_weights)[i][j][k];
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
		std::ofstream output_file_stream;
		output_file_stream.open(fileName, std::ios::binary);
		if (!output_file_stream.is_open()) { return 1; }

		output_file_stream.write((char*)& massiveSize, sizeof(size_t));
		
		ann::nodesCountStorage ncs;
		ncs.setInputNodesCount(inputDataSize);
		ncs.setOutputNodesCount(outputDataSize);
		output_file_stream.write((char*)& ncs, sizeof(ncs));
		
		for (size_t i = 0; i < massiveSize; i++)
		{
			__int64 varInt{};
			
			varInt = afunctions::RandomFunc(static_cast<__int64>(lowerLimit), static_cast<__int64>(upperLimit));
			
			double if1{};
			if (varInt % 2 == 0) if1 = 1.0;
			else if1 = 0.5;
			
			double varDouble = static_cast<double>(varInt);
			output_file_stream.write((char*)& (varDouble), sizeof(double));
			output_file_stream.write((char*) & (if1), sizeof(double));
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

