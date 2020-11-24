#include "net.h"

	
	//Kernel class:
	network_core::NeuralNet::NeuralNet(const size_t input_nodes_count, const size_t hidden_nodes_count, const size_t outputNodesCount, const size_t hiddenLayersCount) // //Kernel constructor, hidden_nodes_count should be the largest
	{
		//Fill variables:
		
		this->nodes_count.setInputNodesCount(input_nodes_count);
		this->nodes_count.setHiddenNodesCount(hidden_nodes_count);
		this->nodes_count.setOutputNodesCount(outputNodesCount);
		this->nodes_count.setHiddenLayersCount(hiddenLayersCount);
		this->learning_rate = 0.1;
		
		//Create and normalize arrays
		this->nodes_weights = new std::vector<std::vector<std::vector<double>>>(hiddenLayersCount + 1, std::vector<std::vector<double>>(hidden_nodes_count, std::vector<double>(hidden_nodes_count, 0)));
		this->nodes_values = new std::vector<std::vector<double>>(hiddenLayersCount + 2, std::vector<double>(hidden_nodes_count, 0));
		this->nodes_error_values = new std::vector<std::vector<double>>(hiddenLayersCount + 1, std::vector<double>(hidden_nodes_count, 0));

		(*nodes_values)[0].resize(input_nodes_count, 0);
		(*nodes_values)[0].shrink_to_fit();
		(*nodes_values)[this->nodes_count.getHiddenLayersCount() + 1].resize(outputNodesCount, 0);
		(*nodes_values)[this->nodes_count.getHiddenLayersCount() + 1].shrink_to_fit();

		(*nodes_error_values)[this->nodes_count.getHiddenLayersCount()].resize(outputNodesCount, 0);
		(*nodes_error_values)[this->nodes_count.getHiddenLayersCount()].shrink_to_fit();

		(*nodes_weights)[0].resize(input_nodes_count);
		(*nodes_weights)[0].shrink_to_fit();
		//
		for (size_t i = 0; i < (*nodes_weights)[this->nodes_count.getHiddenLayersCount()].size(); i++)
		{
			(*nodes_weights)[this->nodes_count.getHiddenLayersCount()][i].resize(outputNodesCount, additional_functions::RandomFunction(0.0, 1.0));
			(*nodes_weights)[this->nodes_count.getHiddenLayersCount()][i].shrink_to_fit();
		}
		//Weights initialization(random values) cicles:
		
		for (size_t i = 0; i < this->nodes_count.getHiddenLayersCount() + 1; i++)
		{
			for (size_t k = 0; k < (*nodes_weights)[i].size(); k++)
			{
				for (size_t j = 0; j < (*nodes_weights)[i][k].size(); j++)
				{
					(*nodes_weights)[i][k][j] = additional_functions::RandomFunction(0.0, 1.0);
				}
			}
		}
		//
	} // //Kernel constructor

	__int64 network_core::NeuralNet::ReadWeightsFile(const std::string weights_storage_file_name) //returns 2 if layers nodes count does not match, returns 1 if file can not be open, 0 if it opens
	{
		std::ifstream input_file_stream;

		input_file_stream.open(weights_storage_file_name, std::ios::binary);
		if (!input_file_stream.is_open()) {
			return 1;
		}

		NodesCountStorage rww;
		input_file_stream.read((char*)&rww, sizeof(rww));
		if (rww != this->nodes_count) {
			return 2;
		}

		double var{};
		for (size_t i = 0; i < this->nodes_count.getHiddenLayersCount() + 1; i++)
		{
			for (size_t j = 0; j < (*nodes_values)[i].size(); j++)
			{
				for (size_t k = 0; k < (*nodes_values)[i + 1].size(); k++)
				{
					input_file_stream.read((char*) & var, sizeof(double));
					(*nodes_weights)[i][j][k] =  var;
				}
			}
		}
		return 0;
	}
	__int64 network_core::NeuralNet::WriteWeightsFile(const std::string weights_storage_file_name) const // returns 1 if file can not be open, 0 if it opens
	{
		std::ofstream output_file_stream;
		output_file_stream.open(weights_storage_file_name, std::ios::binary);
		
		if (!output_file_stream.is_open()) {
			return 1;
		}
		
		output_file_stream.write((char*)& this->nodes_count, sizeof(this->nodes_count));
		
		for (size_t i = 0; i < this->nodes_count.getHiddenLayersCount() + 1; i++)
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
	
	__int64 network_core::NeuralNet::StudyFile(const std::string& dataset_file_name)
	{
		std::ifstream input_file_stream;
	
		input_file_stream.open(dataset_file_name, std::ios::binary);
		if (!input_file_stream.is_open()) { return 1; }
		
		size_t input_data_massive_lenght{};
		input_file_stream.read((char*)& input_data_massive_lenght, sizeof(size_t)); //1st line read count of examples

		NodesCountStorage rww;
		input_file_stream.read((char*)& rww, sizeof(rww)); //2nd line read network count storage class
		if (rww.getInputNodesCount() != this->nodes_count.getInputNodesCount() && rww.getOutputNodesCount() != this->nodes_count.getOutputNodesCount()) { return 2; }

		for (size_t y = 0; y < input_data_massive_lenght; y++)
		{
			double tmp{};
			//input layer initialization
			for (size_t u = 0; u < rww.getInputNodesCount(); u++)
			{
				input_file_stream.read((char*)& tmp, sizeof(double));
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
				input_file_stream.read((char*)& tmp, sizeof(double));
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
						(*nodes_weights)[i][k][j] = (*nodes_weights)[i][k][j] + (learning_rate * (*nodes_error_values)[i][j] * activationFunction((*nodes_values)[i + 1][j], true) * (*nodes_values)[i][k]);
					}
				}
			}

		}
		return 0;
	}
	__int64 network_core::NeuralNet::StudyFileMT(const std::string& dataset_file_name)
	{
		std::ifstream input_file_stream;

		input_file_stream.open(dataset_file_name, std::ios::binary);
		if (!input_file_stream.is_open()) { return 1; }

		size_t input_data_massive_lenght{};
		input_file_stream.read((char*)& input_data_massive_lenght, sizeof(size_t)); //1st line read count of examples

		NodesCountStorage rww;
		input_file_stream.read((char*)& rww, sizeof(rww)); //2nd line read network count storage class
		if (rww.getInputNodesCount() != this->nodes_count.getInputNodesCount() && rww.getOutputNodesCount() != this->nodes_count.getOutputNodesCount()) { return 2; }

		for (size_t y = 0; y < input_data_massive_lenght; y++)
		{
			double tmp{};
			//Get input data from file
			std::vector<double> input_data;
			for (size_t u = 0; u < rww.getInputNodesCount(); u++)
			{
				input_file_stream.read((char*)& tmp, sizeof(double));
				input_data.push_back(tmp);
			}
			//Set input data to network
			this->SetData(input_data);
			//Feed forward
			this->FeedForward();
			//Get expected values from file
			std::vector<double> expectedValues;
			
			for (size_t i = 0; i < rww.getOutputNodesCount(); i++)
			{
				input_file_stream.read((char*)& tmp, sizeof(double));
				expectedValues.push_back(tmp);
			}
			//Calculate error procent for all layers
			this->FeedBack(expectedValues);

			//Adjust weights:
			for (size_t i = 0; i < (*nodes_weights).size(); i++)
			{
				for (size_t j = 0; j < (*nodes_values)[i + 1].size(); j++)
				{
					for (size_t k = 0; k < (*nodes_values)[i].size(); k++)
					{
						(*nodes_weights)[i][k][j] = (*nodes_weights)[i][k][j] + (learning_rate * (*nodes_error_values)[i][j] * activationFunction((*nodes_values)[i + 1][j], true) * (*nodes_values)[i][k]);
					}
				}
			}

		}
		return 0;
	}
	__int8 network_core::NeuralNet::StudyOnce(const std::vector<double> &input_data){

		return 0;
	}

	std::vector<double>* network_core::NeuralNet::ProduceResult(const std::vector<double>& input_values)
	{
		if (this->SetData(input_values)) { return nullptr; }
		
		this->FeedForward();

		static std::vector<double>* outputValues = new std::vector<double>;
		for (size_t i = 0; i < (*nodes_values)[this->nodes_count.getTotalLayersCount() - 1].size(); i++)
		{
			outputValues->push_back((*nodes_values)[this->nodes_count.getTotalLayersCount() - 1][i]);
		}

		return outputValues;
	}
	__int64 network_core::NeuralNet::ProduceResult(const std::string input_data_file_name, const std::string output_data_file_name)
	{
		std::ifstream input_file_stream;

		input_file_stream.open(input_data_file_name, std::ios::binary);
		if (!input_file_stream.is_open()) { return 1; }

		size_t input_data_massive_lenght{};
		input_file_stream.read((char*)& input_data_massive_lenght, sizeof(size_t)); //1st line read count of examples

		NodesCountStorage rww;
		input_file_stream.read((char*)& rww, sizeof(rww)); //2nd line read network count storage class
		if (rww.getInputNodesCount() != this->nodes_count.getInputNodesCount() && rww.getOutputNodesCount() != this->nodes_count.getOutputNodesCount()) { return 2; }
		
		
		std::ofstream output_file_stream;
		output_file_stream.open(output_data_file_name, std::ios::binary);
		if (!input_file_stream.is_open()) { return 3; }
		output_file_stream.write((char*)& input_data_massive_lenght, sizeof(size_t));
		output_file_stream.write((char*)& rww, sizeof(rww)); 
		for (size_t i = 0; i < input_data_massive_lenght; i++)
		{
			
			std::vector<double> input_values;
			for (size_t j = 0; j < rww.getInputNodesCount(); i++)
			{
				input_values.push_back(0);
				input_file_stream.read((char*)& input_values[i], sizeof(double));
			}
			this->SetData(input_values);
			this->FeedForward();

			for (size_t j = 0; j < rww.getOutputNodesCount(); i++)
			{
				input_values.push_back(0);
				output_file_stream.write((char*)& (*nodes_values)[this->nodes_count.getTotalLayersCount() - 1][j], sizeof(double));
			}
		}
		return 0;
	}
	
	void network_core::NeuralNet::FeedForward()
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
	void network_core::NeuralNet::FeedBack(const std::vector<double>& expected_values)
	{
		//Calc output layer errors
		for (size_t i = 0; i < this->nodes_count.getOutputNodesCount(); i++)
		{
			(*nodes_error_values)[this->nodes_count.getTotalLayersCount() - 2][i] = expected_values[i] - (*nodes_values)[this->nodes_count.getTotalLayersCount() - 1][i];
		}
		
		//Calc all other layers errors
		for (size_t i = this->nodes_count.getTotalLayersCount() - 2; i > 0; i--)
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
	void network_core::NeuralNet::WeightsReadjustment()
	{
		for (size_t i = 0; i < (*nodes_weights).size(); i++)
		{
			for (size_t j = 0; j < (*nodes_values)[i + 1].size(); j++)
			{
				for (size_t k = 0; k < (*nodes_values)[i].size(); k++)
				{
					(*nodes_weights)[i][k][j] = (*nodes_weights)[i][k][j] + (learning_rate * (*nodes_error_values)[i][j] * activationFunction((*nodes_values)[i + 1][j], true) * (*nodes_values)[i][k]);
				}
			}
		}
	}

	__int64 network_core::NeuralNet::SetData(const std::vector<double> &input_data) // Return value is difference between network input layer size() and input data size();
	{
		
		if (input_data.size() != this->nodes_count.getInputNodesCount()) return (__int64)this->nodes_count.getInputNodesCount() - input_data.size();

		//Core part:
		for (size_t i = 0; i < (*nodes_values)[0].size() && i < input_data.size(); i++)
		{
			(*nodes_values)[0][i] = input_data[i];
		}
		return 0;
	}
	
	void network_core::NeuralNet::WeightsReinitialisation(const double lower_limit, const double upper_limit)
	{
		for (size_t i = 0; i < this->nodes_count.getHiddenLayersCount() + 1; i++)
		{
			for (size_t j = 0; j < (*nodes_values)[i].size(); j++)
			{
				for (size_t k = 0; k < (*nodes_values)[i + 1].size(); k++)
				{
					(*nodes_weights)[i][j][k] = additional_functions::RandomFunction(lower_limit, upper_limit);
				}
			}
		}
	}
	void network_core::NeuralNet::SetWeights(const double value)
	{
		for (size_t i = 0; i < this->nodes_count.getHiddenLayersCount() + 1; i++)
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
	void network_core::NeuralNet::SetLearningRate(const double value)
	{
		this->learning_rate = value;
	}

	double network_core::NeuralNet::GetLearningRate() const
	{
		return this->learning_rate;
	}
	
	void network_core::NeuralNet::PrintResult() const
	{
		std::cout << "_______________________________________________________" << std::endl;
		for (size_t i = 0; i < this->nodes_count.getOutputNodesCount(); i++)
		{
			std::cout << i + 1 << ". " << (*nodes_values)[this->nodes_count.getHiddenLayersCount() + 1][i] << std::endl;
		}
		std::cout << "_______________________________________________________" << std::endl;
	}
	void network_core::NeuralNet::PrintWeights() const
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
	inline T network_core::NeuralNet::activationFunction(const T value, const bool returnDerivativeValueInstead) const
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
	network_core::NodesCountStorage::NodesCountStorage()
	{
		this->input_nodes_count = 0;
		this->hidden_nodes_count = 0;
		this->outputNodesCount = 0;
		this->hiddenLayersCount = 0;
		this->totalLayersCount = 0;
	}

	auto network_core::NodesCountStorage::operator==(const NodesCountStorage& ex) const
	{
		return (input_nodes_count == ex.input_nodes_count) && (hidden_nodes_count == ex.hidden_nodes_count) && (outputNodesCount == ex.outputNodesCount) && (hiddenLayersCount == ex.hiddenLayersCount);
	}
	bool network_core::NodesCountStorage::operator!=(const NodesCountStorage& ex) const
	{
		return (input_nodes_count != ex.input_nodes_count) && (hidden_nodes_count != ex.hidden_nodes_count) && (outputNodesCount != ex.outputNodesCount) && (hiddenLayersCount != ex.hiddenLayersCount);
	}
	
	size_t network_core::NodesCountStorage::getInputNodesCount() const
	{
		return this->input_nodes_count;
	}
	size_t network_core::NodesCountStorage::getHiddenNodesCount() const
	{
		return this->hidden_nodes_count;
	}
	size_t network_core::NodesCountStorage::getOutputNodesCount() const
	{
		return this->outputNodesCount;
	}
	size_t network_core::NodesCountStorage::getHiddenLayersCount() const
	{
		return this->hiddenLayersCount;
	}
	size_t network_core::NodesCountStorage::getTotalLayersCount() const
	{
		return this->totalLayersCount;
	}
	
	void network_core::NodesCountStorage::setInputNodesCount(const size_t value)
	{
		this->input_nodes_count = value;
		this->totalLayersCount = this->hiddenLayersCount + (size_t) 2;
	}
	void network_core::NodesCountStorage::setHiddenNodesCount(const size_t value)
	{
		this->hidden_nodes_count = value;
		this->totalLayersCount = this->hiddenLayersCount + (size_t)2;
	}
	void network_core::NodesCountStorage::setOutputNodesCount(const size_t value)
	{
		this->outputNodesCount = value;
		this->totalLayersCount = this->hiddenLayersCount + (size_t)2;
	}
	void network_core::NodesCountStorage::setHiddenLayersCount(const size_t value)
	{
		this->hiddenLayersCount = value;
		this->totalLayersCount = this->hiddenLayersCount + (size_t)2;
	}

	void network_core::NodesCountStorage::print()
	{
		std::cout << "Input nodes: " << this->getInputNodesCount() << "  |  Hidden nodes: " << this->getHiddenNodesCount() << "  |  Output nodes: " << this->getOutputNodesCount() << "  |  Hidden layers: " << this->getHiddenLayersCount() << std::endl;
	}
	
	network_core::DataMassiveMaker::DataMassiveMaker()
	{
		this->massive_length = 0;
		this->input_data_length = 0;
		this->expected_values_length = 0;
	}

	__int64 network_core::DataMassiveMaker::printNumbersMassive(std::string dataset_file_name) const
	{
		std::ifstream input_file_stream;
		input_file_stream.open(dataset_file_name, std::ios::binary);
		
		size_t massive_length;
		input_file_stream.read((char*)& massive_length, sizeof(size_t));

		network_core::NodesCountStorage ncs;
		input_file_stream.read((char*)& ncs, sizeof(ncs));
		
		ncs.print();

		double value;
		std::cout.setf(std::ios::fixed);
		for (size_t i = 0; i < massive_length; i++)
		{
			std::cout << "\nInput data: " << std::endl;
			for (size_t j = 0; j < ncs.getInputNodesCount(); j++)
			{
				input_file_stream.read((char*)& value, sizeof(double));
				std::cout << std::setw(15) << std::left << value;
			}
			std::cout << "\nExpected value respectevly: " << std::endl;
			for (size_t j = 0; j < ncs.getOutputNodesCount(); j++)
			{
				input_file_stream.read((char*)& value, sizeof(double));
				std::cout << std::setw(15) << std::left << value;
			}
			std::cout << "\n_______________" << std::endl;
		}


		return 0;
	}

	__int64 network_core::DataMassiveMaker::evenNumbersMassive(const size_t input_data_length, const size_t outputDataSize, const size_t massive_length, const std::string dataset_file_name, const __int64 lower_limit, const __int64 upper_limit) const
	{
		std::ofstream output_file_stream;
		output_file_stream.open(dataset_file_name, std::ios::binary);
		if (!output_file_stream.is_open()) { return 1; }

		output_file_stream.write((char*)& massive_length, sizeof(size_t));
		
		network_core::NodesCountStorage ncs;
		ncs.setInputNodesCount(input_data_length);
		ncs.setOutputNodesCount(outputDataSize);
		output_file_stream.write((char*)& ncs, sizeof(ncs));
		
		for (size_t i = 0; i < massive_length; i++)
		{
			__int64 varInt{};
			
			varInt = additional_functions::RandomFunction(static_cast<__int64>(lower_limit), static_cast<__int64>(upper_limit));
			
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
	inline double additional_functions::RandomFunction(const double lower_limit, const double upper_limit)
 {
	 
	 std::random_device rd;
	 std::mt19937 gen(rd());
	 
	 std::uniform_real_distribution<double> uid(lower_limit, upper_limit);
	 
	 double rv;
	 rv = uid(gen);
	 return rv > 0 ? rv = uid(gen) : rv;
 }
	inline __int64 additional_functions::RandomFunction(const __int64 lower_limit, const __int64 upper_limit)
 {
	
	std::random_device rd;
	std::mt19937 gen(rd());
	
	std::uniform_int_distribution<__int64> uid(lower_limit, upper_limit);
	
	__int64 rv;
	rv = uid(gen);
	return rv > 0 ? rv = uid(gen) : rv;
 }

