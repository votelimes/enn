#include <src/core/net.h>

	
	//Kernel class:
	network_core::NeuralNet::NeuralNet(const size_t input_nodes_count, const size_t hidden_nodes_count, const size_t output_nodes_count, const size_t hidden_layers_count) // //Kernel constructor, hidden_nodes_count should be the largest
	{
		//Fill variables:
		
		this->nodes_count.SetInputNodesCount(input_nodes_count);
		this->nodes_count.SetHiddenNodesCount(hidden_nodes_count);
		this->nodes_count.SetOutputNodesCount(output_nodes_count);
		this->nodes_count.SetHiddenLayersCount(hidden_layers_count);
		this->learning_rate = 0.1;
		
		//Create and normalize arrays
		this->nodes_weights = new std::vector<std::vector<std::vector<double>>>(hidden_layers_count + 1, std::vector<std::vector<double>>(hidden_nodes_count, std::vector<double>(hidden_nodes_count, 0)));
		this->nodes_values = new std::vector<std::vector<double>>(hidden_layers_count + 2, std::vector<double>(hidden_nodes_count, 0));
		this->nodes_error_values = new std::vector<std::vector<double>>(hidden_layers_count + 1, std::vector<double>(hidden_nodes_count, 0));

		(*nodes_values)[0].resize(input_nodes_count, 0);
		(*nodes_values)[0].shrink_to_fit();
		(*nodes_values)[this->nodes_count.GetHiddenLayersCount() + 1].resize(output_nodes_count, 0);
		(*nodes_values)[this->nodes_count.GetHiddenLayersCount() + 1].shrink_to_fit();

		(*nodes_error_values)[this->nodes_count.GetHiddenLayersCount()].resize(output_nodes_count, 0);
		(*nodes_error_values)[this->nodes_count.GetHiddenLayersCount()].shrink_to_fit();

		(*nodes_weights)[0].resize(input_nodes_count);
		(*nodes_weights)[0].shrink_to_fit();
		//
		for (auto i = 0; i < (*nodes_weights)[this->nodes_count.GetHiddenLayersCount()].size(); i++)
		{
			(*nodes_weights)[this->nodes_count.GetHiddenLayersCount()][i].resize(output_nodes_count, additional_functions::RandomFunction(0.0, 1.0));
			(*nodes_weights)[this->nodes_count.GetHiddenLayersCount()][i].shrink_to_fit();
		}
		//Weights initialization(random values) cicles:
		
		for (auto i = 0; i < this->nodes_count.GetHiddenLayersCount() + 1; i++)
		{
			for (auto k = 0; k < (*nodes_weights)[i].size(); k++)
			{
				for (auto j = 0; j < (*nodes_weights)[i][k].size(); j++)
				{
					(*nodes_weights)[i][k][j] = additional_functions::RandomFunction(0.0, 1.0);
				}
			}
		}
		//
	} // //Kernel constructor

	__int64 network_core::NeuralNet::ReadWeightsFile(const std::string weights_storage_file_name) // Returns 2 if layers nodes count does not match, returns 1 if file can not be open, 0 if it opens
	{
		std::ifstream input_file_stream;

		input_file_stream.open(weights_storage_file_name, std::ios::binary);
		if (!input_file_stream.is_open()) {
			return 1;
		}

		NodesCountStorage file_nodes_count;
		input_file_stream.read((char*)&file_nodes_count, sizeof(file_nodes_count));
		
		if (file_nodes_count != this->nodes_count) {
			return 2;
		}

		for (auto i = 0; i < this->nodes_count.GetHiddenLayersCount() + 1; i++)
		{
			for (size_t j = 0; j < (*nodes_values)[i].size(); j++)
			{
				for (size_t k = 0; k < (*nodes_values)[i + 1].size(); k++)
				{
					static double weight_value{};
					input_file_stream.read((char*) & weight_value, sizeof(double));
					(*nodes_weights)[i][j][k] =  weight_value;
				}
			}
		}
		
		return 0;
	}
	__int64 network_core::NeuralNet::WriteWeightsFile(const std::string weights_storage_file_name) const // Returns 1 if file can not be open, 0 if it opens
	{
		std::ofstream output_file_stream;
		output_file_stream.open(weights_storage_file_name, std::ios::binary);
		
		if (!output_file_stream.is_open()) {
			return 1;
		}
		
		output_file_stream.write((char*)& this->nodes_count, sizeof(this->nodes_count));
		
		for (auto i = 0; i < this->nodes_count.GetHiddenLayersCount() + 1; i++)
		{
			for (auto j = 0; j < (*nodes_values)[i].size(); j++)
			{
				for (auto k = 0; k < (*nodes_values)[i + 1].size(); k++)
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

		NodesCountStorage file_nodes_count;
		input_file_stream.read((char*)& file_nodes_count, sizeof(file_nodes_count)); //2nd line read network count storage class
		if (file_nodes_count.GetInputNodesCount() != this->nodes_count.GetInputNodesCount() && file_nodes_count.GetOutputNodesCount() != this->nodes_count.GetOutputNodesCount()) { return 2; }

		for (auto y = 0; y < input_data_massive_lenght; y++)
		{
			double file_stream_buffer{};
			//input layer initialization
			for (auto u = 0; u < file_nodes_count.GetInputNodesCount(); u++)
			{
				input_file_stream.read((char*)& file_stream_buffer, sizeof(double));
				(*nodes_values)[0][u] = file_stream_buffer;
			}
			file_stream_buffer = 0;
			//forward propogation:
			
			for (auto i = 0; i < (*nodes_weights).size(); i++)
			{
				for (auto j = 0; j < (*nodes_values)[i + 1].size(); j++)
				{
					for (auto k = 0; k < (*nodes_weights)[i].size(); k++)
					{
						file_stream_buffer = file_stream_buffer + ((*nodes_weights)[i][k][j] * (*nodes_values)[i][k]);
					}

					(*nodes_values)[i + 1][j] = activation_function(file_stream_buffer, false);

					file_stream_buffer = 0;
				}
			}

			////Calculate error procent for output layer:
			for (auto i = 0; i < (*nodes_values)[(*nodes_values).size() - 1].size(); i++)
			{
				input_file_stream.read((char*)& file_stream_buffer, sizeof(double));
				(*nodes_error_values)[(*nodes_error_values).size() - 1][i] = file_stream_buffer - (*nodes_values)[(*nodes_values).size() - 1][i];
			}
			file_stream_buffer = 0;

			//Calculate error procent for all other layers:

			//concurrency::parallel_for(((*nodes_values).size() - 2), (size_t)0, [&](size_t i)
			for (auto i = (*nodes_values).size() - 2; i > 0; i--)
			{
				for (auto j = 0; j < (*nodes_values)[i].size(); j++)
				{
					for (auto k = 0; k < (*nodes_values)[i + 1].size(); k++)
					{
						file_stream_buffer = file_stream_buffer + ((*nodes_weights)[i][j][k] * (*nodes_error_values)[i][k]);
					}

					(*nodes_error_values)[i - 1][j] = file_stream_buffer;
					file_stream_buffer = 0;
				}
			}

			//Adjust weights:
			for (auto i = 0; i < (*nodes_weights).size(); i++)
			{
				for (auto j = 0; j < (*nodes_values)[i + 1].size(); j++)
				{
					for (auto k = 0; k < (*nodes_values)[i].size(); k++)
					{
						(*nodes_weights)[i][k][j] = (*nodes_weights)[i][k][j] + (learning_rate * (*nodes_error_values)[i][j] * activation_function((*nodes_values)[i + 1][j], true) * (*nodes_values)[i][k]);
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

		NodesCountStorage file_nodes_count;
		input_file_stream.read((char*)& file_nodes_count, sizeof(file_nodes_count)); //2nd line read network count storage class
		if (file_nodes_count.GetInputNodesCount() != this->nodes_count.GetInputNodesCount() && file_nodes_count.GetOutputNodesCount() != this->nodes_count.GetOutputNodesCount()) { return 2; }

		for (auto y = 0; y < input_data_massive_lenght; y++)
		{
			double file_stream_buffer{};
			//Get input data from file
			std::vector<double> input_data;
			for (size_t u = 0; u < file_nodes_count.GetInputNodesCount(); u++)
			{
				input_file_stream.read((char*)& file_stream_buffer, sizeof(double));
				input_data.push_back(file_stream_buffer);
			}
			//Set input data to network
			this->SetData(input_data);
			//Feed forward
			this->FeedForward();
			//Get expected values from file
			std::vector<double> expected_values;
			
			for (auto i = 0; i < file_nodes_count.GetOutputNodesCount(); i++)
			{
				input_file_stream.read((char*)& file_stream_buffer, sizeof(double));
				expected_values.push_back(file_stream_buffer);
			}
			//Calculate error procent for all layers
			this->FeedBack(expected_values);

			//Adjust weights:
			for (auto i = 0; i < (*nodes_weights).size(); i++)
			{
				for (auto j = 0; j < (*nodes_values)[i + 1].size(); j++)
				{
					for (auto k = 0; k < (*nodes_values)[i].size(); k++)
					{
						(*nodes_weights)[i][k][j] = (*nodes_weights)[i][k][j] + (learning_rate * (*nodes_error_values)[i][j] * activation_function((*nodes_values)[i + 1][j], true) * (*nodes_values)[i][k]);
					}
				}
			}

		}
		return 0;
	}
	
	std::vector<double>* network_core::NeuralNet::ProduceResult(const std::vector<double>& input_values)
	{
		if (this->SetData(input_values)) { return nullptr; }
		
		this->FeedForward();

		static std::vector<double>* output_values = new std::vector<double>;
		for (auto i = 0; i < (*nodes_values)[this->nodes_count.GetTotalLayersCount() - 1].size(); i++)
		{
			output_values->push_back((*nodes_values)[this->nodes_count.GetTotalLayersCount() - 1][i]);
		}

		return output_values;
	}
	__int64 network_core::NeuralNet::ProduceResult(const std::string input_data_file_name, const std::string output_data_file_name)
	{
		std::ifstream input_file_stream;

		input_file_stream.open(input_data_file_name, std::ios::binary);
		if (!input_file_stream.is_open()) { return 1; }

		size_t input_data_massive_lenght{};
		input_file_stream.read((char*)& input_data_massive_lenght, sizeof(size_t)); //1st line read count of examples

		NodesCountStorage file_nodes_count;
		input_file_stream.read((char*)& file_nodes_count, sizeof(file_nodes_count)); //2nd line read network count storage class
		if (file_nodes_count.GetInputNodesCount() != this->nodes_count.GetInputNodesCount() && file_nodes_count.GetOutputNodesCount() != this->nodes_count.GetOutputNodesCount()) { return 2; }
		
		
		std::ofstream output_file_stream;
		output_file_stream.open(output_data_file_name, std::ios::binary);
		if (!input_file_stream.is_open()) { return 3; }
		output_file_stream.write((char*)& input_data_massive_lenght, sizeof(size_t));
		output_file_stream.write((char*)& file_nodes_count, sizeof(file_nodes_count)); 
		for (auto i = 0; i < input_data_massive_lenght; i++)
		{
			
			std::vector<double> input_values;
			for (auto j = 0; j < file_nodes_count.GetInputNodesCount(); i++)
			{
				input_values.push_back(0);
				input_file_stream.read((char*)& input_values[i], sizeof(double));
			}
			this->SetData(input_values);
			this->FeedForward();

			for (auto j = 0; j < file_nodes_count.GetOutputNodesCount(); i++)
			{
				input_values.push_back(0);
				output_file_stream.write((char*)& (*nodes_values)[this->nodes_count.GetTotalLayersCount() - 1][j], sizeof(double));
			}
		}
		return 0;
	}
	
	void network_core::NeuralNet::FeedForward()
	{
		double shift_collector{};

		for (auto i = 0; i < (*nodes_weights).size(); i++)
		{
			for (auto j = 0; j < (*nodes_values)[i + 1].size(); j++)
			{
				for (auto k = 0; k < (*nodes_weights)[i].size(); k++)
				{
					shift_collector = shift_collector + ((*nodes_weights)[i][k][j] * (*nodes_values)[i][k]);
				}

				(*nodes_values)[i + 1][j] = activation_function(shift_collector, false);

				shift_collector = 0;
			}
		}
	}
	
	void network_core::NeuralNet::WeightsReadjustment()
	{
		for (auto i = 0; i < (*nodes_weights).size(); i++)
		{
			for (auto j = 0; j < (*nodes_values)[i + 1].size(); j++)
			{
				for (auto k = 0; k < (*nodes_values)[i].size(); k++)
				{
					(*nodes_weights)[i][k][j] = (*nodes_weights)[i][k][j] + (learning_rate * (*nodes_error_values)[i][j] * activation_function((*nodes_values)[i + 1][j], true) * (*nodes_values)[i][k]);
				}
			}
		}
	}	
	void network_core::NeuralNet::WeightsReinitialisation(const double lower_limit, const double upper_limit)
	{
		for (auto i = 0; i < this->nodes_count.GetHiddenLayersCount() + 1; i++)
		{
			for (auto j = 0; j < (*nodes_values)[i].size(); j++)
			{
				for (auto k = 0; k < (*nodes_values)[i + 1].size(); k++)
				{
					(*nodes_weights)[i][j][k] = additional_functions::RandomFunction(lower_limit, upper_limit);
				}
			}
		}
	}
	void network_core::NeuralNet::SetWeights(const double value)
	{
		for (auto i = 0; i < this->nodes_count.GetHiddenLayersCount() + 1; i++)
		{
			for (auto j = 0; j < (*nodes_values)[i].size(); j++)
			{
				for (auto k = 0; k < (*nodes_values)[i + 1].size(); k++)
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
		for (auto i = 0; i < this->nodes_count.GetOutputNodesCount(); i++)
		{
			std::cout << i + 1 << ". " << (*nodes_values)[this->nodes_count.GetHiddenLayersCount() + 1][i] << std::endl;
		}
		std::cout << "_______________________________________________________" << std::endl;
	}
	void network_core::NeuralNet::PrintWeights() const
	{
		
		std::cout.setf(std::ios::fixed);
		for (auto i = 0; i < (*nodes_weights).size(); i++)
		{
			std::cout << "________________LAYER " << i + 1 << " WEIGHTS________________" << std::endl << std::endl;
			for (auto j = 0; j < (*nodes_weights)[i].size(); j++)
			{
				std::cout << "________NODE " << j + 1 << " WEIGHTS________" << std::endl << std::endl;
				for (auto k = 0; k < (*nodes_weights)[i][j].size(); k++)
				{
					std::cout << k + 1 << ". ";
					std::cout << std::setw(15) << std::left << (*nodes_weights)[i][j][k];
				}
				std::cout << std::endl << std::endl;
			}
		}
	}
	

	

	template <class T>
	inline T network_core::NeuralNet::activation_function(const T value, const bool returnDerivativeValueInstead) const
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
		this->output_nodes_count = 0;
		this->hidden_layers_count = 0;
		this->total_layers_count = 0;
	}

	auto network_core::NodesCountStorage::operator==(const NodesCountStorage& ex) const
	{
		return (input_nodes_count == ex.input_nodes_count) && (hidden_nodes_count == ex.hidden_nodes_count) && (output_nodes_count == ex.output_nodes_count) && (hidden_layers_count == ex.hidden_layers_count);
	}
	bool network_core::NodesCountStorage::operator!=(const NodesCountStorage& ex) const
	{
		return (input_nodes_count != ex.input_nodes_count) && (hidden_nodes_count != ex.hidden_nodes_count) && (output_nodes_count != ex.output_nodes_count) && (hidden_layers_count != ex.hidden_layers_count);
	}
	
	size_t network_core::NodesCountStorage::GetInputNodesCount() const
	{
		return this->input_nodes_count;
	}
	size_t network_core::NodesCountStorage::GetHiddenNodesCount() const
	{
		return this->hidden_nodes_count;
	}
	size_t network_core::NodesCountStorage::GetOutputNodesCount() const
	{
		return this->output_nodes_count;
	}
	size_t network_core::NodesCountStorage::GetHiddenLayersCount() const
	{
		return this->hidden_layers_count;
	}
	size_t network_core::NodesCountStorage::GetTotalLayersCount() const
	{
		return this->total_layers_count;
	}
	
	void network_core::NodesCountStorage::SetInputNodesCount(const size_t value)
	{
		this->input_nodes_count = value;
		this->total_layers_count = this->hidden_layers_count + (size_t) 2;
	}
	void network_core::NodesCountStorage::SetHiddenNodesCount(const size_t value)
	{
		this->hidden_nodes_count = value;
		this->total_layers_count = this->hidden_layers_count + (size_t)2;
	}
	void network_core::NodesCountStorage::SetOutputNodesCount(const size_t value)
	{
		this->output_nodes_count = value;
		this->total_layers_count = this->hidden_layers_count + (size_t)2;
	}
	void network_core::NodesCountStorage::SetHiddenLayersCount(const size_t value)
	{
		this->hidden_layers_count = value;
		this->total_layers_count = this->hidden_layers_count + (size_t)2;
	}

	void network_core::NodesCountStorage::Print()
	{
		std::cout << "Input nodes: " << this->GetInputNodesCount() << "  |  Hidden nodes: " << this->GetHiddenNodesCount() << "  |  Output nodes: " << this->GetOutputNodesCount() << "  |  Hidden layers: " << this->GetHiddenLayersCount() << std::endl;
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
		
		ncs.Print();

		double value;
		std::cout.setf(std::ios::fixed);
		for (auto i = 0; i < massive_length; i++)
		{
			std::cout << "\nInput data: " << std::endl;
			for (auto j = 0; j < ncs.GetInputNodesCount(); j++)
			{
				input_file_stream.read((char*)& value, sizeof(double));
				std::cout << std::setw(15) << std::left << value;
			}
			std::cout << "\nExpected value respectevly: " << std::endl;
			for (auto j = 0; j < ncs.GetOutputNodesCount(); j++)
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
		ncs.SetInputNodesCount(input_data_length);
		ncs.SetOutputNodesCount(outputDataSize);
		output_file_stream.write((char*)& ncs, sizeof(ncs));
		
		for (auto i = 0; i < massive_length; i++)
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
	double additional_functions::RandomFunction(const double lower_limit, const double upper_limit)
	{
		
		std::random_device rd;
		std::mt19937 gen(rd());
		
		std::uniform_real_distribution<double> uid(lower_limit, upper_limit);
		
		double rv;
		rv = uid(gen);
		return rv > 0 ? rv = uid(gen) : rv;
	}
	__int64 additional_functions::RandomFunction(const __int64 lower_limit, const __int64 upper_limit)
	{
		
		std::random_device rd;
		std::mt19937 gen(rd());
		
		std::uniform_int_distribution<__int64> uid(lower_limit, upper_limit);
		
		__int64 rv;
		rv = uid(gen);
		return rv > 0 ? rv = uid(gen) : rv;
	}

