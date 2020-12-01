#ifndef NET_H_
#define NET_H_

#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <bitset>


// types defenitions

typedef long long int64;
typedef std::vector<std::vector<double>> vector_2D;
typedef std::vector<std::vector<std::vector<double>>> vector_3D;
//

namespace network_core {
	
	class DataMassiveMaker 
	{
	//SECTION: DATA
	
	private:

		size_t massive_length;
		size_t input_data_length;
		size_t expected_values_length;

	//SECTION: METHODS
	public:

		DataMassiveMaker();

		__int64 printNumbersMassive(std::string file_name) const;
		__int64 evenNumbersMassive(const size_t input_data_length, const size_t output_data_length, const size_t massive_length, const std::string file_name, const __int64 lower_limit, const __int64 upper_limit) const;
	};

	class NodesCountStorage 
	{
	
	//SECTION: DATA
	private:	
		
		size_t input_nodes_count;
		size_t hidden_nodes_count;
		size_t output_nodes_count;
		size_t hidden_layers_count;
		size_t total_layers_count;
	

	//SECTION: METHODS
	public:
		
		NodesCountStorage();
		
		auto operator==(const NodesCountStorage& ex) const;
		bool operator!=(const NodesCountStorage& ex) const;

		size_t GetInputNodesCount() const;
		size_t GetHiddenNodesCount() const;
		size_t GetOutputNodesCount() const;
		size_t GetHiddenLayersCount() const;
		size_t GetTotalLayersCount() const;

		void SetInputNodesCount(const size_t value);
		void SetHiddenNodesCount(const size_t value);
		void SetOutputNodesCount(const size_t value);
		void SetHiddenLayersCount(const size_t value);

		void Print();
	};
	
	class NeuralNet
	{

	//SECTION: DATA
	public:

		NodesCountStorage nodes_count;

	private: 
		
		std::vector<std::vector<std::vector<double>>>* nodes_weights;
	    std::vector<std::vector<double>>* nodes_values;
	    std::vector<std::vector<double>>* nodes_error_values;
		double learning_rate;
		
		
	
	//SECTION: METHODS
		
	private:
		
		template <class T>
		inline T activation_function(const T value, const bool returnDerivativereturnDerivativeValueInstead) const;
		
		void FeedForward();
		template <class T> void FeedBack(const std::vector<T>& expected_values)
		{
			//Calc output layer errors
			for (auto i = 0; i < this->nodes_count.GetOutputNodesCount(); i++)
			{
				(*nodes_error_values)[this->nodes_count.GetTotalLayersCount() - 2][i] = expected_values[i] - (*nodes_values)[this->nodes_count.GetTotalLayersCount() - 1][i];
			}
			
			//Calc all other layers errors
			for (auto i = this->nodes_count.GetTotalLayersCount() - 2; i > 0; i--)
			{
				for (auto j = 0; j < (*nodes_values)[i].size(); j++)
				{
					double shift_collector{};
					for (auto k = 0; k < (*nodes_values)[i + 1].size(); k++)
					{
						shift_collector = shift_collector + ((*nodes_weights)[i][j][k] * (*nodes_error_values)[i][k]);
					}

					(*nodes_error_values)[i - 1][j] = shift_collector;
				}
			}
		}
		void WeightsReadjustment();
		template <class T> __int64 SetData(const std::vector<T>& input_data)
		{
			
			if (input_data.size() != this->nodes_count.GetInputNodesCount()) return (__int64)this->nodes_count.GetInputNodesCount() - input_data.size();

			//Core part:
			for (auto i = 0; i < (*nodes_values)[0].size() && i < input_data.size(); i++)
			{
				(*nodes_values)[0][i] = input_data[i];
			}
			return 0;
		}

	public: 
		
		NeuralNet(const size_t input_nodes_count, const size_t hidden_nodes_count, const size_t output_nodes_count, const size_t hidden_layers_count); //Kernel constructor
			
		__int64 ReadWeightsFile(const std::string weights_storage_file_name); // returns 1 if file can not be open, 0 if it opens
		__int64 WriteWeightsFile(const std::string weights_storage_file_name) const; // returns 1 if file can not be open, 0 if it opens
		
		__int64 StudyFile(const std::string &dataset_file_name);
		__int64 StudyFileMT(const std::string& dataset_file_name);
		template <class T1, class T2> void StudyOnce(const std::vector<T1> &input_data, const std::vector<T2> &expected_values){
		
			this->SetData(input_data);
			this->FeedForward();
			this->FeedBack(expected_values);
		}

		std::vector<double>* ProduceResult(const std::vector<double>& inputValues);
		__int64 ProduceResult(const std::string input_data_file_name, const std::string output_data_file_name);

		void SetWeights(const double value);
		void SetLearningRate(const double value);
		
		double GetLearningRate() const;
		
		void PrintResult() const;
		void PrintWeights() const;

		void WeightsReinitialisation(const double lower_limit, const double upper_limit);

	};


} // namespace nnet
namespace additional_functions {
	
	double RandomFunction(const double lower_limit, const double upper_limit);
	__int64 RandomFunction(const __int64 lower_limit, const __int64 upper_limit);
}

// namespace additional_functions
#endif // NET_H_
