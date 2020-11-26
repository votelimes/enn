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
		template <class T>
		void FeedBack(const std::vector<T>& expValues);
		void WeightsReadjustment();
		template <class T>
		__int64 SetData(const std::vector<T>& inputData);

	public: 
		
		NeuralNet(const size_t input_nodes_count, const size_t hidden_nodes_count, const size_t output_nodes_count, const size_t hidden_layers_count); //Kernel constructor
			
		__int64 ReadWeightsFile(const std::string weightsStorageFileName); // returns 1 if file can not be open, 0 if it opens
		__int64 WriteWeightsFile(const std::string weightsStorageFileName) const; // returns 1 if file can not be open, 0 if it opens
		
		__int64 StudyFile(const std::string &file_name);
		__int64 StudyFileMT(const std::string& file_name);
		template <class T>
		void StudyOnce(const std::vector<T> &input_data, const std::vector<T> &expected_values);

		template <class T>
		std::vector<T>* ProduceResult(const std::vector<T>& inputValues);
		__int64 ProduceResult(const std::string inputDataFileName, const std::string outputDataFileName);

		void SetWeights(const double value);
		void SetLearningRate(const double value);
		
		double GetLearningRate() const;
		
		void PrintResult() const;
		void PrintWeights() const;

		void WeightsReinitialisation(const double lower_limit, const double upper_limit);

	};


} // namespace nnet
namespace additional_functions {
	
	inline double RandomFunction(const double lower_limit, const double upper_limit);
	inline __int64 RandomFunction(const __int64 lower_limit, const __int64 upper_limit);
}

// namespace additional_functions
#endif // NET_H_
