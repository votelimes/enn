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

namespace ann {
	
	class DataMassiveMaker 
	{
	//SECTION: DATA
	
	private:

		size_t massiveSize;
		size_t inputDataSize;
		size_t expectedValuesSize;

	//SECTION: METHODS
	public:

		DataMassiveMaker();

		__int64 printNumbersMassive(std::string fileName) const;
		__int64 evenNumbersMassive(const size_t inputDataSize, const size_t outputDataSize, const size_t massiveSize, const std::string fileName, const __int64 lowerLimit, const __int64 upperLimit) const;
	};

	class NodesCountStorage 
	{
	
	//SECTION: DATA
	private:	
		
		size_t inputNodesCount;
		size_t hiddenNodesCount;
		size_t outputNodesCount;
		size_t hiddenLayersCount;
		size_t totalLayersCount;
	

	//SECTION: METHODS
	public:
		
		NodesCountStorage();
		
		auto operator==(const NodesCountStorage& ex) const;
		bool operator!=(const NodesCountStorage& ex) const;

		size_t getInputNodesCount() const;
		size_t getHiddenNodesCount() const;
		size_t getOutputNodesCount() const;
		size_t getHiddenLayersCount() const;
		size_t getTotalLayersCount() const;

		void setInputNodesCount(const size_t value);
		void setHiddenNodesCount(const size_t value);
		void setOutputNodesCount(const size_t value);
		void setHiddenLayersCount(const size_t value);

		void print();
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
		inline T activationFunction(const T value, const bool returnDerivativereturnDerivativeValueInstead) const;
		
		void FeedForward();
		void FeedBack(const std::vector<double>& expValues);
		void WeightsReadjustment();

		__int64 SetData(const std::vector<double>& inputData);

	public: 
		
		NeuralNet(const size_t inputNodesCount, const size_t hiddenNodesCount, const size_t outputNodesCount, const size_t hiddenLayersCount); //Kernel constructor
			
		__int64 ReadWeightsFile(const std::string weightsStorageFileName); // returns 1 if file can not be open, 0 if it opens
		__int64 WriteWeightsFile(const std::string weightsStorageFileName) const; // returns 1 if file can not be open, 0 if it opens
		
		__int64 StudyFile(const std::string &fileName);
		__int64 StudyFileMT(const std::string& fileName);
		__int8 StudyOnce(const std::vector<double> &input_data);

		std::vector<double>* ProduceResult(const std::vector<double>& inputValues);
		__int64 ProduceResult(const std::string inputDataFileName, const std::string outputDataFileName);

		void SetWeights(const double value);
		void SetLearningRate(const double value);
		
		double GetLearningRate() const;
		
		void PrintResult() const;
		void PrintWeights() const;

		void WeightsReinitialisation(const double lowerLimit, const double upperLimit);

	};


} // namespace nnet
namespace afunctions {
	
	inline double RandomFunction(const double lowerLimit, const double upperLimit);
	inline __int64 RandomFunction(const __int64 lowerLimit, const __int64 upperLimit);
}

// namespace afunctions
#endif // NET_H_
