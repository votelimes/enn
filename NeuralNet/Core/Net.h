#ifndef NET_H_
#define NET_H_

#include "Windows.h"
#include "ppl.h"
#include "omp.h"

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
	
	class dataMassiveMaker 
	{
	//SECTION: DATA
	
	private:

		size_t massiveSize;
		size_t inputDataSize;
		size_t expectedValuesSize;

	//SECTION: METHODS
	public:

		dataMassiveMaker();

		__int64 printNumbersMassive(std::string fileName) const;
		__int64 evenNumbersMassive(const size_t inputDataSize, const size_t outputDataSize, const size_t massiveSize, const std::string fileName, const __int64 lowerLimit, const __int64 upperLimit) const;
	};

	class nodesCountStorage 
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
		
		nodesCountStorage();
		
		auto operator==(const nodesCountStorage& ex) const;
		bool operator!=(const nodesCountStorage& ex) const;

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

		nodesCountStorage nodesCount;

	private: 
		
		std::vector<std::vector<std::vector<double>>>* nodesWeights;
	    std::vector<std::vector<double>>* nodesValues;
	    std::vector<std::vector<double>>* nodesErrorValues;
		double learningRate;
		
		
	
	//SECTION: METHODS
		
	private:
		
		template <class T>
		inline T activationFunction(const T value, const bool returnDerivativereturnDerivativeValueInstead) const;
		
		void feedForward();
		void feedBack(const std::vector<double>& expValues);
		void weightsReadjustment();

		__int64 setData(const std::vector<double>& inputData);

	public: 
		
		NeuralNet(const size_t inputNodesCount, const size_t hiddenNodesCount, const size_t outputNodesCount, const size_t hiddenLayersCount); //Kernel constructor
			
		__int64 readWeightsFromFile(const std::string weightsStorageFileName); // returns 1 if file can not be open, 0 if it opens
		__int64 writeWeightsToFile(const std::string weightsStorageFileName) const; // returns 1 if file can not be open, 0 if it opens
		
		__int64 studyNetworkAuto(const std::string &fileName);
		__int64 studyNetworkFileMT(const std::string& fileName);

		std::vector<double>* produceResult(const std::vector<double>& inputValues);
		
		void setWeights(const double value);
		void setLearningRate(const double value);
		
		double getLearningRate() const;
		
		void printResult() const;
		void printWeights() const;

		void reinitializeWeights(const double lowerLimit, const double upperLimit);

	};


} // namespace nnet
namespace afunctions {
	
	inline double RandomFunc(const double lowerLimit, const double upperLimit);
	inline __int64 RandomFunc(const __int64 lowerLimit, const __int64 upperLimit);
}

// namespace afunctions
#endif // NET_H_
