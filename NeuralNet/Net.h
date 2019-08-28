#ifndef NET_H_
#define NET_H_

#include "Windows.h"

#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>

namespace nnet {
	
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
		__int64 evenNumbersMassive(const size_t inputDataSize, const size_t outputDataSize, const size_t massiveSize, const std::string fileName) const;
	};

	class nodesCountStorage 
	{
	
	//SECTION: DATA
	private:	
		
		size_t inputNodesCount;
		size_t hiddenNodesCount;
		size_t outputNodesCount;
		size_t hiddenLayersCount;
	

	//SECTION: METHODS
	public:
		
		nodesCountStorage();
		
		auto operator==(const nodesCountStorage& ex) const;
		bool operator!=(const nodesCountStorage& ex) const;

		size_t getInputNodesCount() const;
		size_t getHiddenNodesCount() const;
		size_t getOutputNodesCount() const;
		size_t getHiddenLayersCount() const;

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
		
		
		
		__int64 backPropogationManual(const std::vector<double>& expectedValues);
		template <class T>
		inline T activationFunction(const T value, const bool returnDerivativereturnDerivativeValueInstead) const;

	public: 
		
		NeuralNet(const size_t inputNodesCount, const size_t hiddenNodesCount, const size_t outputNodesCount, const size_t hiddenLayersCount); //Kernel constructor
			
		__int64 readWeightsFromFile(const std::string weightsStorageFileName); // returns 1 if file can not be open, 0 if it opens
		__int64 writeWeightsToFile(const std::string weightsStorageFileName) const; // returns 1 if file can not be open, 0 if it opens
			
		void studyNetworkManual(const std::vector<std::vector<double>>& examplesSet, const std::vector<std::vector<double>>& expectedValueslesSet);
		__int64 studyNetworkAuto(const std::string &fileName);
		void forwardPropogationManual();
		 // Return value is difference between network input layer size() and input data size();
		void setWeights(const double value);
		void setLearningRate(const double value);
		__int64 setData(const std::vector<double>& inputData);
		__int64 setData(const std::string fileName);
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