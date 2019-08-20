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

		__int64 evenNumbersMassive(size_t inputDataSize, size_t outputDataSize, size_t massiveSize, std::string& fileName);
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

		void setInputNodesCount(size_t value);
		void setHiddenNodesCount(size_t value);
		void setOutputNodesCount(size_t value);
		void setHiddenLayersCount(size_t value);
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
		
		__int64 setData(std::vector<double>& inputData, bool ignoreWarnings);
		void forwardPropogationManual();
		__int64 backPropogationManual(std::vector<double>& expectedValues, bool ignoreWarnings);
		template <class T>
		T activationFunction(T value, bool returnDerivativereturnDerivativeValueInstead) const;

	public: 
		
		NeuralNet(size_t inputNodesCount, size_t hiddenNodesCount, size_t outputNodesCount, size_t hiddenLayersCount); //Kernel constructor
			
		__int64 readWeightsFromFile(std::string weightsStorageFileName); // returns 1 if file can not be open, 0 if it opens
		__int64 writeWeightsToFile(std::string weightsStorageFileName) const; // returns 1 if file can not be open, 0 if it opens
			
		void studyNetworkManual(std::vector<std::vector<double>>& examplesSet, std::vector<std::vector<double>>& expectedValueslesSet);
		__int64 studyNetworkAuto(std::string &fileName);

		 // Return value is difference between network input layer size() and input data size();
		void setWeights(double value);
		void setLearningRate(double value);

		double getLearningRate() const;

		void reinitializeWeights(double lowerLimit, double upperLimit);

		

	};


} // namespace nnet
namespace afunctions {
	
	inline double RandomFunc(double lowerLimit, double upperLimit);
	inline __int64 RandomFunc(__int64 lowerLimit, __int64 upperLimit);
}

// namespace afunctions
#endif // NET_H_