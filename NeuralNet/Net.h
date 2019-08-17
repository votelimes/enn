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

namespace nnet {
	
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
		
		bool operator==(const nodesCountStorage& ex) const;
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

	private: 
		
		std::vector<std::vector<std::vector<double>>>* nodesWeights;
	    std::vector<std::vector<double>>* nodesValues;
	    std::vector<std::vector<double>>* nodesErrorValues;
	
		nodesCountStorage nodesCount;
		double learningRate;
	
	//SECTION: METHODS
		
	public: 
		
		NeuralNet(size_t inputNodesCount, size_t hiddenNodesCount, size_t outputNodesCount, size_t hiddenLayersCount); //Kernel constructor
			
		__int64 readWeightsFromFile(std::string weightsStorageFileName); // returns 1 if file can not be open, 0 if it opens
		__int64 writeWeightsToFile(std::string weightsStorageFileName) const; // returns 1 if file can not be open, 0 if it opens
			
		void studyNetwork(std::vector<std::vector<double>>& examplesSet, std::vector<std::vector<double>>& expectedValueslesSet);
		void forwardPropogation();
		__int64 backPropogation(std::vector<double>& expectedValues, bool ignoreWarnings);

		__int64 setData(std::vector<double>& inputData, bool ignoreWarnings); // Return value is difference between network input layer size() and input data size();
		void setWeights(double value);

		void reinitializeWeights(double lowerLimit, double upperLimit);

		template <class T>
		T activationFunction(T value, bool returnDerivativereturnDerivativeValueInstead) const;

	};


} // namespace nnet
namespace afunctions {
	
	double RandomFunc(double lowerLimit, double upperLimit);

}

// namespace afunctions
#endif // NET_H_