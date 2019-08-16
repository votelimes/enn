#ifndef NET_H_
#define NET_H_

#include "Windows.h"

#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

namespace nnet {
	
	class nodesCountStorage 
	{
		float classFlag;
		size_t inputNodesCount;
		size_t hiddenNodesCount;
		size_t outputNodesCount;
		size_t hiddenLayersCount;
	
	public:
		
		nodesCountStorage();
		
		size_t getInputNodesCount();
		size_t getHiddenNodesCount();
		size_t getOutputNodesCount();
		size_t getHiddenLayersCount();

		void setInputNodesCount(size_t value);
		void setHiddenNodesCount(size_t value);
		void setOutputNodesCount(size_t value);
		void setHiddenLayersCount(size_t value);
	};
	
	class NeuralNet
	{

	//SECTION: DATA

	private: std::vector<std::vector<std::vector<double>>>* nodesWeights;
	         std::vector<std::vector<double>>* nodesValues;
	         std::vector<std::vector<double>>* nodesErrorValues;
	
			 nodesCountStorage nodesCount;
			 //size_t inputNodesCount;
			 //size_t hiddenNodesCount;
			 //size_t outputNodesCount;
	         //size_t hiddenLayersCount;

			 double learningRate;
	
	//SECTION: METHODS

	public: NeuralNet(size_t inputNodesCount, size_t hiddenNodesCount, size_t outputNodesCount, size_t hiddenLayersCount);
			
			int readWeightsFromFile(std::string weightsStorageFileName);

	        int writeWeightsToFile(std::string weightsStorageFileName);
			
			void studyNetwork(std::vector<std::vector<double>>& examplesSet, std::vector<std::vector<double>>& expectedValueslesSet);
	
  /*-----*/ void forwardPropogation();

			 __int64 backPropogation(std::vector<double>& expectedValues, bool ignoreWarnings);

			 __int64 setData(std::vector<double>& inputData, bool ignoreWarnings); // Return value is difference between network input layer size() and input data size();

			 template <class T>
			 T activationFunction(T value, bool returnDerivativereturnDerivativeValueInstead);

	};


} // namespace nnet
namespace afunctions {
	
	double RandomFunc();

}

// namespace afunctions
#endif // NET_H_