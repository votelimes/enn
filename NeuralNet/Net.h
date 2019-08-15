#ifndef NET_H_
#define NET_H_

#include "Windows.h"

#include <vector>
#include <random>
#include <ctime>
#include <iostream>
#include <fstream>
#include <bitset>

namespace nnet {
	
	class NeuralNet
	{

	//SECTION: DATA

	private: std::vector<std::vector<std::vector<double>>>* nodesWeights;
	         std::vector<std::vector<double>>* nodesValues;
	         std::vector<std::vector<double>>* nodesErrorValues;
	

			 size_t inputNodesCount;
			 size_t hiddenNodesCount;
			 size_t outputNodesCount;
	         size_t hiddenLayersCount;

			 double learningRate;
	
	//SECTION: METHODS

	public: NeuralNet(size_t inputNodesCount, size_t hiddenNodesCount, size_t outputNodesCount, size_t hiddenLayersCount);
			
			template <class T> int readWeightsFromFile(T& weightsStorageFileName);

			template <class T> int writeWeightsToFile(T& weightsStorageFileName);
			
			void studyNetwork(std::vector<std::vector<double>>& examplesSet, std::vector<std::vector<double>>& expectedValueslesSet);
	
	private: void forwardPropogation();

			 int backPropogation(std::vector<double>& expectedValues, bool ignoreWarnings);

			 int setData(std::vector<double>& inputData, bool ignoreWarnings); // Return value is difference between network input layer size() and input data size();

			 template <class T> T activationFunction(T& value, bool returnDerivativereturnDerivativeValueInstead);

	};

} // namespace nnet
namespace afunctions {
	
	double RandomFunc();

}  // namespace afunctions
#endif // NET_H_