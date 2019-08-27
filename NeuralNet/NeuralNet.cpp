// NeuralNet.cpp : Main function file.
//

#include "NeuralNet.h"

int main()
{	
	nai::NetApplicationInterface int_1;
	
	/*nnet::dataMassiveMaker dmm;
	dmm.printNumbersMassive("dataMassive1.bin");*/

	size_t netSize[]{ 1, 5, 1, 1 };
	nnet::NeuralNet nnet_1(netSize[0], netSize[1], netSize[2], netSize[3]);


	int_1.doWork();


}
