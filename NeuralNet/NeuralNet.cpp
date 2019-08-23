// NeuralNet.cpp : Main function file.
//

#include "NeuralNet.h"

int main()
{	
	size_t var = 0;
	std::string command;
	nai::NetApplicationInterface int_1;
	nnet::dataMassiveMaker dmm;
	//dmm.evenNumbersMassive(1, 1, 10000, "dataMassive1.bin");
	dmm.printNumbersMassive("dataMassive1.bin");
	int_1.doWork();


}
