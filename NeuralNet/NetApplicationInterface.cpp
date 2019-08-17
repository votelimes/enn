#include "NetApplicationInterface.h"

nai::NetApplicationInterface::NetApplicationInterface(nnet::NeuralNet &nnet)
{
	this->commandsList.push_back("help");
	this->commandsList.push_back("saveweight");
	this->commandsList.push_back("loadweight");
	this->commandsList.push_back("reinitializeweight");
	this->commandsList.push_back("createnetwork");
	this->commandsList.push_back("trainnetwork");
	this->commandsList.push_back("getresultW");
	this->commandsList.push_back("getresultF");
	this->commandsList.push_back("getnetworkinfo");
	this->commandsList.push_back("exit");
	this->commandsList.push_back("");

	this->commandsDescription.push_back("Shows a list of all commands and their purpose.");
	this->commandsDescription.push_back("Saves current weight values file(user should choose a file name.)");
	this->commandsDescription.push_back("Loads weight values from a file(user should choose a file name).");
	this->commandsDescription.push_back("Reinitializes weight values with a custom value, written by user.");
	this->commandsDescription.push_back("Creates the network according to the parametrs entered by user.");
	this->commandsDescription.push_back("Trains the network using a two data arrays: input values and output values respectively. User should select file names.");
	this->commandsDescription.push_back("Uses a network to get result. User should write input values.");
	this->commandsDescription.push_back("Uses a network to get result. User should write input data file name.");
	this->commandsDescription.push_back("Prints current network parametrs.");
	this->commandsDescription.push_back("Closes a program without saving reults.");
	this->commandsDescription.push_back("");

	this->numberOfNetworkTrainings = 0;
}

nai::NetApplicationInterface::~NetApplicationInterface()
{
}

void nai::NetApplicationInterface::doWork()
{
	std::string command;

	while (true)
	{
		

	}
}
