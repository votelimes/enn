#include "NetApplicationInterface.h"

nai::NetApplicationInterface::NetApplicationInterface()
{
	this->commandsList.push_back("/HELP");
	this->commandsList.push_back("/EXIT");
	this->commandsList.push_back("/SAVEWEIGHTS");
	this->commandsList.push_back("/LOADWEIGHTS");
	this->commandsList.push_back("/REINITIALIZEWEIGHTS");
	this->commandsList.push_back("/CREATENETWORK");
	this->commandsList.push_back("/TRAINNETWORK");
	this->commandsList.push_back("/GETRESULTW");
	this->commandsList.push_back("/GETRESULTF");
	this->commandsList.push_back("/GETNETWORKINFO");
	
	//this->commandsList.push_back("");

	this->commandsDescription.push_back("Shows a list of all commands and their purpose.");
	this->commandsDescription.push_back("Closes a program without saving reults.");
	this->commandsDescription.push_back("Saves current weight values to file. Ex.: /saveweights FILENAME");
	this->commandsDescription.push_back("Loads weight values from a file. Ex.: /saveweights FILENAME");
	this->commandsDescription.push_back("Reinitializes weight values. Ex.: /reinitializeweights 0.415");
	this->commandsDescription.push_back("Creates the network according to the parametrs. Parametrs (input neurons count) (hidden neurons count) (output neurons count) (output layers count). Ex.: /createnetwork 3 15 1 2");
	this->commandsDescription.push_back("Trains the network using a data array file. Ex.: /trainnetwork FILENAME.");
	this->commandsDescription.push_back("Uses a network to get result. Input values count respectevly to network input layer neurons count. Ex.: /getresultw INPUT1 INPUT2 INPUT3...");
	this->commandsDescription.push_back("Uses a network to get result. Using a input data as file. Ex. /getresultf FILENAME");
	this->commandsDescription.push_back("Prints current network parametrs.");
	
	//this->commandsDescription.push_back("");

	this->trainingsCount = 0;
	this->net1 = NULL;

}

nai::NetApplicationInterface::~NetApplicationInterface()
{
}

void nai::NetApplicationInterface::doWork()
{
	while (true)
	{
		std::string command;
		//std::getline(std::cin, command);
		std::cin >> command;
		this->toUpperCase(command);
		__int64 commandIndex = this->findCommand(command);
		std::vector<std::string> parametrsStorage;
		parametrsStorage.clear();
		while (std::cin.peek() != 10)
		{
			std::string temporary;
			std::cin >> temporary;
			parametrsStorage.push_back(temporary);
		}
		//unknownCommand
		if (commandIndex == -1) {
			std::cout << "Unknown command. " << this->useHelp() << std::endl;
			continue;
		}
		//help
		if (commandIndex == 0) {

			for (size_t i = 0; i < this->commandsList.size(); i++) {
				std::string helpS;
				helpS = "\n" + std::to_string(i + 1) + "." + " " + commandsList[i];
				std::cout << std::setw(25) << std::left << helpS << commandsDescription[i] << std::endl;
			}
			continue;
		}
		//exit
		if (commandIndex == 1) { 
			
			break;
		}
		//Write weights (saveWeights)
		if (commandIndex == 2 && parametrsStorage.size() == 1) {
			if (this->net1) {

				if (!this->net1->writeWeightsToFile(parametrsStorage[0])) { std::cout << this->successfullyExecuted() << std::endl; }
				else { std::cout << "Unable to create file." << std::endl; }
			}
			else { std::cout << "Create network first. " << this->useHelp() << std::endl; }
			continue;
		} 
		//Read weights (loadWeights)
		if (commandIndex == 3 && parametrsStorage.size() == 1) {
			if (this->net1) {
				if (!this->net1->readWeightsFromFile(parametrsStorage[0])) { std::cout << "Unable to open file." << std::endl; }
			}
			else { std::cout << "Create network first. " << this->useHelp() << std::endl; }
			continue;
		} 
		//Reinitialize weights
		if (commandIndex == 4 && parametrsStorage.size() == 1) {
			if (this->net1) {
				double weightsValue;
				std::stringstream strst;
				strst << std::fixed << std::setprecision(15) << parametrsStorage[0];
				strst >> weightsValue;
				this->net1->setWeights(weightsValue);
			}
			else { std::cout << "Create network first. " << this->useHelp() << std::endl; }
			continue;
		}
		//Create network
		if (commandIndex == 5 && parametrsStorage.size() == 4) {
			std::vector<size_t> counts;
			for (size_t i = 0; i < 4; i++)
			{	
				std::stringstream strst;
				strst << std::fixed << std::setprecision(15) << parametrsStorage[i];
				size_t tmp;
				strst >> tmp;
				counts.push_back(tmp);
			}
			this->net1 = new nnet::NeuralNet(counts[0], counts[1], counts[2], counts[3]);
			std::cout << this->successfullyExecuted() << std::endl;
		}
		//Train network
		if (commandIndex == 6 && parametrsStorage.size() == 1) {
			if (this->net1) {
				if (net1->studyNetworkAuto(parametrsStorage[0])) std::cout << this->successfullyExecuted() << std::endl;
				else std::cout << "Unable to open file." << std::endl;
			}
			else { std::cout << "Create network first. " << this->useHelp() << std::endl; }
			continue;
		}
		//Getresultw
		if (commandIndex == 7 ) {
			if (this->net1) {
				if (parametrsStorage.size() == this->net1->nodesCount.getInputNodesCount()) {
					
					std::vector<double> inputDataStorage;
					for (size_t i = 0; i < parametrsStorage.size(); i++)
					{
						std::stringstream strst;
						strst << std::fixed << std::setprecision(15) << parametrsStorage[i];
						double tmp;
						strst >> tmp;
						inputDataStorage.push_back(tmp);
					}
					this->net1->setData(inputDataStorage, true);
					this->net1->printResult();
				}
				else std::cout << "Input parametrs count and network input nodes count does not match." << std::endl;
			}
			else { std::cout << "Create network first. " << this->useHelp() << std::endl; }
			continue;
		}
		std::cout << "Unknown attributes. " << this->useHelp() << std::endl;
	}
}

inline __int64 nai::NetApplicationInterface::findCommand(const std::string command) const
{
	for (size_t i = 0; i < this->commandsList.size(); i++) {

		if (command.find(commandsList[i]) != std::string::npos) {

			return i;
		}
	}
	return -1;
}

inline __int64 nai::NetApplicationInterface::checkParametrsCount(const std::string& command, size_t parametrsCount) const
{
	if (command.find(" ") == 0 || command.find(" ") == std::string::npos) { return -1; }

	for (size_t i = 0; i < parametrsCount; i++)
	{

	}
	
	return 0;
}

inline void nai::NetApplicationInterface::toUpperCase(const std::string& paramString)
{
	for (size_t i = 0; i < paramString.size() && paramString[i] != ' '; i++)
	{
		paramString[i] = std::toupper(paramString[i]);
	}
}

inline std::string nai::NetApplicationInterface::useHelp() const
{
	return "Use /help to watch commands info.";
}
inline std::string nai::NetApplicationInterface::successfullyExecuted() const
{
	return "Done!";
}

