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
		if (commandIndex == 2) {  
			if (parametrsStorage.size() == 1) {
				if (this->net1) {

					if (!this->net1->writeWeightsToFile(parametrsStorage[0])) { std::cout << this->successfullyExecuted() << std::endl; }
					else { std::cout << "Can't create file." << std::endl; }
				}
				else { std::cout << "Create network first. " << this->useHelp() << std::endl; }
			}
			else { std::cout << "Unknown attributes. " << this->useHelp() << std::endl; }
		} 
		//Read weights (loadWeights)
		if (commandIndex == 3) {  
			if (this->net1) {
				auto pos1 = command.find("-f");
				if (pos1 != std::string::npos) {
					pos1 += 2;
					if (!this->net1->readWeightsFromFile(this->getParametrsString(pos1, (size_t)0, command))) { std::cout << "Can't read the file. (File may not exist)" << std::endl; }
				}
			}
			else { std::cout << "Create network first. " << this->useHelp() << std::endl; }
		} 
		//Reinitialize weights
		if (commandIndex == 4) {
			if (this->net1) {
				auto pos1 = command.find("-v");
				if (pos1 != std::string::npos) {
					pos1 += 2;
					this->net1->setWeights(this->getParametrNumber(pos1, 0, command));
				}
				else { std::cout << "Unknown attributes. " << this->useHelp() << std::endl; }
			}
			else { std::cout << "Create network first. " << this->useHelp() << std::endl; }
		}
		//Create network
		if (commandIndex == 5) {
			auto pos1 = command.find("-i");
			auto pos2 = command.find("-h");
			auto pos3 = command.find("-o");
			auto pos4 = command.find("-l");

			if (pos1 != std::string::npos && pos2 != std::string::npos && pos3 != std::string::npos && pos4 != std::string::npos) {

				pos1 += 2; pos2 += 2; pos3 += 2; pos4 += 2;
				this->net1 = new nnet::NeuralNet( static_cast<size_t>(this->getParametrNumber(pos1, pos2 - 1, command)), static_cast<size_t>(this->getParametrNumber(pos2, pos3 - 1, command)), static_cast<size_t>(this->getParametrNumber(pos3, pos4 - 1, command)), static_cast<size_t>(this->getParametrNumber(pos4, 0, command)));
				std::cout << this->successfullyExecuted() << std::endl;
			}
			else { std::cout << "Unknown attributes. " << this->useHelp() << std::endl; }
		}
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

inline double nai::NetApplicationInterface::getParametrNumber(const size_t pos1, const size_t pos2, const std::string paramString) const
{
	std::stringstream sst;
	sst << std::setprecision(15) << std::fixed << this->getParametrsString(pos1, pos2, paramString) << std::endl;
	double var;
	sst >> var;
	return var;
}
inline std::string nai::NetApplicationInterface::getParametrsString(std::vector<std::string> &parametrsStorage)
{
	for (size_t i = 0; i < length; i++)
	{

	}
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

