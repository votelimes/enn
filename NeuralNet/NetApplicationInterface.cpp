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
	this->commandsDescription.push_back("Saves current weight values to file. Use -f atribute before the file name. Ex.: /saveweight -f filename");
	this->commandsDescription.push_back("Loads weight values from a file(user should choose a file name).");
	this->commandsDescription.push_back("Reinitializes weight values. Use -v atribute before the value.with Ex.: /reinitializeweights -v 0.415");
	this->commandsDescription.push_back("Creates the network according to the parametrs. Parametrs -i (input neurons count) -h (hidden neurons count) -o (output neurons count) -l (output layers count). Ex.: /createnetwork -i3 -h15 -o1 -l2");
	this->commandsDescription.push_back("Trains the network using a two data arrays: input values and output values respectively. User should select file names.");
	this->commandsDescription.push_back("Uses a network to get result. User should write input values.");
	this->commandsDescription.push_back("Uses a network to get result. User should write input data file name.");
	this->commandsDescription.push_back("Prints current network parametrs.");
	
	//this->commandsDescription.push_back("");

	this->numberOfNetworkTrainings = 0;
	this->net1 = NULL;

}

nai::NetApplicationInterface::~NetApplicationInterface()
{
}

void nai::NetApplicationInterface::doWork()
{
	std::string command;
	__int64 commandIndex;
	while (true)
	{
		std::getline(std::cin, command);
		this->toUpperCase(command);
		commandIndex = this->findCommand(command);

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
			if (this->net1) {
				auto pos1 = command.find("-f");
				if (pos1 != std::string::npos) {
					pos1 += 2;
					if (!this->net1->writeWeightsToFile(this->getParametrString(pos1, (size_t)0, command))) { std::cout << this->successfullyExecuted() << std::endl; }
					else { std::cout << "Can't create file." << std::endl;}
				}
				else { std::cout << "Unknown attributes. " << this->useHelp() << std::endl; }
			}
			else { std::cout << "Create network first. " << this->useHelp() << std::endl; }
		} 
		//Read weights (loadWeights)
		if (commandIndex == 3) {  
			if (this->net1) {
				auto pos1 = command.find("-f");
				if (pos1 != std::string::npos) {
					pos1 += 2;
					if (!this->net1->readWeightsFromFile(this->getParametrString(pos1, (size_t)0, command))) { std::cout << "Can't read the file. (File may not exist)" << std::endl; }
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

inline __int64 nai::NetApplicationInterface::findCommand(std::string &command) const
{
	for (size_t i = 0; i < this->commandsList.size(); i++) {

		if (command.find(commandsList[i]) != std::string::npos) {

			return i;
		}
	}
	return -1;
}

inline double nai::NetApplicationInterface::getParametrNumber(size_t pos1, size_t pos2, std::string paramString) const
{
	std::stringstream sst;
	sst << std::setprecision(15) << std::fixed << this->getParametrString(pos1, pos2, paramString) << std::endl;
	double var;
	sst >> var;
	return var;
}
inline std::string nai::NetApplicationInterface::getParametrString(size_t pos1, size_t pos2, std::string paramString) const
{
	std::string retString = (pos2 != 0 ? paramString.substr(pos1, pos2) : paramString.substr(pos1));

	while (retString[0] == ' ')
	{
		retString = retString.substr(1);
	}
	
	return retString;
}

inline void nai::NetApplicationInterface::toUpperCase(std::string& paramString)
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

