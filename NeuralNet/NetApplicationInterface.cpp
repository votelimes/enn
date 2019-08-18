#include "NetApplicationInterface.h"

nai::NetApplicationInterface::NetApplicationInterface()
{
	this->commandsList.push_back("/help");
	this->commandsList.push_back("/exit");
	this->commandsList.push_back("/saveweight");
	this->commandsList.push_back("/loadweight");
	this->commandsList.push_back("/reinitializeweight");
	this->commandsList.push_back("/createnetwork");
	this->commandsList.push_back("/trainnetwork");
	this->commandsList.push_back("/getresultW");
	this->commandsList.push_back("/getresultF");
	this->commandsList.push_back("/getnetworkinfo");
	
	//this->commandsList.push_back("");

	this->commandsDescription.push_back("Shows a list of all commands and their purpose.");
	this->commandsDescription.push_back("Closes a program without saving reults.");
	this->commandsDescription.push_back("Saves current weight values to file. Use -f atribute before the file name. Ex.: /saveweight -f filename.");
	this->commandsDescription.push_back("Loads weight values from a file(user should choose a file name).");
	this->commandsDescription.push_back("Reinitializes weight values with a custom value, written by user.");
	this->commandsDescription.push_back("Creates the network according to the parametrs entered by user.");
	this->commandsDescription.push_back("Trains the network using a two data arrays: input values and output values respectively. User should select file names.");
	this->commandsDescription.push_back("Uses a network to get result. User should write input values.");
	this->commandsDescription.push_back("Uses a network to get result. User should write input data file name.");
	this->commandsDescription.push_back("Prints current network parametrs.");
	
	//this->commandsDescription.push_back("");

	this->numberOfNetworkTrainings = 0;

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

		commandIndex = this->findCommand(command);

		if (commandIndex == -1) {
			std::cout << "Unknown command. Use /help to see all commands info." << std::endl;
			continue;
		}
		if (commandIndex == 0) {

			for (size_t i = 0; i < this->commandsList.size(); i++) {
				std::string helpS;
				helpS = "\n" + std::to_string(i + 1) + "." + " " + commandsList[i];
				std::cout << std::setw(25) << std::left << helpS << commandsDescription[i] << std::endl;
			}
			continue;
		}
		if (commandIndex == 1) {
			
			break;
		}
		if (commandIndex == 2) {
			
			if (this->net1) {
				auto pos1 = command.find("-f");
				if (pos1 != std::string::npos) {
					pos1 += 2;
					auto fileName = command.substr(pos1, command.size());
					while (fileName[0] == ' ')
					{
						fileName = fileName.substr(1, fileName.size());
					}

					this->net1->writeWeightsToFile(fileName);

				}
				else {
					std::cout << "Unknown attributes. Use /help to see all commands info." << std::endl;
				}
			}
			else 
			{
				std::cout << "Create network first.  Use /help to see all commands info." << std::endl;
			}
		}
		if (commandIndex == 3) {
			if (this->net1) {
				auto pos1 = command.find("-f");
				if (pos1 != std::string::npos) {
					pos1 += 2;
					auto fileName = command.substr(pos1, command.size());
					while (fileName[0] == ' ')
					{
						fileName = fileName.substr(1, fileName.size());
					}
					this->net1->readWeightsFromFile(fileName);
				}
			}
			else 
			{
				std::cout << "Create network first.  Use /help to see all commands info." << std::endl;
			}

		}
		if (commandIndex == 4) {

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

inline __int64 nai::NetApplicationInterface::getParametr(size_t pos, std::string param) const
{
	return 0;
}

