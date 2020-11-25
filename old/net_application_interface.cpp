#include "net_application_interface.h"

net_application_interface::NetApplicationInterface::NetApplicationInterface()
{
	this->commands_list.push_back("/HELP"); // 0
	this->commands_list.push_back("/EXIT"); // 1
	this->commands_list.push_back("/SAVEW"); // 2
	this->commands_list.push_back("/LOADW"); // 3
	this->commands_list.push_back("/RWEIGHTS"); // 4
	this->commands_list.push_back("/CREATEN"); // 5
	this->commands_list.push_back("/TRAINN"); // 6
	this->commands_list.push_back("/GETRTW"); // 7
	this->commands_list.push_back("/GETRF"); // 8
	this->commands_list.push_back("/PRINTNI"); // 9
	this->commands_list.push_back("/PRINTW"); // 10
	this->commands_list.push_back("/CLS"); // 11
	this->commands_list.push_back("/SETLR"); //12
	
	//this->commands_list.push_back("");

	this->commands_description.push_back("Shows a list of all commands and their purpose.");
	this->commands_description.push_back("Closes a program without saving reults.");
	this->commands_description.push_back("Saves current weight values to file. Ex.: /saveweights FILENAME");
	this->commands_description.push_back("Loads weight values from a file. Ex.: /saveweights FILENAME");
	this->commands_description.push_back("Reinitializes weight values. Ex.: /reinitializeweights 0.415");
	this->commands_description.push_back("Creates the network according to the parametrs. Parametrs (input neurons count) (hidden neurons count) (output neurons count) (output layers count). Ex.: /createnetwork 3 15 1 2");
	this->commands_description.push_back("Trains the network a certain number of times(1 is minimum). Ex.: /trainnetworkt FILENAME NUMBEROFTIMES");
	this->commands_description.push_back("Uses a network to get result. Input values count respectevly to network input layer neurons count. Ex.: /getresultw INPUT1 INPUT2 INPUT3...");
	this->commands_description.push_back("Uses a network to get result. Input data should be represented as file. Produce a output data file. Ex.: /getresultf INPUTDATAFILENAME OUTPUTDATAFILENAME");
	this->commands_description.push_back("Prints current network parametrs.");
	this->commands_description.push_back("Prints current network weights.");
	this->commands_description.push_back("Clears terminal screen.");
	this->commands_description.push_back("Sets learning rate(number). Ex.: /setlr RATE");
	//this->commands_description.push_back("");

	this->trainings_count = 0;
	this->core_network = NULL;

}

void net_application_interface::NetApplicationInterface::Start()
{
	while (true)
	{
		std::string command;
		std::cin >> command;
		this->ToUpperCase(command);
		__int64 command_index = this->FindCommand(command);
		std::vector<std::string> parametrs_storage;
		parametrs_storage.clear();
		while (std::cin.peek() != 10)
		{
			std::string buffer;
			std::cin >> buffer;
			parametrs_storage.push_back(buffer);
		}
		//unknownCommand
		if (command_index == -1) {
			std::cout << "Unknown command. " << this->UseHelp() << std::endl;
			continue;
		}
		//help
		if (command_index == 0) {

			for (size_t i = 0; i < this->commands_list.size(); i++) {
				std::string help_string;
				help_string = std::to_string(i + 1) + "." + " " + commands_list[i];
				std::cout << std::setw(25) << std::left << help_string << commands_description[i] << std::endl;
			}
			continue;
		}
		//exit
		if (command_index == 1) { 
			
			break;
		}
		//savew
		if (command_index == 2 && parametrs_storage.size() == 1) {
			if (this->core_network) {

				if (!this->core_network->WriteWeightsFile(parametrs_storage[0])) std::cout << this->SuccessfullyExecuted() << std::endl;
				else std::cout << "Unable to create file." << std::endl;
			}
			else { std::cout << "Create network first. " << this->UseHelp() << std::endl; }
			continue;
		} 
		//loadw
		if (command_index == 3 && parametrs_storage.size() == 1) {
			if (this->core_network) {
				__int64 retParam = this->core_network->ReadWeightsFile(parametrs_storage[0]);
				
				if (retParam == 1) std::cout << "Unable to open file." << std::endl;
				if (retParam == 2) std::cout << "Layers and nodes sizes does not match." << std::endl;
				else std::cout << this->SuccessfullyExecuted() << std::endl;
			}
			else { std::cout << "Create network first. " << this->UseHelp() << std::endl; }
			continue;
		} 
		//Rweights
		if (command_index == 4 && parametrs_storage.size() == 1) {
			if (this->core_network) {
				double weightsValue;
				std::stringstream strst;
				strst << std::fixed << std::setprecision(15) << parametrs_storage[0];
				strst >> weightsValue;
				this->core_network->SetWeights(weightsValue);
			}
			else { std::cout << "Create network first. " << this->UseHelp() << std::endl; }
			continue;
		}
		//Createn
		if (command_index == 5 && parametrs_storage.size() == 4) {
			std::vector<size_t> counts;
			for (size_t i = 0; i < 4; i++)
			{	
				std::stringstream strst;
				strst << std::fixed << std::setprecision(15) << parametrs_storage[i];
				size_t tmp;
				strst >> tmp;
				counts.push_back(tmp);
			}
			this->core_network = new network_core::NeuralNet(counts[0], counts[1], counts[2], counts[3]);
			std::cout << this->SuccessfullyExecuted() << std::endl;
			continue;
		}
		//Trainn
		if (command_index == 6 && parametrs_storage.size() == 2) {
			if (this->core_network) {
				if (!core_network->StudyFileMT(parametrs_storage[0])) {

					__int64 trainings_count{ 1 };
					std::cout << "Number 1 training completed." << std::endl;
					for (size_t i = 1; i < static_cast<size_t>(std::stoi(parametrs_storage[1])); i++)
					{
						trainings_count++;
						core_network->StudyFileMT(parametrs_storage[0]);
						std::cout << "Number " << i + 1 << " training completed." << std::endl;
					}
					std::cout << "The network has been trained " << trainings_count << " times." << std::endl;
					std::cout << this->SuccessfullyExecuted() << std::endl;
				}
				else std::cout << "Unable to open file." << std::endl;
			}
			else { std::cout << "Create network first. " << this->UseHelp() << std::endl; }
			continue;
		}
		//Getrw
		if (command_index == 7 ) {
			if (this->core_network) {
				if (parametrs_storage.size() == this->core_network->nodes_count.GetInputNodesCount()) {
					
					std::vector<double> inputData;
					for (size_t i = 0; i < parametrs_storage.size(); i++)
					{
						std::stringstream strst;
						strst << std::fixed << std::setprecision(15) << parametrs_storage[i];
						double tmp;
						strst >> tmp;
						inputData.push_back(tmp);
					}
					std::vector<double>* outputDataP = this->core_network->ProduceResult(inputData);

					
				}
				else std::cout << "Input parametrs count and network input nodes count does not match." << std::endl;
			}
			else { std::cout << "Create network first. " << this->UseHelp() << std::endl; }
			continue;
		}
		//Getrf
		if (command_index == 8 && parametrs_storage.size() == 2) {
			if (this->core_network) {
				if (this->core_network->ProduceResult(parametrs_storage[0], parametrs_storage[1])) {
					std::cout << this->SuccessfullyExecuted();
				}
				else std::cout << "Unknown parametrs or unable to open file." << std::endl;
			}
			else std::cout << "Create network first." << std::endl;
			continue;
		}
		//PRINTNI
		if (command_index == 9 && parametrs_storage.size() == 0) {
			if (this->core_network) {
				std::cout << "Input nodes: " << this->core_network->nodes_count.GetInputNodesCount() << std::endl;
				std::cout << "Hidden nodes: " << this->core_network->nodes_count.GetHiddenNodesCount() << std::endl;
				std::cout << "Output nodes: " << this->core_network->nodes_count.GetOutputNodesCount() << std::endl;
				std::cout << "Hidden layers: " << this->core_network->nodes_count.GetHiddenLayersCount() << std::endl;
				std::cout << "Total layers: " << this->core_network->nodes_count.GetTotalLayersCount() << std::endl;
				std::cout << "Learning rate: " << this->core_network->GetLearningRate() << std::endl;
			}
			else std::cout << "Create network first." << std::endl;
			continue;
		}
		//PRINTW
		if (command_index == 10 && parametrs_storage.size() == 0) {
			if (this->core_network) {
				this->core_network->PrintWeights();
			}
			else std::cout << "Create network first." << std::endl;
			continue;
		}
		//CLS
		if (command_index == 11 && parametrs_storage.size() == 0) {
			this->Cls();
			continue;
		}
		//SETLR
		if (command_index == 12 && parametrs_storage.size() == 1) {
			if (this->core_network) {
				this->core_network->SetLearningRate(stod(parametrs_storage[0]));
			}
			else std::cout << "Create network first." << std::endl;
			continue;
		}

		std::cout << "Unknown parametrs. " << this->UseHelp() << std::endl;
	}
}

inline __int64 net_application_interface::NetApplicationInterface::FindCommand(const std::string command) const
{
	for (size_t i = 0; i < this->commands_list.size(); i++) {

		if (command.find(commands_list[i]) != std::string::npos) {

			return i;
		}
	}
	return -1;
}

inline __int64 net_application_interface::NetApplicationInterface::CheckParametrsCount(const std::string& command, size_t parametrs_count) const
{
	if (command.find(" ") == 0 || command.find(" ") == std::string::npos) { return -1; }

	for (size_t i = 0; i < parametrs_count; i++)
	{

	}
	
	return 0;
}

inline void net_application_interface::NetApplicationInterface::ToUpperCase(std::string& editable_string)
{
	for (auto i = 0; i < editable_string.size() && editable_string[i] != ' '; i++)
	{
		editable_string[i] = std::toupper(editable_string[i]);
	}
}

inline std::string net_application_interface::NetApplicationInterface::UseHelp() const
{
	return "Use /help to watch commands info.";
}
inline std::string net_application_interface::NetApplicationInterface::SuccessfullyExecuted() const
{
	return "\nDone!\n____________________";
}

inline void net_application_interface::NetApplicationInterface::Cls() const
{
	std::cout << "\x1B[2J\x1B[H";
}

