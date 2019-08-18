#ifndef NET_APPLICATION_INTERFACE_H_
#define NET_APPLICATION_INTERFACE_H_
#include "Net.h"

namespace nai {
	class NetApplicationInterface
	{
		//SECTION: DATA

		std::vector<std::string> commandsList;
		std::vector<std::string> commandsDescription;

		nnet::NeuralNet *net1;

		long double numberOfNetworkTrainings;
		
		//SECTION: METHODS

	private:

		inline __int64 findCommand(std::string& command) const;
		inline __int64 getParametr(size_t pos, std::string param) const;
	public:
		
		NetApplicationInterface();
		~NetApplicationInterface();

		void doWork();
	};
} //namespace nai
#endif //NET_APPLICATION_INTERFACE_H_