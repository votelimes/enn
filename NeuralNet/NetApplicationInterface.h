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
		inline double getParametrNumber(size_t pos1, size_t pos2, std::string paramString) const;
		inline std::string getParametrString(size_t pos1, size_t pos2, std::string paramString) const;
		
		inline void toUpperCase(std::string& paramString);

		inline std::string useHelp() const;
		inline std::string successfullyExecuted() const;

	public:
		
		NetApplicationInterface();
		~NetApplicationInterface();

		void doWork();
	};
} //namespace nai
#endif //NET_APPLICATION_INTERFACE_H_