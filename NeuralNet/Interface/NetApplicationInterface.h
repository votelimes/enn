#ifndef NET_APPLICATION_INTERFACE_H_
#define NET_APPLICATION_INTERFACE_H_
#include "R:\Projects\NeuralNet\NeuralNet\Core\Net.h"

namespace nai {
	class NetApplicationInterface
	{
		//SECTION: DATA

		std::vector<std::string> commandsList;
		std::vector<std::string> commandsDescription;

		ann::NeuralNet *net1;

		long double trainingsCount;
		
		//SECTION: METHODS

	private:

		inline __int64 findCommand(const std::string command) const;
		inline __int64  checkParametrsCount(const std::string &command, size_t parametrsCount) const;

		inline void toUpperCase(std::string& paramString);

		inline std::string useHelp() const;
		inline std::string successfullyExecuted() const;
		inline void cls() const;

	public:
		
		NetApplicationInterface();
		~NetApplicationInterface();

		void doWork();
	};
} //namespace nai
#endif //NET_APPLICATION_INTERFACE_H_
