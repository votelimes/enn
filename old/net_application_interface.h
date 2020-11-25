#ifndef NET_APPLICATION_INTERFACE_H_
#define NET_APPLICATION_INTERFACE_H_
#include "net.h"

namespace net_application_interface {
	class NetApplicationInterface
	{
		//SECTION: DATA

		std::vector<std::string> commands_list;
		std::vector<std::string> commands_description;

		network_core::NeuralNet *core_network;

		long double trainings_count;
		
		//SECTION: METHODS

	private:

		inline __int64 FindCommand(const std::string command) const;
		inline __int64  CheckParametrsCount(const std::string &command, size_t parametrs_count) const;

		inline void ToUpperCase(std::string& editable_string);

		inline std::string UseHelp() const;
		inline std::string SuccessfullyExecuted() const;
		inline void Cls() const;

	public:
		
		NetApplicationInterface();
		~NetApplicationInterface();

		void Start();
	};
} //namespace net_application_interface
#endif //NET_APPLICATION_INTERFACE_H_
