#include <inc/anet_core.h>

int main()
{
	network_core::NeuralNet net_core(8, 5, 2, 1);

	auto number = 51;
	auto binary_string = std::bitset<8>(number).to_string();

	std::vector<int> binary_set;

	std::for_each(binary_string.begin(), binary_string.end(), [&binary_set] (auto &symbol)
	{
		binary_set.push_back(symbol - '0');
	});

	return 0;
}
