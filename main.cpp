#include <inc/anet_core.h>

// tests

void even_numbers_test(){

	network_core::NeuralNet net_core(8, 5, 2, 1);

	const size_t iterrations_count = 1000;

	for (size_t i = 0; i < iterrations_count; i++)
	{
		int input_value = additional_functions::RandomFunction((__int64)0, (__int64)127);
		
		auto binary_string = std::bitset<8>(input_value).to_string();

		std::vector<int64> input_data;

		std::for_each(binary_string.begin(), binary_string.end(), [&input_data] (auto &symbol)
		{
			input_data.push_back(symbol - '0');
		});

		std::vector<double> expected_data(2);
		expected_data[0] = 0.99;
		expected_data[1] = 0.01;

		if (input_value %2 != 0){
			std::swap(expected_data[0], expected_data[1]);
		}

		net_core.StudyOnce(input_data, expected_data);

		std::cout << "Iterration number " << i + 1 << std::endl;

		if (iterrations_count - i <= 15){

			std::cout << "Input data: " << input_value << std::endl;
			std::cout << "Result data " << std::endl; 
			net_core.PrintResult();

		}
	}

}

// main
int main()
{
	even_numbers_test();

	return 0;
}
