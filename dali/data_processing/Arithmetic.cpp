#include "dali/data_processing/Arithmetic.h"

using std::vector;
using std::string;
using std::to_string;

namespace arithmetic {
    std::vector<std::string> symbols = {"+", "*", "-"};
    int NUM_SYMBOLS = symbols.size();

    vector<std::pair<vector<string>, vector<string>>> generate(int num, int expression_length) {
        vector<std::pair<vector<string>, vector<string>>> examples;
        int i = 0;
        while (i < num) {
            vector<string> example;
            auto expr_length = utils::randint(1, std::max(1, expression_length));
            bool use_operator = false;
            for (int j = 0; j < expr_length; j++) {
                if (use_operator) {
                    auto operation = symbols[utils::randint(0, NUM_SYMBOLS-1)];
                    example.push_back(operation);
                    use_operator = false;
                } else {
                    auto value = std::to_string(utils::randint(0, 9));
                    example.push_back(value);
                    use_operator = true;
                }
            }
            if (!use_operator) {
                auto value = std::to_string(utils::randint(0, 9));
                example.push_back(value);
                use_operator = true;
            }
            int result = 0;
            {
                int product_so_far = 1;
                vector<string> multiplied;
                for (auto& character : example) {
                    if (utils::in_vector(symbols, character)) {
                        if (character == "*") {
                            // do nothing
                        } else {
                            multiplied.push_back(to_string(product_so_far));
                            multiplied.push_back(character);
                            product_so_far = 1;
                        }
                    } else {
                        product_so_far *= character[0] - '0';
                    }
                }
                multiplied.push_back(to_string(product_so_far));

                string last_operator = "";
                for (auto& character: multiplied) {
                    if (utils::in_vector(symbols, character)) {
                        last_operator = character;
                    } else {
                        if (last_operator == "") {
                            result = std::stoi(character);
                        } else if (last_operator == "+") {
                            result += std::stoi(character);
                        } else if (last_operator == "-") {
                            result -= std::stoi(character);
                        } else {
                            assert(NULL == "Unknown operator.");
                        }
                    }
                }
            }
            if (result > -50 && result < 50) {
                i++;
                auto res = to_string(result);
                vector<string> character_result;
                for (int j = 0; j < res.size(); j++) {
                    character_result.emplace_back(res.begin()+j, res.begin()+j+1);
                }
                examples.emplace_back(
                    example,
                    character_result
                );
            }
        }
        return examples;
    }
}