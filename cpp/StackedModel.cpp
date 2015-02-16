#include "StackedModel.h"

using std::shared_ptr;
using std::vector;
using std::make_shared;
using std::ofstream;
using std::to_string;
using std::stringstream;
using std::string;

/**
Parameters
----------

Create a vector of shared pointers to the underlying matrices
of the model. Useful for saving, loading parameters all at once
and for telling Solvers which parameters should be updated
during each training loop.

Outputs
-------

std::vector<std::shared_ptr<Mat<T>>> parameters : vector of model parameters

**/

template<typename T>
vector<typename StackedModel<T>::shared_mat> StackedModel<T>::parameters() {
	vector<shared_mat> parameters;
	parameters.push_back(embedding);

	auto decoder_params = decoder.parameters();
	parameters.insert(parameters.end(), decoder_params.begin(), decoder_params.end());
	for (auto& cell : cells) {
		auto cell_params = cell.parameters();
		parameters.insert(parameters.end(), cell_params.begin(), cell_params.end());
	}
	return parameters;
}

/**
Configuration
-------------

Return a map with keys corresponding to hyperparameters for
the model and where values are vectors of strings containing
the assignments to each hyperparameter for the loaded model.

Useful for saving the model to file and reloading it later.

Outputs
-------

std::map<std::string, std::vector< std::string >> config : configuration map

**/

template<typename T>
typename StackedModel<T>::config_t StackedModel<T>::configuration() {
	config_t config;
	config["output_size"].emplace_back(to_string(output_size));
	config["input_size"].emplace_back(to_string(input_size));
	config["vocabulary_size"].emplace_back(to_string(vocabulary_size));
	for (auto& v : hidden_sizes)
		config["hidden_sizes"].emplace_back(to_string(v));
	return config;
}

/**
Save Configuration
------------------

Save model configuration as a text file with key value pairs.
Values are vectors of string, and keys are known by the model.

Input
-----

std::string fname : where to save the configuration

**/
template<typename T>
void StackedModel<T>::save_configuration(std::string fname) {
	
	auto config = configuration();
	utils::map_to_file(config, fname);
}

template<typename T>
void StackedModel<T>::save(std::string dirname) {
	utils::ensure_directory(dirname);
	// Save the matrices:
	auto params = parameters();
	utils::save_matrices(params, dirname);
	dirname += "config.md";
	save_configuration(dirname);
}

template<typename T>
void StackedModel<T>::add_options_to_CLI(optparse::OptionParser& parser) {
	parser.set_defaults("stack_size", "4");
	parser
		.add_option("-stack", "--stack_size")
		.help("How many LSTMs should I stack ?").metavar("INT");
	parser.set_defaults("input_size", "100");
	parser
		.add_option("-i", "--input_size")
		.help("Size of the word vectors").metavar("INT");
	parser.set_defaults("hidden", "100");
	parser
		.add_option("-h", "--hidden")
		.help("How many Cells and Hidden Units should each LSTM have ?").metavar("INT");
	parser.set_defaults("decay_rate", "0.95");
	parser
		.add_option("-decay", "--decay_rate")
		.help("What decay rate should RMSProp use ?").metavar("FLOAT");
	parser.set_defaults("rho", "0.95");
	parser
		.add_option("--rho")
		.help("What rho / learning rate should the Solver use ?").metavar("FLOAT");

	parser.set_defaults("save", "");
	parser.add_option("--save")
		.help("Where to save the model to ?").metavar("FOLDER");

	parser.set_defaults("load", "");
	parser.add_option("--load")
		.help("Where to load the model from ?").metavar("FOLDER");
}

template<typename T>
StackedModel<T> StackedModel<T>::build_from_CLI(optparse::Values& options, int vocab_size, int output_size, bool verbose) {
	using utils::from_string;
	string load_location = options["load"];
	if (verbose)
		std::cout << "Load location         = " << ((load_location == "") ? "N/A" : load_location)       << std::endl;
	// Load or Construct the model
	auto model = (load_location != "") ?
		StackedModel<T>::load(load_location) :
		StackedModel<T>(
			vocab_size,
			from_string<int>(options["input_size"]),
			from_string<int>(options["hidden"]),
			from_string<int>(options["stack_size"]) < 1 ? 1 : from_string<int>(options["stack_size"]),
			output_size);
	if (verbose) {
		std::cout << ((load_location == "") ? "Constructed Stacked LSTMs" : "Loaded Model") << std::endl;
		std::cout << "Vocabulary size       = " << model.embedding->n      << std::endl;
		std::cout << "Input size            = " << model.input_size        << std::endl;
		std::cout << "Output size           = " << model.output_size       << std::endl;
		std::cout << "Stack size            = " << model.stack_size        << std::endl;
	}
	return model;
}
/**
Load
----

Load a saved copy of this model from a directory containing the
configuration file named "config.md", and from ".npy" saves of
the model parameters in the same directory.

Inputs
------

std::string dirname : directory where the model is currently saved

Outputs
-------

StackedModel<T> model : the saved model

**/
template<typename T>
StackedModel<T> StackedModel<T>::load(std::string dirname) {
	// fname should be a directory:
	utils::ensure_directory(dirname);
	// load the configuration file
	auto config_name = dirname + "config.md";

	auto config = utils::text_to_map(config_name);

	utils::assert_map_has_key(config, "input_size");
	utils::assert_map_has_key(config, "hidden_sizes");
	utils::assert_map_has_key(config, "vocabulary_size");
	utils::assert_map_has_key(config, "output_size");

	// construct the model using the map
	auto model =  StackedModel<T>(config);

	// get the current parameters of the model.
	auto params = model.parameters();

	// get the new parameters from the saved numpy files
	utils::load_matrices(params, dirname);

	return model;
}

template<typename T>
T StackedModel<T>::masked_predict_cost(
	graph_t& G,
	shared_index_mat data,
	shared_index_mat target_data,
	shared_eigen_index_vector start_loss,
	shared_eigen_index_vector codelens,
	uint offset) {

	auto initial_state    = lstm::initial_states(hidden_sizes);
	auto num_hidden_sizes = hidden_sizes.size();

	shared_mat input_vector;
	shared_mat memory;
	shared_mat logprobs;
	// shared_mat probs;
	T cost = 0.0;

	auto n = data->cols();
	for (uint i = 0; i < n-1; ++i) {
		// pick this letter from the embedding
		input_vector = G.rows_pluck(embedding, data->col(i));
		// pass this letter to the LSTM for processing
		initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
		// classifier takes as input the final hidden layer's activation:
		logprobs      = decoder.activate(G, initial_state.second[num_hidden_sizes-1]);
		cost += G.needs_backprop ? masked_cross_entropy(
										logprobs,
										i,
										start_loss,
										codelens,
										(target_data->col(i+1).array() - offset).matrix()) :
								  masked_cross_entropy_no_grad(
										logprobs,
										i,
										start_loss,
										codelens,
										(target_data->col(i+1).array() - offset).matrix());
	}
	return cost;
}

template<typename T>
T StackedModel<T>::masked_predict_cost(
	graph_t& G,
	shared_index_mat data,
	shared_index_mat target_data,
	uint start_loss,
	shared_eigen_index_vector codelens,
	uint offset) {

	auto initial_state    = lstm::initial_states(hidden_sizes);
	auto num_hidden_sizes = hidden_sizes.size();

	shared_mat input_vector;
	shared_mat memory;
	shared_mat logprobs;
	// shared_mat probs;
	T cost = 0.0;

	auto n = data->cols();
	for (uint i = 0; i < n-1; ++i) {
		// pick this letter from the embedding
		input_vector = G.rows_pluck(embedding, data->col(i));
		// pass this letter to the LSTM for processing
		initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
		// classifier takes as input the final hidden layer's activation:
		logprobs      = decoder.activate(G, initial_state.second[num_hidden_sizes-1]);
		cost += G.needs_backprop ? masked_cross_entropy(
										logprobs,
										i,
										start_loss,
										codelens,
										(target_data->col(i+1).array() - offset).matrix()) :
								  masked_cross_entropy_no_grad(
										logprobs,
										i,
										start_loss,
										codelens,
										(target_data->col(i+1).array() - offset).matrix());
	}
	return cost;
}

// Private method that names the parameters
// For better debugging and reference
template<typename T>
void StackedModel<T>::name_parameters() {
	embedding->set_name("Embedding");
	decoder.W->set_name("Decoder W");
	decoder.b->set_name("Decoder Bias");
}

// Private method for building lstm cells:
template<typename T>
void StackedModel<T>::construct_LSTM_cells() {
	cells = StackedCells<lstm>(input_size, hidden_sizes);
}

template<typename T>
StackedModel<T>::StackedModel (int _vocabulary_size, int _input_size, int hidden_size, int _stack_size, int _output_size)
	:
	input_size(_input_size),
	output_size(_output_size),
	vocabulary_size(_vocabulary_size),
	stack_size(_stack_size),
	decoder(hidden_size, _output_size) {

	embedding = make_shared<mat>(vocabulary_size, input_size, (T) -0.05, (T) 0.05);
	for (int i = 0; i < stack_size;i++)
		hidden_sizes.emplace_back(hidden_size);
	construct_LSTM_cells();
	name_parameters();
}

using utils::from_string;

/**
StackedModel Constructor from configuration map
----------------------------------------------------

Construct a model from a map of configuration parameters.
Useful for reinitializing a model that was saved to a file
using the `utils::file_to_map` function to obtain a map of
configurations.

Inputs
------

std::map<std::string, std::vector<std::string>& config : model hyperparameters

**/
template<typename T>
StackedModel<T>::StackedModel (
	const typename StackedModel<T>::config_t& config)
	:
	vocabulary_size(from_string<int>(config.at("vocabulary_size")[0])),
	output_size(from_string<int>(config.at("output_size")[0])),
	input_size(from_string<int>(config.at("input_size")[0])),
	stack_size(config.at("hidden_sizes").size()),
	decoder(
		from_string<int>(config.at("hidden_sizes")[config.at("hidden_sizes").size()-1]),
		from_string<int>(config.at("output_size")[0]))
{
	embedding = make_shared<mat>(vocabulary_size, input_size, (T) -0.05, (T) 0.05);
	for (auto& v : config.at("hidden_sizes"))
		hidden_sizes.emplace_back(from_string<int>(v));

	construct_LSTM_cells();
	name_parameters();
}

template<typename T>
StackedModel<T>::StackedModel (int _vocabulary_size, int _input_size, int _output_size, std::vector<int>& _hidden_sizes)
	:
	input_size(_input_size),
	output_size(_output_size),
	vocabulary_size(_vocabulary_size),
	stack_size(_hidden_sizes.size()),
	hidden_sizes(_hidden_sizes),
	decoder(_hidden_sizes[_hidden_sizes.size()-1], _output_size) {

	embedding = make_shared<mat>(vocabulary_size, input_size, (T) -0.05, (T) 0.05);
	construct_LSTM_cells();
	name_parameters();
}

// Nested Templates !!
template<typename T>
template<typename K>
std::vector<int> StackedModel<T>::reconstruct(
    K example,
    int eval_steps,
    int symbol_offset) {

	graph_t G(false);
	shared_mat input_vector;
	auto initial_state = lstm::initial_states(hidden_sizes);
	auto n = example.cols() * example.rows();
	for (uint i = 0; i < n; ++i) {
		// pick this letter from the embedding
		input_vector  = G.row_pluck(embedding, example(i));
		// pass this letter to the LSTM for processing
		initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
		// decoder takes as input the final hidden layer's activation:
	}
	vector<int> outputs;
	auto last_symbol = argmax(decoder.activate(G, initial_state.second[stack_size-1]));
	outputs.emplace_back(last_symbol);
	last_symbol += symbol_offset;

	for (uint j = 0; j < eval_steps - 1; j++) {
		input_vector  = G.row_pluck(embedding, last_symbol);
		initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
		last_symbol   = argmax(decoder.activate(G, initial_state.second[stack_size-1]));
		outputs.emplace_back(last_symbol);
		last_symbol += symbol_offset;
	}
	return outputs;
}

template<typename T>
template<typename K>
std::vector<utils::OntologyBranch::shared_branch> StackedModel<T>::reconstruct_lattice(
    K example,
    utils::OntologyBranch::shared_branch root,
    int eval_steps) {

	graph_t G(false);
	shared_mat input_vector;
	shared_mat memory;
	auto pos = root;
	auto initial_state = lstm::initial_states(hidden_sizes);
	auto n = example.cols() * example.rows();
	for (uint i = 0; i < n; ++i) {
		// pick this letter from the embedding
		input_vector  = G.row_pluck(embedding, example(i));
		// pass this letter to the LSTM for processing
		initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
		// decoder takes as input the final hidden layer's activation:
	}
	vector<utils::OntologyBranch::shared_branch> outputs;
	// Take the argmax over the available options (0 for go back to
	// root, and 1..n for the different children of the current position)
	auto last_turn = argmax_slice(decoder.activate(G, initial_state.second[stack_size-1]), 0, pos->children.size() + 1);
	// if the turn is 0 go back to root, else go to one of the children using
	// the lattice pointers:
	pos = (last_turn == 0) ? root : pos->children[last_turn-1];
	// add this decision to the output :
	outputs.emplace_back(pos);
	for (uint j = 0; j < eval_steps - 1; j++) {
		input_vector  = G.row_pluck(embedding, pos->id);
		initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
		last_turn     = argmax_slice(decoder.activate(G, initial_state.second[stack_size-1]), 0, pos->children.size() + 1);
		pos           = (last_turn == 0) ? root : pos->children[last_turn-1];
		outputs.emplace_back(pos);
	}
	return outputs;
}

// Nested Templates !!
template<typename T>
template<typename K>
string StackedModel<T>::reconstruct_string(
    K example,
    const utils::Vocab& lookup_table,
    int eval_steps,
    int symbol_offset) {
	auto reconstruction = reconstruct(example, eval_steps, symbol_offset);
	stringstream rec;
	for (auto& cat : reconstruction) {
		rec << (
			(cat < lookup_table.index2word.size()) ?
				lookup_table.index2word.at(cat) :
				(
					cat == lookup_table.index2word.size() ? "**END**" : "??"
				)
			) << ", ";
	}
	return rec.str();
}

// Nested Templates !!
template<typename T>
template<typename K>
string StackedModel<T>::reconstruct_lattice_string(
    K example,
    utils::OntologyBranch::shared_branch root,
    int eval_steps) {
	auto reconstruction = reconstruct_lattice(example, root, eval_steps);
	stringstream rec;
	for (auto& cat : reconstruction)
		rec << ((&(*cat) == &(*root)) ? "⟲" : cat->name) << ", ";
	return rec.str();
}

typedef Eigen::Block< Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic>, 1, Eigen::Dynamic, !Eigen::RowMajor> index_row;
typedef Eigen::VectorBlock< index_row, Eigen::Dynamic> sliced_row;

typedef Eigen::Block< Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Dynamic, 1, !Eigen::RowMajor> index_col;
typedef Eigen::VectorBlock< index_col, Eigen::Dynamic> sliced_col;

template string StackedModel<float>::reconstruct_string(sliced_row, const utils::Vocab&, int, int);
template string StackedModel<double>::reconstruct_string(sliced_row, const utils::Vocab&, int, int);

template string StackedModel<float>::reconstruct_string(index_row, const utils::Vocab&, int, int);
template string StackedModel<double>::reconstruct_string(index_row, const utils::Vocab&, int, int);

template string StackedModel<float>::reconstruct_string(sliced_col, const utils::Vocab&, int, int);
template string StackedModel<double>::reconstruct_string(sliced_col, const utils::Vocab&, int, int);

template string StackedModel<float>::reconstruct_string(index_col, const utils::Vocab&, int, int);
template string StackedModel<double>::reconstruct_string(index_col, const utils::Vocab&, int, int);

template vector<int> StackedModel<float>::reconstruct(sliced_row, int, int);
template vector<int> StackedModel<double>::reconstruct(sliced_row, int, int);

template vector<int> StackedModel<float>::reconstruct(index_row, int, int);
template vector<int> StackedModel<double>::reconstruct(index_row, int, int);

template vector<int> StackedModel<float>::reconstruct(sliced_col, int, int);
template vector<int> StackedModel<double>::reconstruct(sliced_col, int, int);

template vector<int> StackedModel<float>::reconstruct(index_col, int, int);
template vector<int> StackedModel<double>::reconstruct(index_col, int, int);

template vector<utils::OntologyBranch::shared_branch> StackedModel<float>::reconstruct_lattice(sliced_row, utils::OntologyBranch::shared_branch, int);
template vector<utils::OntologyBranch::shared_branch> StackedModel<double>::reconstruct_lattice(sliced_row, utils::OntologyBranch::shared_branch, int);

template vector<utils::OntologyBranch::shared_branch> StackedModel<float>::reconstruct_lattice(index_row, utils::OntologyBranch::shared_branch, int);
template vector<utils::OntologyBranch::shared_branch> StackedModel<double>::reconstruct_lattice(index_row, utils::OntologyBranch::shared_branch, int);

template vector<utils::OntologyBranch::shared_branch> StackedModel<float>::reconstruct_lattice(sliced_col, utils::OntologyBranch::shared_branch, int);
template vector<utils::OntologyBranch::shared_branch> StackedModel<double>::reconstruct_lattice(sliced_col, utils::OntologyBranch::shared_branch, int);

template vector<utils::OntologyBranch::shared_branch> StackedModel<float>::reconstruct_lattice(index_col, utils::OntologyBranch::shared_branch, int);
template vector<utils::OntologyBranch::shared_branch> StackedModel<double>::reconstruct_lattice(index_col, utils::OntologyBranch::shared_branch, int);

template string StackedModel<float>::reconstruct_lattice_string(sliced_row, utils::OntologyBranch::shared_branch, int);
template string StackedModel<double>::reconstruct_lattice_string(sliced_row, utils::OntologyBranch::shared_branch, int);

template string StackedModel<float>::reconstruct_lattice_string(index_row, utils::OntologyBranch::shared_branch, int);
template string StackedModel<double>::reconstruct_lattice_string(index_row, utils::OntologyBranch::shared_branch, int);

template string StackedModel<float>::reconstruct_lattice_string(sliced_col, utils::OntologyBranch::shared_branch, int);
template string StackedModel<double>::reconstruct_lattice_string(sliced_col, utils::OntologyBranch::shared_branch, int);

template string StackedModel<float>::reconstruct_lattice_string(index_col, utils::OntologyBranch::shared_branch, int);
template string StackedModel<double>::reconstruct_lattice_string(index_col, utils::OntologyBranch::shared_branch, int);

template class StackedModel<float>;
template class StackedModel<double>;