#include <algorithm>
#include <atomic>
#include <Eigen/Eigen>
#include <fstream>
#include <ostream>
#include <fstream>
#include <iterator>
#include <chrono>

#include "dali/core.h"
#include "dali/utils.h"
#include "dali/utils/NlpUtils.h"
#include "dali/data_processing/Paraphrase.h"
#include "dali/data_processing/Glove.h"
#include "dali/visualizer/visualizer.h"

using namespace std::placeholders;
using std::atomic;
using std::chrono::seconds;
using std::ifstream;
using std::make_shared;
using std::min;
using std::ofstream;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::tie;
using std::tuple;
using std::vector;
using utils::assert2;
using utils::vsum;
using utils::Vocab;
using utils::MS;

static const int ADADELTA_TYPE = 0;
static const int ADAGRAD_TYPE  = 1;
static const int SGD_TYPE      = 2;
static const int ADAM_TYPE     = 3;
static const int RMSPROP_TYPE  = 4;

typedef double REAL_t;


// training flags
DEFINE_int32(minibatch,                5,          "What size should be used for the minibatches ?");
DEFINE_int32(patience,                 5,          "How many unimproving epochs to wait through before witnessing progress ?");
// files
DEFINE_string(results_file,            "",         "Where to save test performance.");
DEFINE_string(save_location,           "",         "Where to save test performance.");
DEFINE_string(test,                    "",         "Where is the test set?");
// solvers
DEFINE_string(solver,                  "adadelta", "What solver to use (adadelta, sgd, adam)");
DEFINE_double(learning_rate,           0.01,       "Learning rate for SGD and Adagrad.");
DEFINE_double(reg,                     0.0,        "What penalty to place on L2 norm of weights?");
// model
DEFINE_int32(input_size,               100,        "Size of embeddings.");
DEFINE_int32(hidden,                   100,        "What hidden size to use.");
DEFINE_int32(stack_size,               2,          "How tall is the great LSTM tower.");
DEFINE_int32(gate_second_order,        50,         "How many second order terms to consider in gate");
DEFINE_bool(lstm_shortcut,             true,       "Should shortcut be used in LSTMs");
DEFINE_bool(lstm_memory_feeds_gates,   true,       "Should memory be fed to gates in LSTMs");
DEFINE_double(dropout,                 0.3,        "How much dropout noise to add to the problem ?");
DEFINE_double(memory_penalty,          0.1,        "Coefficient in front of memory penalty");
DEFINE_string(memory_penalty_curve,    "flat",     "Type of annealing used on gate memory penalty (flat, linear, square)");
// features
DEFINE_bool(svd_init,                  true,       "Initialize weights using SVD?");
DEFINE_bool(end_token,                 true,       "Whether to add a token indicating end of sentence.");
ThreadPool* pool;


template<typename T>
class SparseStackedLSTM : public StackedLSTM<T> {
    public:
        typedef typename StackedLSTM<T>::state_t state_t;

        SecondOrderCombinator<T> gate_encoder;

        SparseStackedLSTM() : StackedLSTM<T>() {
        }

        SparseStackedLSTM(const SparseStackedLSTM<T>& other, bool copy_w, bool copy_dw) :
            StackedLSTM<T>(other, copy_w, copy_dw),
            gate_encoder(other.gate_encoder, copy_w, copy_dw) {
        }

        SparseStackedLSTM shallow_copy() const {
            return SparseStackedLSTM(*this, false, true);
        }

        vector<Mat<T>> parameters() const {
            auto params = StackedLSTM<T>::parameters();
            auto gate_params = gate_encoder.parameters();
            params.insert(params.end(), gate_params.begin(), gate_params.end());
            return params;
        }

        SparseStackedLSTM(int input_size,
                          vector<int> hidden_sizes,
                          int gate_second_order,
                          bool shortcut,
                          bool memory_feeds_gates) :
                StackedLSTM<T>(input_size, hidden_sizes, shortcut, memory_feeds_gates),
                gate_encoder(input_size, vsum(hidden_sizes), gate_second_order) {
        }

        Mat<T> activate_gate(Mat<T> input, Mat<T> hidden) const {
            return gate_encoder.activate(input, hidden).sum().sigmoid();
        }

        // returns next state and memory
        std::tuple<state_t, Mat<T>> activate(Mat<T> input, state_t prev_state, T dropout_probability) const {
            auto current_hiddens =  MatOps<T>::vstack(LSTM<T>::State::hiddens(prev_state));
            auto gate_memory     =  activate_gate(input, current_hiddens);
            auto gated_input     =  input.eltmul_broadcast_rowwise(gate_memory);
            auto next_state      =  StackedLSTM<T>::activate(prev_state, gated_input, dropout_probability);

            return std::make_tuple(next_state, gate_memory);
        }

        // returns last state and memory at every step
        std::tuple<state_t, vector<Mat<T>>> activate_sequence(vector<Mat<T>> inputs,
                                                              state_t state,
                                                              T dropout_probability) const {
            vector<Mat<T>> memories;
            for (auto& input: inputs) {
                Mat<T> memory;
                std::tie(state, memory) = activate(input, state, dropout_probability);
                memories.push_back(memory);
            }
            return make_tuple(state, memories);
        }
};

template<typename T>
class ParaphraseModel {
    typedef typename StackedLSTM<T>::state_t state_t;

    public:
        int input_size;
        int vocab_size;
        vector<int> hidden_sizes;
        T dropout_probability;

        SparseStackedLSTM<T> sentence_encoder;
        Mat<T> end_of_sentence_token;
        Mat<T> embedding_matrix;


        ParaphraseModel(int input_size,
                        int vocab_size,
                        vector<int> hidden_sizes,
                        int gate_second_order,
                        T dropout_probability) :
                input_size(input_size),
                vocab_size(vocab_size),
                hidden_sizes(hidden_sizes),
                dropout_probability(dropout_probability),
                sentence_encoder(input_size,
                                 hidden_sizes,
                                 gate_second_order,
                                 FLAGS_lstm_shortcut,
                                 FLAGS_lstm_memory_feeds_gates),
                end_of_sentence_token(input_size, 1, weights<T>::uniform(1.0 / input_size)),
                embedding_matrix(vocab_size, input_size, weights<T>::uniform(1.0 / input_size))
                 {
        }

        ParaphraseModel(const ParaphraseModel& other, bool copy_w, bool copy_dw) :
                input_size(other.input_size),
                vocab_size(other.vocab_size),
                hidden_sizes(other.hidden_sizes),
                dropout_probability(other.dropout_probability),
                sentence_encoder(other.sentence_encoder, copy_w, copy_dw),
                end_of_sentence_token(other.end_of_sentence_token, copy_w, copy_dw),
                embedding_matrix(other.embedding_matrix, copy_w, copy_dw) {

        }

        ParaphraseModel<T> shallow_copy() const {
            return ParaphraseModel<T>(*this, false, true);
        }

        vector<Mat<T>> parameters() const {
            vector<Mat<T>> res;

            auto params = sentence_encoder.parameters();
            res.insert(res.end(), params.begin(), params.end());

            for (auto& matrix: { end_of_sentence_token,
                                 embedding_matrix
                                 }) {
                res.emplace_back(matrix);
            }
            return res;
        }

        // returns sentence and vector of memories
        std::tuple<Mat<T>, vector<Mat<T>>> encode_sentence(vector<uint> sentence, bool use_dropout) const {
            vector<Mat<T>> embeddings;
            for (auto& word_idx: sentence) {
                embeddings.emplace_back(embedding_matrix[word_idx]);
            }
            if (FLAGS_end_token)
                embeddings.push_back(end_of_sentence_token);

            auto state = sentence_encoder.initial_states();
            vector<Mat<T>> memories;
            std::tie(state, memories) = sentence_encoder.activate_sequence(
                    embeddings,
                    state,
                    use_dropout ? dropout_probability : 0.0);
            auto sentence_hidden = MatOps<T>::vstack(LSTM<T>::State::hiddens(state));
            return std::make_tuple(sentence_hidden, memories);
        }

        std::tuple<Mat<T>, vector<Mat<T>>, vector<Mat<T>>> similarity(vector<uint> sentence1,
                                                                      vector<uint> sentence2,
                                                                      bool use_dropout) const {
            Mat<T> sentence1_hidden, sentence2_hidden;
            vector<Mat<T>> sentence1_memories, sentence2_memories;
            std::tie(sentence1_hidden, sentence1_memories) =
                    encode_sentence(sentence1, use_dropout);
            std::tie(sentence2_hidden, sentence2_memories) =
                    encode_sentence(sentence2, use_dropout);

            auto s1_norm = sentence1_hidden.square().sum().sqrt();
            auto s2_norm = sentence2_hidden.square().sum().sqrt();
            auto similarity_score = (sentence1_hidden * sentence2_hidden ).sum() / (s1_norm * s2_norm);
            return std::make_tuple(similarity_score, sentence1_memories, sentence2_memories);
        }

        tuple<Mat<T>, vector<Mat<T>>, vector<Mat<T>>> error(
                const paraphrase::numeric_example_t& example) const {
            vector<uint> sentence1, sentence2;
            double correct_score;
            std::tie(sentence1, sentence2, correct_score) = example;

            Mat<T> similarity_score;
            vector<Mat<T>> memory1, memory2;
            std::tie(similarity_score, memory1, memory2) =
                    similarity(sentence1, sentence2, true);

            auto error_value = (similarity_score - correct_score)^2;

            return std::make_tuple(error_value, memory1, memory2);
        }

        tuple<double, vector<double>, vector<double>> predict_with_memories(vector<uint> sentence1, vector<uint> sentence2) const {
            graph::NoBackprop nb;
            vector<Mat<T>> memory1_mat, memory2_mat;
            Mat<T> similarity_mat;
            std::tie(similarity_mat, memory1_mat, memory2_mat) =
                    similarity(sentence1, sentence2, false);

            auto extract_double  = [](Mat<T> m) { return m.w()(0,0); };
            vector<double> memory1, memory2;

            std::transform(memory1_mat.begin(), memory1_mat.end(), std::back_inserter(memory1), extract_double);
            std::transform(memory2_mat.begin(), memory2_mat.end(), std::back_inserter(memory2), extract_double);

            auto similarity_score = extract_double(similarity_mat);
            return make_tuple(similarity_score, memory1, memory2);
        }

        double predict(vector<uint> sentence1, vector<uint> sentence2) const {
            double score;
            vector<double> ignored1, ignored2;
            std::tie(score, ignored1, ignored2) = predict_with_memories(sentence1, sentence2);

            return score;
        }

};

typedef ParaphraseModel<REAL_t> model_t;


int main (int argc,  char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "Textual Similarity using single LSTM\n"
        "------------------------------------\n"
        "\n"
        " @author Jonathan Raiman & Szymon Sidor\n"
        " @date May 20th 2015"
    );
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    int memory_penalty_curve_type;
    if (FLAGS_memory_penalty_curve == "flat") {
        memory_penalty_curve_type = 0;
    } else if (FLAGS_memory_penalty_curve == "linear") {
        memory_penalty_curve_type = 1;
    } else if (FLAGS_memory_penalty_curve == "square") {
        memory_penalty_curve_type = 2;
    } else {
        utils::assert2(false, "memory_penalty_curve can only be flat, linear, or square.");
    }

    auto epochs = FLAGS_epochs;
    int rampup_time = 10;

    auto paraphrase_data = paraphrase::STS_2015::load_train();
    auto word_vocab      = Vocab();

    word_vocab = Vocab(paraphrase::get_vocabulary(paraphrase_data, FLAGS_min_occurence), true);

    auto vocab_size     = word_vocab.size();
    auto dataset        = paraphrase::convert_to_indexed_minibatches(
        word_vocab,
        paraphrase_data,
        FLAGS_minibatch
    );
    // No validation set yet...
    decltype(dataset) validation_set;
    {
        auto paraphrase_valid_data =  paraphrase::STS_2015::load_dev();
        validation_set = paraphrase::convert_to_indexed_minibatches(
            word_vocab,
            paraphrase_valid_data,
            FLAGS_minibatch
        );
    }

    pool = new ThreadPool(FLAGS_j);

    vector<int> hidden_sizes;

    auto model = model_t(FLAGS_input_size,
                         word_vocab.size(),
                         vector<int>(FLAGS_stack_size, FLAGS_hidden),
                         FLAGS_gate_second_order,
                         FLAGS_dropout);

    if (FLAGS_lstm_shortcut && FLAGS_stack_size == 1) {
        std::cout << "shortcut flag ignored: Shortcut connections only take effect with stack size > 1" << std::endl;
    }

    int solver_type;
    if (FLAGS_solver == "adadelta") {
        solver_type = ADADELTA_TYPE;
    } else if (FLAGS_solver == "adam") {
        solver_type = ADAM_TYPE;
    } else if (FLAGS_solver == "sgd") {
        solver_type = SGD_TYPE;
    } else if (FLAGS_solver == "adagrad") {
        solver_type = ADAGRAD_TYPE;
    } else {
        utils::exit_with_message("Did not recognize this solver type.");
    }
    if (dataset.size() == 0) utils::exit_with_message("Dataset is empty");

    std::cout << "     Vocabulary size : " << vocab_size << std::endl
              << "      minibatch size : " << FLAGS_minibatch << std::endl
              << "   number of threads : " << FLAGS_j << std::endl
              << "        Dropout Prob : " << FLAGS_dropout << std::endl
              << " Max training epochs : " << FLAGS_epochs << std::endl
              << "   First Hidden Size : " << model.hidden_sizes[0] << std::endl
              << "           LSTM type : " << (FLAGS_lstm_memory_feeds_gates ? "Graves 2013" : "Zaremba 2014") << std::endl
              << "          Stack size : " << model.hidden_sizes.size() << std::endl
              << " # training examples : " << dataset.size() * FLAGS_minibatch - (FLAGS_minibatch - dataset[dataset.size() - 1].size()) << std::endl
              << "              Solver : " << FLAGS_solver << std::endl;

    vector<vector<Mat<REAL_t>>> thread_params;

    // what needs to be optimized:
    vector<model_t> thread_models;
    for (int i = 0; i < FLAGS_j; i++) {
        // create a copy for each training thread
        // (shared memory mode = Hogwild)
        thread_models.push_back(model.shallow_copy());

        auto thread_model_params = thread_models.back().parameters();
        // take a slice of all the parameters except for embedding.
        thread_params.emplace_back(
            thread_model_params.begin(),
            thread_model_params.end()
        );
    }
    auto params = model.parameters();

    // Rho value, eps value, and gradient clipping value:
    std::shared_ptr<Solver::AbstractSolver<REAL_t>> solver;
    switch (solver_type) {
        case ADADELTA_TYPE:
            solver = make_shared<Solver::AdaDelta<REAL_t>>(params, 0.95, 1e-9, 100.0, (REAL_t) FLAGS_reg);
            break;
        case ADAM_TYPE:
            solver = make_shared<Solver::Adam<REAL_t>>(params, 0.1, 0.001, 1e-9, 100.0, (REAL_t) FLAGS_reg);
            break;
        case SGD_TYPE:
            solver = make_shared<Solver::SGD<REAL_t>>(params, 100.0, (REAL_t) FLAGS_reg);
            dynamic_cast<Solver::SGD<REAL_t>*>(solver.get())->step_size = FLAGS_learning_rate;
            break;
        case ADAGRAD_TYPE:
            solver = make_shared<Solver::AdaGrad<REAL_t>>(params, 1e-9, 100.0, (REAL_t) FLAGS_reg);
            dynamic_cast<Solver::AdaGrad<REAL_t>*>(solver.get())->step_size = FLAGS_learning_rate;
            break;
        default:
            utils::exit_with_message("Did not recognize this solver type.");
    }

    REAL_t best_validation_score = 0.0;
    int epoch = 0;
    int best_epoch = 0;
    double patience = 0;
    string best_file = "";
    REAL_t best_score = 0.0;

    shared_ptr<Visualizer> visualizer;
    if (!FLAGS_visualizer.empty())
        visualizer = make_shared<Visualizer>(FLAGS_visualizer, true);

    // if no training should occur then use the validation set
    // to see how good the loaded model is.
    if (epochs == 0) {
        best_validation_score = paraphrase::pearson_correlation(
                validation_set,
                std::bind(&model_t::predict, &model, _1, _2),
                FLAGS_j);
        std::cout << "correlation = " << best_validation_score << std::endl;
    }

    while (patience < FLAGS_patience && epoch < epochs) {
        double memory_penalty = 0.0;

        if (memory_penalty_curve_type == 1) { // linear
            memory_penalty = std::min(
                FLAGS_memory_penalty,
                (REAL_t) (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch) / (double)(rampup_time))))
            );
        } else if (memory_penalty_curve_type == 2) { // square
            memory_penalty = std::min(
                FLAGS_memory_penalty,
                (REAL_t) (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch * epoch) / (double)(rampup_time * rampup_time))))
            );
        }

        atomic<int> examples_processed(0);

        int total_examples = 0;
        for (auto& minibatch: dataset) total_examples += minibatch.size();

        ReportProgress<double> journalist(
            utils::MS() << "Epoch " << ++epoch, // what to say first
            total_examples // how many steps to expect before being done with epoch
        );

        for (int batch_id = 0; batch_id < dataset.size(); ++batch_id) {
            pool->run([&word_vocab, &visualizer, &solver_type, &thread_params, &thread_models,
                       batch_id, &journalist, &solver, &dataset,
                       &best_validation_score, &examples_processed,
                       &memory_penalty]() {
                auto& thread_model     = thread_models[ThreadPool::get_thread_number()];
                auto& params           = thread_params[ThreadPool::get_thread_number()];
                auto& minibatch        = dataset[batch_id];
                // many forward steps here:
                REAL_t minibatch_error = 0.0;
                for (auto& example : minibatch) {
                    Mat<REAL_t> partial_error;
                    vector<Mat<REAL_t>> memory1, memory2;
                    std::tie(partial_error, memory1, memory2) =
                            thread_model.error(example);

                    if (memory_penalty > 0) {
                        auto memory = MatOps<REAL_t>::add(memory1) + MatOps<REAL_t>::add(memory2);
                        partial_error = partial_error + memory_penalty * memory;
                    }

                    partial_error = partial_error / minibatch.size();

                    minibatch_error += partial_error.w()(0,0);

                    partial_error.grad();
                    graph::backward(); // backpropagate
                    examples_processed++;
                }
                // One step of gradient descent
                solver->step(params);

                journalist.tick(examples_processed, minibatch_error);

                if (visualizer != nullptr) {
                    visualizer->throttled_feed(seconds(5), [&word_vocab, &visualizer, &minibatch, &thread_model]() {
                        // pick example
                        vector<uint> sentence1, sentence2;
                        double true_score;

                        std::tie(sentence1, sentence2, true_score) =
                                minibatch[utils::randint(0, minibatch.size()-1)];

                        double predicted_score;
                        vector<double> memory1, memory2;
                        std::tie(predicted_score, memory1, memory2) =
                                thread_model.predict_with_memories(sentence1, sentence2);

                        auto vs1  = make_shared<visualizable::Sentence<REAL_t>>(
                                word_vocab.decode(sentence1));
                        vs1->set_weights(memory1);

                        auto vs2  = make_shared<visualizable::Sentence<REAL_t>>(
                                word_vocab.decode(sentence2));
                        vs2->set_weights(memory2);

                        auto msg1 = make_shared<visualizable::Message>(MS() << "Predicted score: " << predicted_score);
                        auto msg2 = make_shared<visualizable::Message>(MS() << "True score: " << true_score);

                        auto grid = make_shared<visualizable::GridLayout>();

                        grid->add_in_column(0, vs1);
                        grid->add_in_column(0, vs2);
                        grid->add_in_column(1, msg1);
                        grid->add_in_column(1, msg2);

                        return grid->to_json();
                    });
                }
                // report minibatch completion to progress bar
            });
        }
        pool->wait_until_idle();
        journalist.done();
        auto new_validation = paraphrase::pearson_correlation(
            validation_set,
            std::bind(&model_t::predict, &model, _1, _2),
            FLAGS_j);
        if (solver_type == ADAGRAD_TYPE) {
            solver->reset_caches(params);
        }
        if (new_validation + 1e-6 < best_validation_score) {
            // lose patience:
            patience += 1;
        } else {
            // recover some patience:
            patience = std::max(patience - 1, 0.0);
            best_validation_score = new_validation;
        }
        if (best_validation_score != new_validation) {
            std::cout << "Epoch (" << epoch << ") Best validation score = " << best_validation_score << "% ("<< new_validation << "%), patience = " << patience << std::endl;
        } else {
            std::cout << "Epoch (" << epoch << ") Best validation score = " << best_validation_score << "%, patience = " << patience << std::endl;
            best_epoch = epoch;
        }
        if (new_validation > best_score) {
            best_score = new_validation;
            // save best:
            /*if (!FLAGS_save_location.empty()) {
                model.save(FLAGS_save_location);
                best_file = FLAGS_save_location;
            }*/
        }
    }

    /*if (!FLAGS_test.empty()) {
        auto test_set =  // load test set.

        if (!FLAGS_save_location.empty() && !best_file.empty()) {
            std::cout << "loading from best validation parameters \"" << best_file << "\"" << std::endl;
            auto params = model.parameters();
            utils::load_matrices(params, best_file);
        }

        // write test code and reporting here.

        auto test_score = paraphrase::pearson_correlation(validation_set, pred_fun, FLAGS_j);

        auto recall = SST::average_recall(test_set, pred_fun, FLAGS_j);

        std::cout << "Done training" << std::endl;
        std::cout << "Test recall "  << std::get<0>(recall) << "%, root => " << std::get<1>(recall)<< "%" << std::endl;
        if (!FLAGS_results_file.empty()) {
            ofstream fp;
            fp.open(FLAGS_results_file.c_str(), std::ios::out | std::ios::app);
            fp         << FLAGS_solver
               << "\t" << FLAGS_minibatch
               << "\t" << (FLAGS_fast_dropout ? "fast" : "std")
               << "\t" << FLAGS_dropout
               << "\t" << FLAGS_hidden
               << "\t" << std::get<0>(recall)
               << "\t" << std::get<1>(recall)
               << "\t" << best_epoch
               << "\t" << FLAGS_memory_penalty
               << "\t" << FLAGS_memory_penalty_curve;
            if ((FLAGS_solver == "adagrad" || FLAGS_solver == "sgd")) {
                fp << "\t" << FLAGS_learning_rate;
            } else {
                fp << "\t" << "N/A";
            }
            fp  << "\t" << FLAGS_reg << std::endl;
        }
    }


    // Write test accuracy here.
    */
}
