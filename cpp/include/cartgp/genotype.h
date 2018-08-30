#ifndef __CARTGP_GENOTYPE__
#define __CARTGP_GENOTYPE__

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace cartgp {

// Integer type that is being used for storing genes. Since those genes
// represent connections and functions, the same type is also being used
// for those matters.
using GeneInt = unsigned int;

/**
 * This class is a part of the Novelty Search algorithm. It's a container
 * responsible for storing fitnesess and behaviours of a population,
 * so they could easilly passed betweend various components.
 *
 * @tparam kBehavioralComponents
 */
template <std::size_t kBehavioralComponents>
struct PopulationBehaviours {
    std::vector<double> fitneses;
    std::vector<std::array<double, kBehavioralComponents>> behaviours;

    PopulationBehaviours() : fitneses{}, behaviours{} {}

    PopulationBehaviours(PopulationBehaviours &&o) noexcept {
        fitneses = std::move(o.fitneses);
        behaviours = std::move(o.behaviours);
    }

    PopulationBehaviours &operator=(PopulationBehaviours &&o) noexcept {
        fitneses = std::move(o.fitneses);
        behaviours = std::move(o.behaviours);
        return *this;
    }
};

/**
 * This class is a part of the Novelty Search algorithm.
 * It tracks the history of behaviours between generations, so we could detect
 * when behaviour becomes novel.
 *
 * The implementation is based on paper "Abandoning Objectives: Evolution
 * through the Search for Novelty Alone" by J. Lehman and K. O. Stanley.
 *
 * @tparam NumOfBehaviours how many behaviours we need to track
 */
template <std::size_t NumOfBehaviours>
class NoveltyEstimator {
public:

    // Behavioral components of a single individual
    using BehaviourComponents = std::array<double, NumOfBehaviours>;

    /**
     * Constructs new novelty estimator.
     * @param max_history_size How many types of behaviours we keep in
     *   the history. The larger the number, the better we can track if the
     *   behavious has already been seen before.
     * @param k_nearest How many nearest points are gonna be used to estimate
     *   the novelty of a particular individual.
     */
    NoveltyEstimator(const std::size_t max_history_size,
                     const std::size_t k_nearest)
        : max_history_size_{max_history_size}, k_nearest_{k_nearest} {}

    /**
     * Estimates how novel the behaviour of the given population actually is
     * based on the performances of the previous generations.
     *
     * @param all_siblings_behaviours Behavioral components for each individual
     *   in the population. Each individual has multiple (NumOfBehaviours)
     *   components, describing its behaviour.
     * @return Returns "novelty score" for every individual from the population.
     */
    std::vector<double>
    calculate_novelties(const std::vector<BehaviourComponents>
                        &all_siblings_behaviours) const {
        std::vector<double> result(all_siblings_behaviours.size());
        for (std::size_t s = 0; s < all_siblings_behaviours.size(); ++s) {
            auto &individ_behaviour = all_siblings_behaviours[s];
            std::vector<double> distances;
            distances.reserve(history_of_behaviours_.size() +
                              all_siblings_behaviours.size() - 1);

            // Calculating distances to individuals from the history
            for (auto &history_item: history_of_behaviours_) {
                double distance = 0;
                for (std::size_t b = 0; b < NumOfBehaviours; ++b) {
                    distance += std::pow(history_item[b] - individ_behaviour[b],
                                         2);
                }
                distances.push_back(std::sqrt(distance));
            }

            // Calculating distances to other siblings
            for (std::size_t i = 0; i < all_siblings_behaviours.size(); ++i) {
                if (i == s)
                    continue;
                auto &sibling_behaviour = all_siblings_behaviours[i];
                double distance = 0;
                for (std::size_t b = 0; b < NumOfBehaviours; ++b) {
                    distance += std::pow(
                        sibling_behaviour[b] - individ_behaviour[b], 2);
                }
                distances.push_back(std::sqrt(distance));
            }
            // Choosing k-nearest and calculating average distance to them
            std::sort(distances.begin(), distances.end());
            std::size_t max_distances_to_sum =
                std::min<std::size_t>(distances.size(), k_nearest_);
            double total_distance = 0;
            for (std::size_t i = 0; i < max_distances_to_sum; ++i) {
                total_distance += distances[i];
            }
            result[s] = (max_distances_to_sum > 0)
                        ? total_distance / max_distances_to_sum
                        : 0;
        }
        return result;
    }

    void add_to_history(const BehaviourComponents &behaviour,
                        const double novelty) {
        if (history_of_behaviours_.size() < max_history_size_) {
            history_of_behaviours_.push_back(behaviour);
            history_of_novelties_.push_back(novelty);
        } else {
            std::random_device random_device;
            std::mt19937 random_generator(random_device());
            std::uniform_int_distribution<std::size_t> dis(
                0, max_history_size_ - 1);
            std::size_t to_replace = dis(random_generator);
            history_of_behaviours_[to_replace] = behaviour;
            history_of_novelties_[to_replace] = novelty;
        }
    }

    double standard_dev_of_novelties() const {
        std::size_t data_size = history_of_novelties_.size();
        if (data_size == 0)
            return 0;
        auto avg = avg_novelty();
        double diffsum = 0;
        for (int i = 0; i < data_size; ++i) {
            diffsum += std::pow(history_of_novelties_[i] - avg, 2);
        }
        return std::sqrt(diffsum / data_size);
    }

    double avg_novelty() const {
        std::size_t data_size = history_of_novelties_.size();
        if (data_size == 0)
            return 0;
        double sum_novelty = 0;
        for (int i = 0; i < data_size; ++i) {
            sum_novelty += history_of_novelties_[i];
        }
        return sum_novelty / data_size;
    }

    double median_novelty() const {
        std::size_t data_size = history_of_novelties_.size();
        if (data_size == 0)
            return 0;
        std::vector<double> copy_of_history(history_of_novelties_);
        std::sort(copy_of_history.begin(), copy_of_history.end());
        std::size_t middle = data_size / 2;
        if (data_size % 2 == 0) {
            return (copy_of_history[middle] + copy_of_history[middle - 1]) / 2;
        } else {
            return copy_of_history[middle];
        }
    }

    std::size_t history_size() const { return history_of_novelties_.size(); }

private:
    const std::size_t max_history_size_;
    const std::size_t k_nearest_;
    std::vector<std::array<double, NumOfBehaviours>> history_of_behaviours_;
    std::vector<double> history_of_novelties_;
};

template <typename T>
struct Function {
  using FunctionRef = std::function<T(const std::vector<T> &arguments)>;

  const std::string name;
  const FunctionRef function;
};



class Genotype;

template <typename T>
using FitnessFunction = std::function<double(
    const Genotype &genotype, const std::vector<Function<T>> &functions)>;
template <typename T, std::size_t kBehavioralComponents>
using BehavioralFunction =
std::function<PopulationBehaviours<kBehavioralComponents>(
    const std::vector<Genotype> &population,
    const std::vector<Function<T>> &functions)>;

struct SolutionInfo {
    std::size_t steps;
    double fitness;
};

class Genotype {

public:
    /**
     * Basic constructor. Creates a new Genotype with each node
     * representing a function of some `arity`. Since the particular
     * set of functions can vary slightly, the expected number of them
     * must be specified through `num_function`.
     * `max_levels_back` specifies how far back (how many columns back)
     * each node can be linked to.
     * `num_inputs` and `num_outputs` define the size of the input and
     * the output vectors respectively.
     * `num_rows` and `num_columns` define the layout (the grid)
     * of the genotype. */
    Genotype(const GeneInt arity,
             const GeneInt num_functions,
             const GeneInt max_levels_back,
             const GeneInt num_inputs,
             const GeneInt num_outputs,
             const GeneInt num_rows,
             const GeneInt num_columns,
             const std::vector<GeneInt> &raw)
        : arity_{arity}, num_functions_{num_functions},
          max_levels_back_{max_levels_back}, num_inputs_{num_inputs},
          num_outputs_{num_outputs}, num_rows_{num_rows},
          num_columns_{num_columns}, genome_(raw),
          active_nodes_(find_active_nodes_()) {
        if (raw.size() != num_outputs + (arity + 1) * num_columns * num_rows) {
            throw std::invalid_argument("The genome has invalid size");
        }
    }

    /** Creates a new Genotype with a randomly initialized genome in the most
     * unconstrained configuration: when num_rows = 1,
     * and num_columns = max_levels_back = depth.
     * This is the best and most general configuration for the majority
     * of the tasks. */
    Genotype(const GeneInt arity, const GeneInt num_functions,
             const GeneInt num_inputs, const GeneInt num_outputs,
             const GeneInt depth)
        : Genotype(arity, num_functions, depth, num_inputs,
                   num_outputs, 1, depth,
                   random_genotype_(arity, num_functions, depth,
                                    num_inputs, num_outputs, 1,
                                    depth)) {}

    /** Creates a new Genotype with a randomly initialized genome */
    Genotype(const GeneInt arity,
             const GeneInt num_functions,
             const GeneInt max_levels_back,
             const GeneInt num_inputs,
             const GeneInt num_outputs,
             const GeneInt num_rows,
             const GeneInt num_columns)
        : Genotype(arity, num_functions, max_levels_back, num_inputs,
                   num_outputs, num_rows, num_columns,
                   random_genotype_(arity, num_functions, max_levels_back,
                                    num_inputs, num_outputs, num_rows,
                                    num_columns)) {}

    /** Creates a random genotype of the same size/configuration
     * as the given one */
    static Genotype random_like(const Genotype &o) {
        return Genotype(o.arity_, o.num_functions_, o.max_levels_back_,
                        o.num_inputs_, o.num_outputs_, o.num_rows_,
                        o.num_columns_);
    }

    Genotype(const Genotype &o)
        : arity_{o.arity_}, num_functions_{o.num_functions_},
          max_levels_back_{o.max_levels_back_}, num_inputs_{o.num_inputs_},
          num_outputs_{o.num_outputs_}, num_rows_{o.num_rows_},
          num_columns_{o.num_columns_}, genome_(o.genome_),
          active_nodes_(o.active_nodes_) {}

    Genotype(Genotype &&o) noexcept
        : arity_{o.arity_}, num_functions_{o.num_functions_},
          max_levels_back_{o.max_levels_back_}, num_inputs_{o.num_inputs_},
          num_outputs_{o.num_outputs_}, num_rows_{o.num_rows_},
          num_columns_{o.num_columns_}, genome_(std::move(o.genome_)),
          active_nodes_(std::move(o.active_nodes_)) {}

    Genotype &operator=(Genotype &&o) noexcept {
        arity_ = o.arity_;
        num_functions_ = o.num_functions_;
        max_levels_back_ = o.max_levels_back_;
        num_inputs_ = o.num_inputs_;
        num_outputs_ = o.num_outputs_;
        num_rows_ = o.num_rows_;
        num_columns_ = o.num_columns_;
        genome_ = std::move(o.genome_);
        active_nodes_ = std::move(o.active_nodes_);
        return *this;
    }

    std::vector<GeneInt> raw() const { return genome_; }
    GeneInt arity() const { return arity_; }
    GeneInt max_levels_back() const { return max_levels_back_; }
    GeneInt num_inputs() const { return num_inputs_; }
    GeneInt num_outputs() const { return num_outputs_; }
    GeneInt num_rows() const { return num_rows_; }
    GeneInt num_columns() const { return num_columns_; }
    const std::vector<GeneInt> &active_nodes() const {
        return active_nodes_;
    }

    template <typename T>
    std::vector<T> evaluate(const std::vector<Function<T>> &functions,
                            const std::vector<T> &input_data) const {
        if (input_data.size() != num_inputs_) {
            throw std::invalid_argument(
                "The number of arguments provided (" +
                    std::to_string(input_data.size()) +
                    ") "
                    "should be equal to the number of inputs (" +
                    std::to_string(num_inputs_) + ")");
        }
        validate_vector_of_functions_<T>(functions);

        std::vector<T> node_output(
            static_cast<std::size_t>(num_rows_ * num_columns_ + num_inputs_));
        std::copy(input_data.begin(), input_data.end(), node_output.begin());

        std::vector<T> args(arity_);
        for (auto active_node_index : active_nodes_) {
            const GeneInt *node_start = node_by_index_(active_node_index);
            for (GeneInt connection = 0; connection < arity_; ++connection) {
                args[connection] = node_output[node_start[connection + 1]];
            }
            GeneInt function_id = node_start[0];
            node_output[active_node_index] =
                functions[function_id].function(args);
        }

        std::vector<T> result(num_outputs_);
        const GeneInt *outputs = outputs_();
        for (GeneInt output = 0; output < num_outputs_; ++output) {
            GeneInt connected_to = outputs[output];
            result[output] = node_output[connected_to];
        }
        return result;
    }

    /**
     * Mutates genome until at least `max_active_mutations` active genes"
     * were mutated. Since the mutations are random, it is not guaranteed
     * that the mutated genes will actually get values different from
     * the old ones.
     *
     * @param max_active_mutations
     * @return
     */
    Genotype mutate(const std::size_t max_active_mutations = 1) {
        std::vector<GeneInt> new_genome(genome_);
        std::random_device random_device;
        std::mt19937 random_generator(random_device());
        const GeneInt genes_per_node = arity_ + 1;
        const GeneInt node_genes_num =
            genes_per_node * num_rows_ * num_columns_;
        std::uniform_int_distribution<GeneInt> gene_dist(
            0, node_genes_num + num_outputs_ - 1);
        std::uniform_int_distribution<GeneInt> func_dist(0, num_functions_ - 1);
        std::uniform_int_distribution<GeneInt> output_dist(
            0, num_inputs_ + num_rows_ * num_columns_ - 1);
        std::size_t hit_active_gene = 0;
        std::size_t total_mutations_made = 0;
        do {
            GeneInt gene_to_change = gene_dist(random_generator);
            GeneInt new_value;
            if (gene_to_change < node_genes_num) {
                GeneInt index_within_node = gene_to_change % genes_per_node;
                GeneInt index_among_nodes =
                    gene_to_change / genes_per_node + num_inputs_;
                if (index_within_node == 0) {
                    // It's a functional gene
                    new_value = func_dist(random_generator);
                } else {
                    // It's a connection gene
                    GeneInt column = gene_to_change / (num_rows_ * genes_per_node);
                    std::uniform_int_distribution<GeneInt> connection_dist(
                        ((column < max_levels_back_)
                         ? 0
                         : num_inputs_ + (column - max_levels_back_) * num_rows_),
                        num_inputs_ + column * num_rows_ - 1);
                    new_value = connection_dist(random_generator);
                }
                if (std::find(active_nodes_.begin(), active_nodes_.end(),
                              index_among_nodes) != active_nodes_.end()) {
                    ++hit_active_gene;
                }
            } else {
                // It's an output gene
                new_value = output_dist(random_generator);
                ++hit_active_gene;
            }
            new_genome[gene_to_change] = new_value;
            ++total_mutations_made;
        } while (hit_active_gene < max_active_mutations);
        return Genotype(arity_, num_functions_, max_levels_back_, num_inputs_,
                        num_outputs_, num_rows_, num_columns_, new_genome);
    }

    bool is_valid() const {
        for (GeneInt column = 0; column < num_columns_; ++column) {
            for (GeneInt row = 0; row < num_rows_; ++row) {
                const GeneInt *node_start =
                    &genome_[(arity_ + 1) * (column * num_rows_ + row)];
                GeneInt function_id = node_start[0];
                if (function_id >= num_functions_)
                    return false;
                for (GeneInt connection = 0;
                     connection < arity_; ++connection) {
                    GeneInt link = node_start[connection + 1];
                    if (column == 0) {
                        if (link >= num_inputs_)
                            return false;
                    } else {
                        if (link > num_inputs_ + column * num_rows_)
                            return false;
                        if (column > max_levels_back_) {
                            if (link > (num_inputs_ +
                                    (column - max_levels_back_) * num_rows_))
                                return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    /**
     * Converts phenotype (The actual expression represented by all
     * active genes) into a human-readable string.
     */
    template <typename T>
    std::string phenotype_to_string(const std::vector<Function<T>> &functions) {
        validate_vector_of_functions_<T>(functions);
        std::stringstream buffer;
        for (auto node_index : active_nodes_) {
            const GeneInt *node_start = node_by_index_(node_index);
            buffer << "(" << node_index << ":" << functions[node_start[0]].name;
            for (GeneInt connection = 0; connection < arity_; ++connection) {
                GeneInt link = node_start[connection + 1];
                if (link < num_inputs_) {
                    buffer << " In" << link;
                } else {
                    buffer << " " << link;
                }
            }
            buffer << ") ";
        }
        const GeneInt *outputs = outputs_();
        for (GeneInt i = 0; i < num_outputs_; ++i) {
            buffer << "(Out" << i << ": " << outputs[i] << ")";
            if (i < num_outputs_ - 1) {
                buffer << " ";
            }
        }
        return buffer.str();
    }

    /**
     * Returns a list of string where each item contains complete expression
     * for one of the outputs. The size of the list is thus equal to the
     * number of outputs.
     */
    template <typename T>
    std::vector<std::string> explain_outputs(
        const std::vector<Function<T>> &functions) {
        std::vector<std::string> result;
        validate_vector_of_functions_<T>(functions);
        const GeneInt *outputs = outputs_();
        for (GeneInt i = 0; i < num_outputs_; ++i) {
            result.push_back(unravel_connection_(functions, outputs[i]));
        }
        return result;
    }

    /**
     * Evolves new solutions until new generations stop producing
     * any improvements greater than the `stable_margin` margin for at least
     * `steps_to_stabilize` iterations.
     *
     * HINT: You can continue evolving the winner later if you're not satisfied
     * with the result.
     *
     * @tparam T
     * @param functions a set of actual functions that are going to be executed
     *   at each node
     * @param num_offsprings How many mutated offsprings should be created
     *   (the total size of the population will be num_offsprings + 1)
     * @param stable_margin difference in fitness between generations that is
     *  low enough to call the generation "stable" (close to the final solution)
     * @param steps_to_stabilize How many stable generations (with differences
     *   in fitness < stable_margin) we need to go through before we stop
     *   the evolution.
     * @param fitness_function Fitness function used for choosing the best
     *   individual within the population. The larger it is, the "better" the
     *   individual is.
     * @return
     */
    template <typename T>
    std::pair<Genotype, SolutionInfo>
    evolve(const std::vector<Function<T>> &functions,
           const std::size_t num_offsprings,
           const double stable_margin,
           const std::size_t steps_to_stabilize,
           FitnessFunction<T> fitness_function) {

        Genotype promoted(*this);
        double previous_fitness = fitness_function(promoted, functions);
        std::size_t steps_without_changes = 0;
        std::size_t total_steps = 0;
        bool first_run = true;

        do {
            Genotype best(promoted);
            double max_fitness = previous_fitness;

            for (std::size_t child_id = 0;
                    child_id < num_offsprings; ++child_id) {
                Genotype child =
                    first_run ? Genotype::random_like(promoted) : promoted.mutate();
                double child_fitness = fitness_function(child, functions);
                if (child_fitness >= max_fitness || std::isnan(max_fitness)) {
                    max_fitness = child_fitness;
                    best = std::move(child);
                }
            }
            double fitness_difference = std::abs(max_fitness - previous_fitness);
            if (fitness_difference < stable_margin) {
                ++steps_without_changes;
            } else {
                steps_without_changes = 0;
            }
            previous_fitness = max_fitness;
            promoted = std::move(best);
            first_run = false;
            total_steps += 1;
        } while (steps_without_changes < steps_to_stabilize);
        return std::pair<Genotype, SolutionInfo>(
            promoted,
            {total_steps, previous_fitness});
    }

    /**
     * Evolution using novelty search algorithm.
     *
     * The implementation is based on paper "Abandoning Objectives: Evolution
     * through the Search for Novelty Alone" by J. Lehman and K. O. Stanley.
     */
    template <typename T, std::size_t kBehavioralComponents>
    std::pair<Genotype, SolutionInfo>
    evolve_with_novelty(
        const std::vector<Function<T>> &functions,
        const std::size_t num_offsprings,
        const double stable_margin,
        const std::size_t steps_to_stabilize,
        NoveltyEstimator<kBehavioralComponents> &novelty_estimator,
        const double novelty_influence,
        const BehavioralFunction<T, kBehavioralComponents> behavioral_function)
    {

        Genotype promoted(*this);
        double previous_fitness = 0;
        std::size_t steps_without_changes = 0;
        std::size_t total_steps = 0;
        bool first_run = true;
        std::vector<Genotype> population;
        population.reserve(num_offsprings + 1);

        do {
            population.clear();
            population.push_back(promoted);
            // Genotype best(promoted);
            for (std::size_t child_id = 0;
                     child_id < num_offsprings; ++child_id) {
                population.push_back(std::move(
                    first_run ? Genotype::random_like(promoted) : promoted.mutate()));
            }
            PopulationBehaviours<kBehavioralComponents> behaviours =
                behavioral_function(population, functions);
            std::size_t best_in_population = 0;
            double max_fitness = 0;
            std::vector<double> novelties =
                novelty_estimator.calculate_novelties(behaviours.behaviours);
            double median_novelty = novelty_estimator.median_novelty();
            for (std::size_t individ = 0;
                    individ < population.size(); ++individ) {
                double combined_fitness =
                    ((1 - novelty_influence) * behaviours.fitneses[individ] +
                     novelty_influence * novelties[individ]);
                if (individ == 0) {
                    max_fitness = combined_fitness;
                    best_in_population = 0;
                } else if (combined_fitness >= max_fitness || std::isnan(max_fitness)) {
                    max_fitness = combined_fitness;
                    best_in_population = individ;
                }
            }

            if (novelty_estimator.history_size() < 10 ||
                novelties[best_in_population] > median_novelty) {
                novelty_estimator.add_to_history(
                    behaviours.behaviours[best_in_population],
                    novelties[best_in_population]);
            }

            if (!first_run) {
                double fitness_difference = std::abs(
                    max_fitness - previous_fitness);
                if (fitness_difference < stable_margin) {
                    ++steps_without_changes;
                } else {
                    steps_without_changes = 0;
                }
            }
            previous_fitness = max_fitness;
            promoted = std::move(population[best_in_population]);
            first_run = false;
            total_steps += 1;
        } while (steps_without_changes < steps_to_stabilize);
        return std::pair<Genotype, SolutionInfo>(
            promoted,
            {total_steps, previous_fitness});
    }

private:

    /* Assigns random (but valid) values to all connections,
     * functions and outputs */
    static std::vector<GeneInt>
    random_genotype_(const GeneInt arity,
                     const GeneInt num_functions,
                     const GeneInt max_levels_back,
                     const GeneInt num_inputs,
                     const GeneInt num_outputs,
                     const GeneInt num_rows,
                     const GeneInt num_columns) {
        std::random_device random_device;
        std::mt19937 random_generator(random_device());
        std::vector<GeneInt> genome(
            (arity + 1) * num_rows * num_columns + num_outputs, 0);

        std::uniform_int_distribution<GeneInt > func_dist(0, num_functions - 1);
        // First we fill every node with valid connections and function IDs
        for (GeneInt column = 0; column < num_columns; ++column) {
            std::uniform_int_distribution<GeneInt> connection_dist(
                ((column < max_levels_back)
                 ? 0
                 : num_inputs + (column - max_levels_back) * num_rows),
                num_inputs + column * num_rows - 1);
            for (GeneInt row = 0; row < num_rows; ++row) {
                GeneInt random_function_id = func_dist(random_generator);
                GeneInt *node_start =
                    &genome[(arity + 1) * (column * num_rows + row)];
                for (GeneInt input = 0; input < arity; ++input) {
                    GeneInt conn = connection_dist(random_generator);
                    node_start[0] = random_function_id;
                    node_start[input + 1] = conn;
                }
            }
        }
        // Now we initialize the outputs
        std::uniform_int_distribution<GeneInt> output_dist(
            0, num_inputs + num_rows * num_columns - 1);
        GeneInt *outputs = &genome[(arity + 1) * (num_columns * num_rows)];
        for (GeneInt output = 0; output < num_outputs; ++output) {
            outputs[output] = output_dist(random_generator);
        }
        return genome;
    }

    GeneInt arity_;
    GeneInt num_functions_;
    GeneInt max_levels_back_;
    GeneInt num_inputs_;
    GeneInt num_outputs_;
    GeneInt num_rows_;
    GeneInt num_columns_;
    std::vector<GeneInt> genome_;
    std::vector<GeneInt> active_nodes_;

    /** Returns a pointer to the first byte of a node addressed by its global
     * index. The range of this index includes inputs, though because
     * the're virtual they cannot be addressed. So the global_index must
     * always be >= num_inputs for the function to work properly */
    const GeneInt *node_by_index_(const GeneInt global_index) const {
        return &genome_[(arity_ + 1) * (global_index - num_inputs_)];
    }

    const GeneInt *outputs_() const {
        return &genome_[(arity_ + 1) * (num_columns_ * num_rows_)];
    }

    /** Returns the list of active nodes indices (the nodes which are part of the
     * phenotype). The first possible index is num_inputs
     * (just like in node_by_index_ function). */
    std::vector<GeneInt> find_active_nodes_() const {
        std::vector<bool> to_evaluate(
            num_inputs_ + num_outputs_ + num_rows_ * num_columns_, false);
        std::vector<GeneInt> nodes_in_use;
        nodes_in_use.reserve(num_rows_ * num_columns_);
        const GeneInt *outputs = outputs_();
        for (GeneInt i = 0; i < num_outputs_; ++i) {
            GeneInt connected_to = outputs[i];
            if (connected_to >= num_inputs_) {
                to_evaluate[connected_to] = true;
            }
        }
        for (GeneInt node_index = num_inputs_ + num_rows_ * num_columns_ - 1;
             node_index >= num_inputs_; --node_index) {
            if (to_evaluate[node_index]) {
                const GeneInt *connections = node_by_index_(node_index) + 1;
                for (GeneInt connection = 0; connection < arity_; ++connection) {
                    GeneInt connected_to = connections[connection];
                    if (connected_to >= num_inputs_) {
                        to_evaluate[connected_to] = true;
                    }
                }
                nodes_in_use.push_back(node_index);
            }
        }
        std::sort(nodes_in_use.begin(), nodes_in_use.end());
        return nodes_in_use;
    }

    template <typename T>
    void validate_vector_of_functions_(
        const std::vector<Function<T>> &functions) const {
        if (functions.size() != num_functions_) {
            throw std::invalid_argument(
                "The number of functions provided (" +
                std::to_string(functions.size()) +
                ") "
                "should be equal to the required number of functions "
                "for the genome (" +
                std::to_string(num_functions_) + ")");
        }
    }

    template <typename T>
    std::string unravel_connection_(const std::vector<Function<T>> &functions,
                                    const GeneInt node_index) {
        if (node_index < num_inputs_) {
            return "In" + std::to_string(node_index);
        } else {
            std::stringstream buffer;
            const GeneInt *node_start = node_by_index_(node_index);
            buffer << functions[node_start[0]].name << "(";
            for (GeneInt connection = 0; connection < arity_; ++connection) {
                GeneInt link = node_start[connection + 1];
                buffer << unravel_connection_(functions, link);
                if (connection < arity_ - 1) { buffer << ", "; }
            }
            buffer << ")";
            return buffer.str();
        }
    }
};

} // namespace cartgp

#endif // __CARTGP_GENOTYPE__
