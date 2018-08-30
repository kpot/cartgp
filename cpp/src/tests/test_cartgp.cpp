#include <iostream>
#include <sstream>
#include <numeric>
#include <functional>
#include <cmath>

#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include "cartgp/genotype.h"

double target_function(double x) {
    return x * x + 2 * x + 1;
};


TEST_CASE("Check that random genome initialization works correctly") {
    using namespace cartgp;

    SECTION("Testing basic properties") {
        const unsigned int arity = 2,
              max_levels_back = 4,
              num_functions = 7,
              num_inputs = 3,
              num_outputs = 2,
              num_rows = 2,
              num_columns = 3;
        Genotype genome(
                arity, num_functions, max_levels_back,
                num_inputs, num_outputs, num_rows, num_columns);
        auto raw_form = genome.raw();
        REQUIRE(raw_form.size() == num_rows * num_columns * (arity + 1) + num_outputs);
    }

    SECTION("Testing random initialization") {
        const unsigned int arity = 2,
              max_levels_back = 4,
              num_functions = 7,
              num_inputs = 3,
              num_outputs = 2,
              num_rows = 2,
              num_columns = 3;
        for (int repetition = 0; repetition < 100; ++repetition) {
            Genotype genome(
                    arity, num_functions, max_levels_back,
                    num_inputs, num_outputs, num_rows, num_columns);
            auto raw_form = genome.raw();
            REQUIRE(raw_form.size() == num_rows * num_columns * (arity + 1) + num_outputs);
            for (GeneInt column = 0; column < num_columns; ++column) {
                for (unsigned int row = 0; row < num_rows; ++row) {
                    CAPTURE(column);
                    CAPTURE(row);
                    unsigned int *node_start = &raw_form[(arity + 1) * (column * num_rows + row)];
                    std::stringstream buffer;
                    for (int k = 0; k < arity + 1; ++k) {
                        buffer << node_start[k] << ", ";
                    }
                    INFO("Node raw representation: " << buffer.str());
                    unsigned int function_id = node_start[0];
                    REQUIRE(function_id >= 0);
                    REQUIRE(function_id <= num_functions - 1);
                    for (unsigned int connection=0; connection < arity; ++connection) {
                        CAPTURE(connection);
                        unsigned int link = node_start[connection + 1];
                        if (column == 0) {
                            REQUIRE(link < num_inputs);
                        } else {
                            REQUIRE(link <= num_inputs + column * num_rows);
                            if (column < max_levels_back) {
                                REQUIRE(link >= 0);
                            } else {
                                REQUIRE(link <= num_inputs + (column - max_levels_back) * num_rows);
                            }
                        }
                    }
                }
            }
            REQUIRE(genome.is_valid());
        }
    }

    SECTION("Test finding active nodes") {
        SECTION("Testing genome with no active nodes") {
            Genotype genome(
                2, 1, 2, 2, 2, 2, 2,
                {0, 0, 1,   0, 0, 1,  // col1 nodes are connected to the inputs
                 0, 2, 3,   0, 0, 1,  // col2 nodes connected to col1 and inputs
                 0, 1 // outputs connected directly to inputs (no active nodes)
                 });
            REQUIRE(genome.active_nodes().size() == 0);
        }

        SECTION("Testing genome with one active node") {
            Genotype genome(
                2, 1, 2, 2, 2, 2, 2,
                {0, 0, 1,   0, 0, 1,  // col0 nodes are connected to the inputs
                 0, 2, 3,   0, 0, 1,  // col1 nodes connected to col0 and inputs
                 2, 2 // outputs connected directly to node (r0, c0)
                 });
            std::vector<unsigned int> expected{2};
            REQUIRE(genome.active_nodes() == expected);
        }

        SECTION("Testing genome with all but one node active") {
            Genotype genome(
                2, 1, 2, 2, 2, 2, 2,
                {0, 0, 1,   0, 0, 1,  // col0 nodes are connected to the inputs
                 0, 2, 2,   0, 0, 1,  // col1 nodes connected to col0 and inputs
                 4, 5 // outputs connected directly to nodes (r0, c1) & (r1, c1)
                 });
            std::vector<unsigned int> expected{2, 4, 5};
            REQUIRE(genome.active_nodes() == expected);
        }

        SECTION("Testing genome with all nodes being active") {
            Genotype genome(
                2, 1, 3, 2, 2, 2, 3,
                {0, 0, 1,   0, 0, 1,  // col0 nodes are connected to the inputs
                 0, 2, 3,   0, 0, 1,  // col1 nodes connected to col0 and inputs
                 0, 4, 5,   0, 0, 1,  // col2 nodes connected to col2 and inputs
                 6, 7 // outputs connected directly to nodes (r0, c1) & (r1, c1)
                 });
            std::vector<unsigned int> expected{2, 3, 4, 5, 6, 7};
            REQUIRE(genome.active_nodes() == expected);
        }
    }

    SECTION("Testing evaluation") {
        // All nodes are involved
        Genotype genome1(
                2, 1, 3, 2, 2, 2, 3,
                {0, 0, 1,   0, 0, 1,  // col0 nodes are connected to the inputs
                 0, 2, 3,   0, 0, 1,  // col1 nodes connected to col0 and inputs
                 0, 4, 5,   0, 0, 1,  // col2 nodes connected to col2 and inputs
                 6, 7 // outputs connected directly to nodes (r0, c1) & (r1, c1)
                 });


        // One node is excluded
        Genotype genome2(
                2, 2, 3, 2, 2, 2, 3,
                {0, 0, 1,   0, 0, 1,  // col0 nodes are connected to the inputs
                 0, 2, 3,   1, 0, 1,  // col1 nodes connected to col0 and inputs
                 0, 4, 4,   0, 0, 1,  // col2 nodes connected to col2 and inputs
                 6, 7 // outputs connected directly to nodes (r0, c1) & (r1, c1)
                 });

        Function<double> function_42 = {
            "42",
            [](const std::vector<double> &arguments) -> double {
                return 42; 
            }
        };

        Function<double> function_sum = {
            "+",
            [](const std::vector<double> &arguments) -> double {
                double result = 0;
                for (auto arg: arguments) {
                    result += arg;
                }
                return result;
            }
        };

        REQUIRE(genome1.evaluate<double>({function_42}, {1, 2}) ==
                (std::vector<double>{42, 42}));
        REQUIRE(genome1.evaluate<double>({function_sum}, {1, 2}) ==
                (std::vector<double>{9, 3}));
        REQUIRE(genome1.evaluate<double>({function_sum}, {2, 3}) ==
                (std::vector<double>{15, 5}));
        REQUIRE(genome2.evaluate<double>({function_sum, function_42}, {2, 3}) ==
                (std::vector<double>{20, 5}));
    }

    SECTION("Testing mutation") {
        // All nodes are involved
        Genotype genome(
                2, 2, 3, 2, 2, 2, 3,
                {0, 0, 1,   0, 0, 1,  // col0 nodes are connected to the inputs
                 0, 2, 3,   0, 0, 1,  // col1 nodes connected to col0 and inputs
                 0, 4, 5,   0, 0, 1,  // col2 nodes connected to col2 and inputs
                 6, 7 // outputs connected directly to nodes (r0, c1) & (r1, c1)
                 });
        int total_differences = 0;
        const unsigned int genome_size = genome.raw().size();
        std::vector<int> touched(genome_size, 0);
        // we need a lot of repetitions to ensure we mutated everything at
        // least once
        for (int retry=0; retry < 1000; ++retry) {
            auto new_genome = genome.mutate();
            REQUIRE(new_genome.is_valid());
            // Exactly one active gene should be changed, though sometimes the
            // mutation changes nothing (not too many options)
            auto raw_genome = genome.raw();
            auto raw_new_genome = new_genome.raw();
            int differences = 0;
            for (auto i = 0; i < genome_size; ++i) {
                if (raw_genome[i] != raw_new_genome[i]) {
                    ++differences;
                    ++touched[i];
                }
            }
            REQUIRE(differences <= 1);
            total_differences += differences;
        }
        // we check that all genes were affected by the mutations
        std::stringstream buffer;
        for (int k = 0; k < genome_size; ++k) {
            buffer << touched[k] << ", ";
        }
        INFO("number of mutations per gene: " << buffer.str());
        for (auto i = 0; i < genome_size; ++i) {
            REQUIRE(touched[i] > 0);
        }
        
        REQUIRE(total_differences > 0);
    }

    SECTION("Test getting human-readable representation") {
        Genotype genome(
                2, 2, 3, 2, 2, 2, 3,
                {0, 0, 1,   0, 0, 1,  // col0 nodes are connected to the inputs
                 0, 2, 3,   0, 0, 1,  // col1 nodes connected to col0 and inputs
                 0, 4, 5,   0, 0, 1,  // col2 nodes connected to col2 and inputs
                 6, 7 // outputs connected directly to nodes (r0, c1) & (r1, c1)
                 });
        Function<double> function_sum = {
            "+",
            [](const std::vector<double> &arguments) -> double {
                double result = 0;
                for (auto arg: arguments) { result += arg; }
                return result;
            }
        };
        Function<double> function_mul = {
            "*",
            [](const std::vector<double> &arguments) -> double {
                double result = 0;
                for (auto arg: arguments) { result *= arg; }
                return result;
            }
        };
        
        auto phenotype = genome.phenotype_to_string<double>({function_sum, function_mul});
        REQUIRE(phenotype == "(2:+ In0 In1) (3:+ In0 In1) (4:+ 2 3) "
                "(5:+ In0 In1) (6:+ 4 5) (7:+ In0 In1) (Out0: 6) (Out1: 7)");
    }

    SECTION("Testing evolution") {
        std::vector<Function<double>> functions {
            {"*", [](const std::vector<double> &arguments) -> double {
                return std::accumulate(arguments.begin(), arguments.end(), 1.0, std::multiplies<double>());
            }},

            {"+", [](const std::vector<double> &arguments) -> double {
                return std::accumulate(arguments.begin(), arguments.end(), 0.0, std::plus<double>());
            }},

            {"-", [](const std::vector<double> &arguments) -> double {
                return std::accumulate(arguments.begin() + 1, arguments.end(), arguments[0], std::minus<double>());
            }},

            {"/", [](const std::vector<double> &arguments) -> double {
                return std::accumulate(arguments.begin() + 1, arguments.end(), arguments[0], std::divides<double>());
            }},

        };

        std::vector<double> args({3.5, 2});
        REQUIRE(functions[0].function(args) == 7);
        REQUIRE(functions[1].function(args) == 5.5);
        REQUIRE(functions[2].function(args) == 1.5);
        REQUIRE(functions[3].function(args) == 1.75);

        const unsigned int arity = 2,
              num_columns = 100,
              num_rows = 1,
              max_levels_back = num_columns,
              num_inputs = 2,
              num_outputs = 1;


        Genotype genome(arity, functions.size(), max_levels_back, num_inputs, num_outputs, num_rows, num_columns);
        FitnessFunction<double> fitness = [](
                const Genotype &genotype,
                const std::vector<Function<double>> &functions) -> double {
            double error = 0;
            for (double x = 0; x < 10; x += 0.1) {
                double eval_result = genotype.evaluate<double>(functions, {x, 1})[0];
                error += std::abs(target_function(x) - eval_result);
            }
            return -error;
        };
        Genotype perfect_genome(  // relies on getting X and 1 from the input
                2, functions.size(), 5, 2, 1, 1, 5,
                {0, 0, 0, // #2 getting x * x
                 1, 1, 1, // #3 getting 2 
                 0, 0, 3, // #4 getting 2 * x
                 1, 2, 4, // #5 getting x * x + 2 * x
                 1, 1, 5, // #6 getting x * x + 2 * x + 1 together
                 6 // solution
                 });
        // First we check that our perfect genome corectly represents our
        // function
        REQUIRE(perfect_genome.evaluate<double>(functions, {0, 1})[0] == target_function(0));
        REQUIRE(perfect_genome.evaluate<double>(functions, {1, 1})[0] == target_function(1));
        REQUIRE(perfect_genome.evaluate<double>(functions, {2, 1})[0] == target_function(2));
        REQUIRE(perfect_genome.evaluate<double>(functions, {3, 1})[0] == target_function(3));
        REQUIRE(perfect_genome.evaluate<double>(functions, {3.1415, 1})[0] == target_function(3.1415));

        REQUIRE(fitness(perfect_genome, functions) == 0);

        auto winner = perfect_genome.evolve<double>(functions, 4, 1e-6, 10, fitness).first;
        REQUIRE(winner.raw() == perfect_genome.raw());

        auto original_fitness = fitness(genome, functions);
        winner = genome.evolve<double>(functions, 4, 1e-6, 100, fitness).first;
        if (std::isnan(original_fitness)) {
            REQUIRE(std::isnan(fitness(winner, functions)) == false);
        } else {
            REQUIRE(fitness(winner, functions) > original_fitness);
        }

    }
}

TEST_CASE("Checking NoveltyEstimator") {
    using namespace cartgp;

    SECTION("Calculation of the distances") {
        const unsigned int history_size = 1, k_nearest = 4;
        NoveltyEstimator<2> estimator(history_size, k_nearest);
        std::vector<double> distances = estimator.calculate_novelties(
                {{0, 1}, {2, 1}, {4, 1}, {10, 1}});
        REQUIRE(estimator.standard_dev_of_novelties() == 0.0);
        // First we check distances calculated only between siblings
        REQUIRE(distances == (std::vector<double> {
                    (2 + 4 + 10)/3.0,
                    (2 + 2 + 8) / 3.0,
                    (4 + 2 + 6) / 3.0,
                    (10 + 8 + 6) / 3.0}));
        // Now between siblings and one item from the history
        estimator.add_to_history({-1, 1}, 2);
        distances = estimator.calculate_novelties({{0, 1}, {2, 1}, {4, 1}, {10, 1}});
        REQUIRE(distances == (std::vector<double> {
                    (2 + 4 + 10 + 1)/4.0,
                    (2 + 2 + 8 + 3) /4.0,
                    (4 + 2 + 6 + 5) /4.0,
                    (10 + 8 + 6 + 11) /4.0}));
        // Now we overwhelm the history. New record should replace an old one.
        estimator.add_to_history({-2, 1}, 1);
        distances = estimator.calculate_novelties({{0, 1}, {2, 1}, {4, 1}, {10, 1}});
        REQUIRE(distances == (std::vector<double> {
                    (2 + 4 + 10 + 2)/4.0,
                    (2 + 2 + 8 + 4) /4.0,
                    (4 + 2 + 6 + 6) /4.0,
                    (10 + 8 + 6 + 12) /4.0}));
        REQUIRE(estimator.standard_dev_of_novelties() == 0.0);
    }

    SECTION("Limiting results to only k neighbours") {
        const unsigned int history_size = 1, k_nearest =2;
        // Now we check estimator with very short max_k_neares to check how
        // correctly we choose closest neigbours
        NoveltyEstimator<2> estimator2(history_size, k_nearest);
        auto distances = estimator2.calculate_novelties(
                {{0, 1}, {2, 1}, {4, 1}, {10, 1}});
        // First we check distances calculated only between siblings
        REQUIRE(distances == (std::vector<double> {
                    (2 + 4)/2.0,
                    (2 + 2) / 2.0,
                    (4 + 2) / 2.0,
                    (8 + 6) / 2.0}));
    }

    SECTION("Standard deviation of novelties") {
        const unsigned int history_size = 3, k_nearest = 2;
        NoveltyEstimator<2> estimator(history_size, k_nearest);
        REQUIRE(estimator.standard_dev_of_novelties() == 0);
        estimator.add_to_history({-2, 1}, 1);
        REQUIRE(estimator.standard_dev_of_novelties() == 0);
        estimator.add_to_history({-2, 1}, 2);
        estimator.add_to_history({-2, 1}, 3);
        REQUIRE(estimator.standard_dev_of_novelties() == std::sqrt(2.0 / 3.0));
        REQUIRE(estimator.avg_novelty() == 2);
        REQUIRE(estimator.median_novelty() == 2);
    }
}
