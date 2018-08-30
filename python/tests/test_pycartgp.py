import unittest
import math

import pycartgp


class TestPyCartGP(unittest.TestCase):
    available_functions = [
        ("plus", lambda args: args[0] + args[1]),
        ("minus", lambda args: args[0] - args[1]),
        ("mul", lambda args: args[0] * args[1]),
    ]

    def setUp(self):
        self.func_set = pycartgp.genotype_functions(self.available_functions)

    def test_customized_genotype_creation(self):
        genotype = pycartgp.Genotype(
            arity=2, num_functions=len(self.func_set), max_levels_back=3,
            num_inputs=2, num_outputs=8, num_rows=5, num_columns=7)
        self.assertEqual(genotype.num_rows, 5)
        self.assertEqual(genotype.num_columns, 7)
        self.assertEqual(genotype.num_inputs, 2)
        self.assertEqual(genotype.num_outputs, 8)
        self.assertEqual(genotype.max_levels_back, 3)

    def test_simplified_genotype_creation(self):
        genotype = pycartgp.Genotype(
            arity=2,
            num_functions=len(self.func_set),
            num_inputs=2, num_outputs=8, depth=30)
        self.assertEqual(genotype.num_rows, 1)
        self.assertEqual(genotype.num_columns, 30)
        self.assertEqual(genotype.num_outputs, 8)
        self.assertEqual(genotype.num_inputs, 2)
        self.assertEqual(genotype.max_levels_back, 30)

    def test_explaining_outputs(self):
        genotype = pycartgp.Genotype(
            arity=2,
            num_functions=len(self.func_set),
            num_inputs=2, num_outputs=8, depth=30)
        expressions = genotype.explain_outputs(self.func_set)
        self.assertIsInstance(expressions, list)
        self.assertEqual(len(expressions), 8)
        for e in expressions:
            self.assertIsInstance(e, str)
            self.assertGreater(len(e), 0)

    def test_explaining_phenotype(self):
        genotype = pycartgp.Genotype(
            arity=2,
            num_functions=len(self.func_set),
            num_inputs=2, num_outputs=8, depth=30)
        phenotype_string = genotype.phenotype_to_string(self.func_set)
        self.assertIsInstance(phenotype_string, str)
        self.assertGreater(len(phenotype_string), 0)

    def test_evolution(self):
        def formula(x):
            return 2 * x * x + 3 * x + 5

        all_x = [0.1 * (i - 50) for i in range(100)]
        all_y = list(map(formula, all_x))

        def fitness(_genotype: pycartgp.Genotype, _functions) -> float:
            error = 0
            for x, y in zip(all_x, all_y):
                _y = _genotype.evaluate(_functions, [1.0, x])
                error += (y - _y[0])**2
            return -math.sqrt(error / len(all_x))

        genotype = pycartgp.Genotype(
            arity=2,
            num_functions=len(self.func_set),
            num_inputs=2, num_outputs=1, depth=100)
        solution, info = genotype.evolve(
            self.func_set, num_offsprings=4, stable_margin=0.001,
            steps_to_stabilize=500, fitness_function=fitness)
        self.assertAlmostEqual(
            info.fitness, 0,
            msg=("CGP couldn't find a solution. Sometimes this happens, "
                 "although normally this should be a rare case."))


if __name__ == '__main__':
    unittest.main()

