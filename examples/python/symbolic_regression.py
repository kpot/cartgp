import sympy
import math
import pycartgp


# First we create a dataset - a bunch of X and Y for a "unknown" function
def unknown_function(x):
    return 2 * x * x + 3 * x + 5


all_x = [0.1 * (i - 50) for i in range(100)]
all_y = [unknown_function(x) for x in all_x]


# Next we define a fitness function evaluating genotypes during the search
def fitness(_genotype: pycartgp.Genotype, _functions) -> float:
    error = 0
    for x, y in zip(all_x, all_y):
        _y = _genotype.evaluate(_functions, [x])[0]
        error += (y - _y)**2
    return -error / len(all_x)


# The list of functions we're going to build the solution from
available_functions = [
    ("plus", lambda args: args[0] + args[1]),
    ("minus", lambda args: args[0] - args[1]),
    ("mul", lambda args: args[0] * args[1]),
    ("div", lambda args: (args[0] / args[1]) if args[1] != 0 else 1),
]

# We create a random genotype, the ancestor of our final solution
genotype = pycartgp.Genotype(
    arity=2,
    num_functions=len(available_functions),
    num_inputs=1,  # we input x and constant 1.0 (to speed up the search)
    num_outputs=1, depth=100)
# and run the evolution running until the solution stabilizes
solution, info = genotype.evolve(
    available_functions, num_offsprings=4, stable_margin=1e-6,
    steps_to_stabilize=1000, fitness_function=fitness)


print('Steps taken:', info.steps)
print('Final fitness:', info.fitness)
print('Evolved (lengthy) expression:',
      solution.explain_outputs(available_functions)[0])
# We can evaluate the result using SymPy symbols and get a valid SymPy
# expression that is much more readable
sympy_expr = solution.evaluate(available_functions, [sympy.symbols('x')])
print('-' * 20)
print('Simplified expression:', sympy.simplify(sympy_expr))
