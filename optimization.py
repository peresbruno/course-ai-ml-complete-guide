from typing import List
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose


products = [
  ('Refrigerador A', 0.751, 999.90),
  ('Celular', 0.0000899, 2911.12),
  ('TV 55', 0.400, 4346.99),
  ('TV 50', 0.290, 3999.90),
  ('TV 42', 0.200, 2999.00),
  ('Notebook A', 0.00350, 2499.90),
  ('Ventilador', 0.496, 199.90),
  ('Microondas A', 0.0424, 308.66),
  ('Microondas B', 0.0544, 429.90),
  ('Microondas C', 0.0319, 299.29),
  ('Refrigerador B', 0.635, 849.00),
  ('Refrigerador C', 0.870, 1199.89),
  ('Notebook B', 0.498, 1999.90),
  ('Notebook C', 0.527, 3999.00)
]

available_space_in_m3 = 3

def fitness_fn(solution: List[int]) -> float:
  total_m3 = 0
  total_value = 0
  for idx in range(len(products)):
    if solution[idx] == 1:
      product_m3 = products[idx][1]
      product_price = products[idx][2]

      total_m3 += product_m3
      if total_m3 > available_space_in_m3:
        return 0
      
      total_value += product_price
    
  return total_value

def print_solution(solution: List[int]) -> None:
  total_m3 = 0
  total_value = 0
  for idx in range(len(products)):
    if solution[idx] == 1:
      product_name = products[idx][0]
      product_m3 = products[idx][1]
      product_price = products[idx][2]

      total_m3 += product_m3
      total_value += product_price

      print(f'Product: {product_name}; Space: {product_m3}m³; Price: ${product_price}')
    
  print(f'Total value: $ {total_value}')
  print(f'Total space: {total_m3} m³')


fitness = mlrose.CustomFitness(fitness_fn)
problem = mlrose.DiscreteOpt(length=14, fitness_fn=fitness, maximize=True, max_val=2)

solution, cost = mlrose.hill_climb(problem)

print('HILL CLIMB')
print(f'cost: $ {cost}')
print_solution(solution)

print('===========')

print('SIMULATED ANEEALING')
solution, cost = mlrose.simulated_annealing(problem, max_attempts=1000, schedule=mlrose.decay.GeomDecay(init_temp=10000))
print(f'cost: $ {cost}')
print_solution(solution)

print('===========')

print('GENETIC ALG')

solution, cost = mlrose.genetic_alg(problem, pop_size=1000, mutation_prob=0.1, max_attempts=100)
print(f'cost: $ {cost}')
print_solution(solution)