import matplotlib.pyplot as plt
import random
import time

from enum import Enum


# Define each possible move
class Move(Enum):
    COOPERATE = 1
    DEFECT = 2
    OPP_PREV = 3
    NOT_OPP_PREV = 4


# Number of rounds
rounds = 30
# Chance that noise impacts a move
noise_rate = 0.05


# Represent each strategy
class Strategy:
    def __init__(self, moves: list[Move]):
        self.moves = moves
        self.fitness: int = self.calculate_fitness()

    def calculate_fitness(self) -> int:
        fitness = 0

        # Test against each fixed strategy
        for fixed_strat in fixed_strategies:
            # Store each strat's previous move
            s1_prev = None
            s2_prev = None

            for i in range(rounds):
                # s1 = input, s2 = fixed
                s1_move = decode_move(self.moves[i], s2_prev)
                s2_move = fixed_strat(s1_prev)

                if random.random() < noise_rate:
                    s1_move = not s1_move
                if random.random() < noise_rate:
                    s2_move = not s2_move

                # If s1 defects and s2 cooperates
                if not s1_move and s2_move:
                    fitness += 5
                # If both cooperate
                elif s1_move and s2_move:
                    fitness += 3
                # If both defect
                elif not s1_move and not s2_move:
                    fitness += 1

                s1_prev = s1_move
                s2_prev = s2_move
        
        return fitness


def generate_strats(pop_size: int):
    strats: list[Strategy] = []
    for _ in range(pop_size):
        moves: list[Move] = []

        # Cooperate or defect on the first move
        moves.append(Move.COOPERATE if random.random() < 0.5 else Move.DEFECT)
        
        # Randomly choose remaining moves
        for _ in range(rounds-1):
            moves.append(random.choice(list(Move)))

        strats.append(Strategy(moves))

    return strats


# Fixed strats

def always_cooperate(_) -> bool:
    return True

def always_defect(_) -> bool:
    return False

def tit_for_tat(opp_prev: bool) -> bool:
    return True if opp_prev is None else opp_prev

# Choose which strategies to compete with
fixed_strategies = [always_cooperate, always_defect, tit_for_tat]


# Convert strategy's Move to a bool (coop/defect)
def decode_move(s1_move: Move, s2_prev: bool) -> bool:
    if s1_move == Move.COOPERATE:
        return True
    if s1_move == Move.DEFECT:
        return False
    if s1_move == Move.OPP_PREV:
        return s2_prev
    return not s2_prev


# Perform tournament selection on a given list of strats (individuals)
def perform_selection(strats: list[Strategy]):
    # Select 7 at random and return the top 2 fittest strats
    selection = random.sample(strats, 7)
    best_two = sorted(selection, key=lambda strat: strat.fitness)[2:]
    return best_two[0], best_two[1]


# Perform ordered crossover on two given strats
def ordered_crossover(parent_1: Strategy, parent_2: Strategy):
    size = len(parent_1.moves)
    start, end = sorted(random.sample(range(size), 2))

    # Copy random section from parent 1 to the offspring
    child_strat = [None] * size
    child_strat[start:end] = parent_1.moves[start:end]

    pos = end        
    # Fill in remaining moves sourced from parent 2
    for i in range(size):
        pos = end % size
        if child_strat[pos] is None:
            child_strat[pos] = parent_2.moves[pos]
            end += 1

    return Strategy(child_strat)


# Perform uniform crossover on two given strats
def uniform_crossover(parent_1: Strategy, parent_2: Strategy):
    size = len(parent_1.moves)
    child_strat = [None] * size

    for i in range(size):
        child_strat[i] = parent_1.moves[i] if random.random() < 0.5 else parent_2.moves[i]

    return Strategy(child_strat)


# Perform crossover on two given strats, subject to a given rate
# If crossover is not performed, return parent 1
def perform_crossover(strat_1: Strategy, strat_2: Strategy, crossover_rate: float):
    if random.random() < crossover_rate:
        # 50/50 chance of which crossover algorithm is selected
        if random.random() < 0.5:
            return ordered_crossover(strat_1, strat_2)
        return uniform_crossover(strat_1, strat_2)
    return strat_1


# Perform swap mutation on a given strat
def swap_mutation(strat: Strategy):
    i, j = random.sample(range(len(strat.moves)), 2)
    strat.moves[i], strat.moves[j] = strat.moves[j], strat.moves[i]


# Perform scramble mutation on a given strat
def scramble_mutation(strat: Strategy):
    i, j = sorted(random.sample(range(len(strat.moves)), 2))
    sublist = strat.moves[i:j]
    random.shuffle(sublist)
    strat.moves[i:j] = sublist


# Perform mutation on a given strat, subject to a given rate
def perform_mutation(strat: Strategy, mutation_rate: float):
    if random.random() < mutation_rate:
        # 50/50 chance of mutation algorithm used
        if random.random() < 0.5:
            swap_mutation(strat)
        else:
            scramble_mutation(strat)
        # Set a valid first move
        strat.moves[0] = Move.COOPERATE if random.random() < 0.5 else Move.DEFECT


# Run the genetic algorithm given several options and the list of moves
def run_genetic_algorithm(pop_size: int, max_iter: int, elitism_rate: int,
                          crossover_rate: float, mutation_rate: float):
    strats: list[Strategy] = []
    fitness_history: list[int] = []
    
    # Create initial random population
    strats: list[Strategy] = generate_strats(pop_size)

    # Perform iterations until limit is reached
    for _ in range(max_iter):
        new_strats: list[Strategy] = []
        
        # Elitism: let top n fittest progress
        new_strats.extend(sorted(strats, key=lambda strat: strat.fitness, reverse=True)[:elitism_rate])

        # Perform selection, crossover, and mutation (subject to respective rates)
        for _ in range(pop_size-elitism_rate):
            strat_1, strat_2 = perform_selection(strats)
            new_strat = perform_crossover(strat_1, strat_2, crossover_rate)
            perform_mutation(new_strat, mutation_rate)
            new_strat.fitness = new_strat.calculate_fitness()
            new_strats.append(new_strat)

        # Log the best strat (fittest individual) in the population
        fitness_history.append(max(new_strats, key=lambda strat: strat.fitness).fitness)
        # Overwrite old population with new population
        strats = new_strats

    # Find the best/fittest strat and return with fitness history
    best_strat = max(strats, key=lambda strat: strat.fitness)
    return best_strat, fitness_history


def main():
    # Parameters
    population_size = 50        # Size of the population
    maximum_iterations = 1000   # Maximum number of iterations performed
    elitism_rate = 10           # How many top strats should progress to the next round
    crossover_rate = 0.8        # Rate at which crossover is performed
    mutation_rate = 0.2         # Rate at which mutation is performed

    # Run the genetic algorithm with the above parameters
    start_time = time.time()
    best_strat, fitness_history = run_genetic_algorithm(population_size, maximum_iterations,
                                                       elitism_rate, crossover_rate, mutation_rate)
    end_time = time.time()

    print(f'Genetic algorithm completed in {end_time-start_time:.3f} seconds.')
    print(f'The best strategy found has a fitness of {best_strat.fitness}:')
    print([move.name for move in best_strat.moves])

    # Create fitness history graph
    plt.plot(fitness_history)
    plt.xlabel("Generations")
    plt.ylabel("Best fitness (score)")
    plt.title("Fitness over generations")
    plt.show()


if __name__ == "__main__":
    main()
