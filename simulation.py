import random
import pandas as pd
import numpy as np


class Airplane:
    def __init__(self, id, min_fuel, max_fuel, min_arrival_time, max_arrival_time):
        self.id = id
        self.fuel_consumption_rate = random.uniform(5, 20)
        self.expected_landing_time = random.uniform(min_arrival_time, max_arrival_time)
        self.fuel_level = random.uniform(min_fuel, max_fuel)

        self.emergency_fuel = (self.fuel_consumption_rate * 60)
        self.fuel_level = max(self.fuel_level, self.emergency_fuel)
        self.fuel_level_final = self.fuel_level - self.emergency_fuel
        self.remaining_flying_time = self.fuel_level_final / self.fuel_consumption_rate

        self.is_urgent = self.fuel_level_final < self.emergency_fuel or self.remaining_flying_time < 1


def generate_airplane_stream(num_airplanes, min_fuel, max_fuel, min_arrival_time, max_arrival_time):
    return [Airplane(i, min_fuel, max_fuel, min_arrival_time, max_arrival_time) for i in range(1, num_airplanes + 1)]

"""
Schedule landings for airplanes based on their urgency and expected landing time.

@param airplane_stream: A stream of airplanes to be scheduled for landing.
@type airplane_stream: list[Airplane]
@return: A DataFrame containing the scheduled landing information including airplane ID, actual landing time,
         urgency status, and the landing strip assigned.
@rtype: pandas.DataFrame
"""

def schedule_landings(airplane_stream):
    urgent_airplanes = sorted([ap for ap in airplane_stream if ap.is_urgent],
                              key=lambda x: x.remaining_flying_time)
    non_urgent_airplanes = sorted([ap for ap in airplane_stream if not ap.is_urgent],
                                  key=lambda x: x.expected_landing_time)
    
    sorted_airplanes = urgent_airplanes + non_urgent_airplanes
    
    landing_schedule = []
    landing_strip_availability = [0, 0, 0]
    landing_strip_index = 0

    for airplane in sorted_airplanes:
        chosen_strip = landing_strip_index % 3
        next_available_time_with_gap = landing_strip_availability[chosen_strip] + 3/60
        actual_landing_time = max(airplane.expected_landing_time, next_available_time_with_gap)
        
        if airplane.is_urgent and actual_landing_time > airplane.remaining_flying_time:
            actual_landing_time = airplane.remaining_flying_time

        landing_strip_availability[chosen_strip] = actual_landing_time + 3
        landing_schedule.append((airplane.id, actual_landing_time, airplane.is_urgent, chosen_strip + 1))

        landing_strip_index += 1

    return pd.DataFrame(landing_schedule, columns=["Airplane ID", "Actual Landing Time", "Urgent", "Landing Strip"])


#hill climbing, ga
def evaluate_landing_schedule(landing_schedule_df, airplane_stream):
    for index, row in landing_schedule_df.iterrows():
        airplane = next((ap for ap in airplane_stream if ap.id == row['Airplane ID']), None)
        if airplane:
            difference = abs(airplane.expected_landing_time - row['Actual Landing Time'])
            urgency_penalty = 100 if airplane.is_urgent else 0
            score = 1000 - difference - urgency_penalty
            landing_schedule_df.at[index, 'Score'] = score

    total_score = landing_schedule_df['Score'].sum()
    return total_score

#hill climbing
def get_successors(landing_schedule_df, airplane_stream): 
    if len(landing_schedule_df) <= 1:
        return [landing_schedule_df]
    successors = []
    for i in range(len(landing_schedule_df)): 
        for j in range(i + 1, len(landing_schedule_df)): 
            new_schedule_df = landing_schedule_df.copy()
            new_schedule_df.iloc[i], new_schedule_df.iloc[j] = new_schedule_df.iloc[j].copy(), new_schedule_df.iloc[i].copy()
            successors.append(new_schedule_df)
    return successors

def get_tabu_successors(landing_schedule_df, airplane_stream, tabu_list, current_score, num_neighbors=10):
    neighbors = []
    for _ in range(num_neighbors):
        # Choose two random airplanes
        i, j = random.sample(range(len(airplane_stream)), 2)
        # Create a copy of the current landing schedule
        neighbor_df = landing_schedule_df.copy()
        # Switch the two airplanes and their landing strip assignments
        neighbor_df.iloc[i], neighbor_df.iloc[j] = landing_schedule_df.iloc[j].copy(), landing_schedule_df.iloc[i].copy()
        # Convert the neighbor to a string to check if it's in the tabu list
        neighbor_str = neighbor_df.to_string()
        # Calculate the score of the neighbor
        neighbor_score = evaluate_landing_schedule(neighbor_df, airplane_stream)
        # If the neighbor is not in the tabu list, or if it's better than the current solution (aspiration criterion), add it to the list of neighbors
        if neighbor_str not in tabu_list or neighbor_score > current_score:
            neighbors.append(neighbor_df)
    return neighbors



"""
Optimize the landing schedule for airplanes using a genetic algorithm.

This function applies a genetic algorithm to optimize the landing schedule for airplanes. Genetic algorithms are population-based metaheuristic optimization techniques inspired by the principles of natural selection and genetics.

The algorithm begins by generating an initial population of landing schedules using the 'generate_initial_schedule' function. It then iterates through a predefined number of generations, during each of which it evaluates the fitness of each individual (schedule) in the population.

In each generation, the algorithm selects parents based on their fitness scores, performs crossover to create offspring, and applies mutation to introduce genetic diversity. Elitism is implemented by preserving the best individuals from each generation.

After the main genetic algorithm loop, the function selects the top individuals from the best generations, replacing the worst individuals in the current population. It returns the best landing schedule found along with its corresponding score.

@param airplane_stream: A list of airplanes to schedule for landing.
@type airplane_stream: list[Airplane]
@param population_size: The size of the population. Default is 50.
@type population_size: int, optional
@param generations: The number of generations to run the genetic algorithm. Default is 50.
@type generations: int, optional
@param crossover_rate: The probability of crossover between parents. Default is 0.8.
@type crossover_rate: float, optional
@param mutation_rate: The probability of mutation for each gene. Default is 0.1.
@type mutation_rate: float, optional
@return: A tuple containing the best optimized landing schedule (DataFrame) and its corresponding score.
@rtype: tuple(pandas.DataFrame, float)
"""

class GeneticAlgorithmScheduler:
    def __init__(self, airplane_stream, population_size=50, generations=50, crossover_rate=0.8, mutation_rate=0.1):
        self.airplane_stream = airplane_stream
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        return [self.generate_initial_schedule() for _ in range(self.population_size)]

    def generate_initial_schedule(self):
        shuffled_stream = random.sample(self.airplane_stream, len(self.airplane_stream))
        return schedule_landings(shuffled_stream)

    def calculate_fitness(self, schedule):
        return evaluate_landing_schedule(schedule, self.airplane_stream)

    def selection(self):
        fitness_scores = [self.calculate_fitness(schedule) for schedule in self.population]
        probabilities = 1 / (1 + np.array(fitness_scores))
        probabilities /= probabilities.sum()
        selected_indices = np.random.choice(range(len(self.population)), size=self.population_size, replace=False, p=probabilities)
        return [self.population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, parent1.shape[0] - 2)
            child1 = pd.concat([parent1.iloc[:crossover_point], parent2.iloc[crossover_point:]]).reset_index(drop=True)
            child2 = pd.concat([parent2.iloc[:crossover_point], parent1.iloc[crossover_point:]]).reset_index(drop=True)
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, schedule):
        for index in range(len(schedule)):
            if random.random() < self.mutation_rate:
                replacement_plane = random.choice(self.airplane_stream)
                replacement_index = schedule[schedule['Airplane ID'] == replacement_plane.id].index[0]
                schedule.at[index, 'Actual Landing Time'], schedule.at[replacement_index, 'Actual Landing Time'] = schedule.at[replacement_index, 'Actual Landing Time'], schedule.at[index, 'Actual Landing Time']
        return schedule

    def run(self):
        best_score = float('inf')
        best_schedule = None
        stale_generations = 0

        for generation in range(self.generations):
            new_population = []
            parents = self.selection()

            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            self.population = new_population[:self.population_size]

            current_best_score = min([self.calculate_fitness(schedule) for schedule in self.population])
            if current_best_score < best_score:
                best_score = current_best_score
                best_schedule = self.population[[self.calculate_fitness(schedule) for schedule in self.population].index(best_score)]
                stale_generations = 0  
            else:
                stale_generations += 1 

            print(f"Generation {generation}: Best Score - {best_score}")

            if stale_generations >= 5:
                print("No improvement over the last 5 generations. Stopping early.")
                break

        return best_schedule, best_score

