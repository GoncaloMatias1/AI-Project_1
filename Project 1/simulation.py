import random
import pandas as pd
import numpy as np
import math
from queue import PriorityQueue


class Airplane:
    # Initialize the Airplane object with the given parameters
    def __init__(self, id, min_fuel, max_fuel, min_arrival_time, max_arrival_time):
        self.id = id # Unique identifier for the airplane
        
        # Randomly generate fuel consumption rate between 5 and 20 liters per minute
        self.fuel_consumption_rate = random.uniform(5, 20)
        
        # Generate a random expected landing time within the given range
        self.expected_landing_time = random.uniform(min_arrival_time, max_arrival_time)
        
        # Initialize the fuel level with a random value within the given range
        self.fuel_level = random.uniform(min_fuel, max_fuel)

        # Calculate the emergency fuel level (fuel rate * 60 minutes)
        self.emergency_fuel = (self.fuel_consumption_rate * 60)
        
        # Ensure the fuel level is at least as high as the emergency fuel level
        self.fuel_level = max(self.fuel_level, self.emergency_fuel)
        
        # Calculate the final fuel level and remaining flying time
        self.fuel_level_final = self.fuel_level - self.emergency_fuel
        self.remaining_flying_time = self.fuel_level_final / self.fuel_consumption_rate

        # Check if the airplane is in an urgent situation (low fuel or short remaining flying time)
        self.is_urgent = self.fuel_level_final < self.emergency_fuel or self.remaining_flying_time < 1


def generate_airplane_stream(num_airplanes, min_fuel, max_fuel, min_arrival_time, max_arrival_time):
    return [Airplane(i, min_fuel, max_fuel, min_arrival_time, max_arrival_time) for i in range(1, num_airplanes + 1)]

"""
Schedule landings for airplanes based on their urgency and expected landing time.

This function schedules the landing of airplanes based on their urgency status and expected landing time. It first sorts the airplanes into urgent and non-urgent categories. Within each category, airplanes are sorted by their remaining flying time or expected landing time.

The function then iterates over the sorted list of airplanes, assigning each airplane to a landing strip. It ensures that there is a minimum gap of 3 minutes between consecutive landings on the same strip. If an urgent airplane's remaining flying time is less than the next available time on the strip, it is scheduled to land immediately.

The function returns a DataFrame containing the scheduled landing information including airplane ID, actual landing time, urgency status, and the landing strip assigned.

@param airplane_stream: A stream of airplanes to be scheduled for landing.
@type airplane_stream: list[Airplane]
@return: A DataFrame containing the scheduled landing information including airplane ID, actual landing time,
         urgency status, and the landing strip assigned.
@rtype: pandas.DataFrame
"""

def schedule_landings(airplane_stream):
    # Sort the airplanes into urgent and non-urgent categories.
    urgent_airplanes = sorted([ap for ap in airplane_stream if ap.is_urgent],
                              key=lambda x: x.remaining_flying_time)
    non_urgent_airplanes = sorted([ap for ap in airplane_stream if not ap.is_urgent],
                                  key=lambda x: x.expected_landing_time)
    
    # Combine the sorted lists into one.
    sorted_airplanes = urgent_airplanes + non_urgent_airplanes
    
    # Initialize the landing schedule and the availability times for each landing strip.
    landing_schedule = []
    landing_strip_availability = [0, 0, 0]
    landing_strip_index = 0

    # Iterate over the sorted list of airplanes.
    for airplane in sorted_airplanes:
        # Choose a landing strip for the airplane.
        chosen_strip = landing_strip_index % 3
        # Calculate the next available time on the chosen strip with a 3-minute gap.
        next_available_time_with_gap = landing_strip_availability[chosen_strip] + 3/60
        # The actual landing time is the later of the airplane's expected landing time and the next available time on the strip.
        actual_landing_time = max(airplane.expected_landing_time, next_available_time_with_gap)
        
        # If the airplane is urgent and its remaining flying time is less than the next available time on the strip, it is scheduled to land immediately.
        if airplane.is_urgent and actual_landing_time > airplane.remaining_flying_time:
            actual_landing_time = airplane.remaining_flying_time

        # Update the next available time on the chosen strip.
        landing_strip_availability[chosen_strip] = actual_landing_time + 3
        # Add the landing information to the schedule.
        landing_schedule.append((airplane.id, actual_landing_time, airplane.is_urgent, chosen_strip + 1))

        # Move to the next landing strip.
        landing_strip_index += 1

    # Return the landing schedule as a DataFrame.
    return pd.DataFrame(landing_schedule, columns=["Airplane ID", "Actual Landing Time", "Urgent", "Landing Strip"])


def evaluate_landing_schedule(landing_schedule_df, airplane_stream):
    # Iterate through each row in the landing schedule DataFrame
    for index, row in landing_schedule_df.iterrows():
        # Find the airplane with the matching ID from the airplane stream
        airplane = next((ap for ap in airplane_stream if ap.id == row['Airplane ID']), None)
        if airplane:
            # Calculate the difference between the expected landing time and the actual landing time
            difference = abs(airplane.expected_landing_time - row['Actual Landing Time'])
            # Calculate the urgency penalty based on whether the airplane is urgent or not
            urgency_penalty = 100 if airplane.is_urgent else 0
            # Calculate the score by subtracting the difference and the urgency penalty from 1000
            score = 1000 - difference - urgency_penalty
            # Store the score in the landing schedule DataFrame
            landing_schedule_df.at[index, 'Score'] = score

    # Calculate the total score by summing up the scores in the landing schedule DataFrame
    total_score = landing_schedule_df['Score'].sum()
    
    # Return the total score
    return total_score


def get_successors(landing_schedule_df, airplane_stream): 
    if len(landing_schedule_df) <= 1:
        # If the landing schedule dataframe has 1 or no rows, then there is no possible swapping,
        # so we return the original dataframe as the only possible successor.
        return [landing_schedule_df]
    successors = []
    
    # We iterate over all pairs of airplanes in the landing schedule dataframe,
    # excluding pairs that consist of the same airplane.        
    for i in range(len(landing_schedule_df)): 
        for j in range(i + 1, len(landing_schedule_df)): 
            
            # We create a deep copy of the landing schedule dataframe,
            # so that we can modify it without affecting the original dataframe.
            new_schedule_df = landing_schedule_df.copy()
            
            # We swap the landing schedule of the two airplanes at index i and j,
            # effectively generating a new successor state.
            new_schedule_df.iloc[i], new_schedule_df.iloc[j] = new_schedule_df.iloc[j].copy(), new_schedule_df.iloc[i].copy()
            
            # We add the new successor state to the list of successors.
            successors.append(new_schedule_df)
    return successors

"""
Generate a list of successor states (neighbours or solutions) for the hill climbing and tabu search algorithms.

This function generates a list of successor states by randomly swapping two planes in the landing schedule. It 
creates a copy of the current landing schedule, swaps two planes, and recalculates the actual landing time and 
scores for each plane in the new schedule.

The function uses a deque to simulate the availability of landing strips. It removes the strip that was just used 
and adds the time when the strip will become available again. The deque is then sorted to ensure the earliest 
available strip is always first.

The function repeats this process for a specified number of successors and returns the list of successor states.

@param landing_schedule_df: The current landing schedule.
@type landing_schedule_df: pandas.DataFrame
@param airplane_stream: A list of airplanes to schedule for landing.
@type airplane_stream: list[Airplane]
@param num_successors: The number of successors to generate. Default is 15.
@type num_successors: int, optional
@return: A list of successor states.
@rtype: list[pandas.DataFrame]
"""

def get_Hill_Tabu_successors(landing_schedule_df, airplane_stream, num_successors=4):
    
    # Initialize an empty list to store successors
    successors = []
    
    # Get the number of planes from the landing schedule DataFrame
    num_planes = len(landing_schedule_df)
    
    # Create a dictionary for O(1) access to airplane objects using their IDs
    airplane_dict = {ap.id: ap for ap in airplane_stream}  # Create a dictionary for O(1) access
    
    # Generate the specified number of successors
    for _ in range(num_successors):
        # Randomly choose two planes to swap
        i, j = random.sample(range(num_planes), 2)
        # Create a copy of the landing schedule DataFrame
        new_schedule_df = landing_schedule_df.copy()
        # Swap the positions of the two chosen planes in the new schedule
        new_schedule_df.iloc[[i, j]] = new_schedule_df.iloc[[j, i]].values

        # Recalculate the Actual Landing Time and scores for the affected planes in the new schedule
        # Initialize a priority queue for strip availability times
        strip_availability_times = PriorityQueue()
        for _ in range(3):  # Initialize with 3 strips all available at time 0
            strip_availability_times.put(0)

        # Iterate over the two swapped planes' indexes in sorted order
        for index in sorted([i, j]):
            # Access the corresponding airplane object using the dictionary
            airplane = airplane_dict[new_schedule_df.at[index, 'Airplane ID']]
            # Calculate the current time based on the strip availability time and the airplane's expected landing time
            current_time = max(strip_availability_times.get(), airplane.expected_landing_time)
            # Update the Actual Landing Time in the new schedule
            new_schedule_df.at[index, 'Actual Landing Time'] = current_time
            # Calculate the difference between the expected and actual landing times
            difference = abs(airplane.expected_landing_time - current_time)
            # Apply the urgency penalty if the airplane is urgent
            urgency_penalty = 100 if airplane.is_urgent else 0
            # Calculate the score for the airplane
            score = 1000 - difference - urgency_penalty
            # Update the Score in the new schedule
            new_schedule_df.at[index, 'Score'] = score
            # Add the time when the strip will become available again to the priority queue
            strip_availability_times.put(current_time + 3)

        # Append the new schedule to the list of successors
        successors.append(new_schedule_df)

    # Return the list of successors
    return successors


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
    """
    This class implements a Genetic Algorithm Scheduler for scheduling airplane landings.
    """

    def __init__(self, airplane_stream, population_size=50, generations=50, crossover_rate=0.8, mutation_rate=0.1):
        """
        Initialize the GeneticAlgorithmScheduler with the given parameters.

        :param airplane_stream: A list of airplanes to be scheduled
        :param population_size: The number of individuals in the population (default: 50)
        :param generations: The number of generations to run the algorithm (default: 50)
        :param crossover_rate: The probability of crossover between two parents (default: 0.8)
        :param mutation_rate: The probability of mutation in a child (default: 0.1)
        """
        self.airplane_stream = airplane_stream
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        """
        Generate the initial population of schedules.

        :return: A list of initial schedules
        """
        return [self.generate_initial_schedule() for _ in range(self.population_size)]

    def generate_initial_schedule(self):
        """
        Generate an initial schedule by shuffling the airplane stream.

        :return: A schedule of landings
        """
        shuffled_stream = random.sample(self.airplane_stream, len(self.airplane_stream))
        return schedule_landings(shuffled_stream)

    def calculate_fitness(self, schedule):
        """
        Calculate the fitness score of a given schedule.

        :param schedule: A schedule of landings
        :return: The fitness score of the schedule
        """
        return evaluate_landing_schedule(schedule, self.airplane_stream)

    def selection(self):
        """
        Select parents based on their fitness scores.

        :return: A list of parents
        """
        fitness_scores = [self.calculate_fitness(schedule) for schedule in self.population]
        probabilities = 1 / (1 + np.array(fitness_scores))
        probabilities /= probabilities.sum()
        selected_indices = np.random.choice(range(len(self.population)), size=self.population_size, replace=False, p=probabilities)
        return [self.population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to generate two children.

        :param parent1: The first parent
        :param parent2: The second parent
        :return: Two children
        """
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, parent1.shape[0] - 2)
            child1 = pd.concat([parent1.iloc[:crossover_point], parent2.iloc[crossover_point:]]).reset_index(drop=True)
            child2 = pd.concat([parent2.iloc[:crossover_point], parent1.iloc[crossover_point:]]).reset_index(drop=True)
            return child1, child2
        else:
            return parent1, parent2

    def mutate(self, schedule):
        """
        Mutate a given schedule.

        :param schedule: A schedule of landings
        :return: The mutated schedule
        """
        for index in range(len(schedule)):
            if random.random() < self.mutation_rate:
                replacement_plane = random.choice(self.airplane_stream)
                replacement_index = schedule[schedule['Airplane ID'] == replacement_plane.id].index[0]
                schedule.at[index, 'Actual Landing Time'], schedule.at[replacement_index, 'Actual Landing Time'] = schedule.at[replacement_index, 'Actual Landing Time'], schedule.at[index, 'Actual Landing Time']
        return schedule

    def run(self):
        # Initialize variables for the best score, best schedule, and stale generations
        best_score = float('inf')
        best_schedule = None
        stale_generations = 0

        # Run the genetic algorithm for the specified number of generations
        for generation in range(self.generations):
            # Create a new population by performing crossover and mutation on the current population
            new_population = []
            parents = self.selection()

            while len(new_population) < self.population_size:
                # Select two parents from the current population
                parent1, parent2 = random.sample(parents, 2)

                # Perform crossover on the selected parents to create two new children
                child1, child2 = self.crossover(parent1, parent2)

                # Perform mutation on the first child
                child1 = self.mutate(child1)

                # Perform mutation on the second child
                child2 = self.mutate(child2)

                # Add the two new children to the new population
                new_population.extend([child1, child2])

            # Replace the current population with the new population
            self.population = new_population[:self.population_size]

            # Calculate the fitness score of the current population
            current_best_score = min([self.calculate_fitness(schedule) for schedule in self.population])

            # If the current best score is better than the previous best score, update the best score and schedule
            if current_best_score < best_score:
                best_score = current_best_score
                best_schedule = self.population[[self.calculate_fitness(schedule) for schedule in self.population].index(best_score)]
                stale_generations = 0
            # Otherwise, increment the count of stale generations
            else:
                stale_generations += 1

            # Print the current generation number and the best score
            print(f"Generation {generation}: Best Score - {best_score}")

            # If there have been 5 or more stale generations, stop the algorithm early
            if stale_generations >= 5:
                print("No improvement over the last 5 generations. Stopping early.")
                break

        # Return the best schedule and its fitness score
        return best_schedule, best_score

"""
Optimize the landing schedule for airplanes using hill climbing.

This function applies the hill climbing algorithm to improve the landing schedule for airplanes. Hill climbing is a local search algorithm that iteratively moves towards the best neighboring solution in the solution space.

The algorithm begins by determining if an airplane is urgent based on its fuel level and remaining flying time. It then generates an initial landing schedule using the 'schedule_landings' function and initializes the current score along with a list to store scores.

The function repeatedly explores neighboring landing schedules until no improvement is found. It selects the neighboring schedule with the highest score, assuming it as the next state. If the next score is equal to the current score, indicating no improvement, the search terminates.

The function returns the optimized landing schedule and an empty list of scores (since hill climbing does not keep track of past states).

@param airplane_stream: A list of airplanes to schedule for landing.
@type airplane_stream: list[Airplane]
@return: A tuple containing the optimized landing schedule (DataFrame) and an empty list of scores.
@rtype: tuple(pandas.DataFrame, list)
"""

def hill_climbing_schedule_landings(airplane_stream):
    # Mark urgent airplanes based on their fuel levels and expected landing times.
    for airplane in airplane_stream:
        airplane.is_urgent = (airplane.fuel_level_final < airplane.emergency_fuel or
                                airplane.remaining_flying_time < airplane.expected_landing_time)

    # Generate an initial landing schedule using the schedule_landings function.
    landing_schedule_df = schedule_landings(airplane_stream)

    # Initialize the current score and a list to store the scores of each iteration.
    current_score = evaluate_landing_schedule(landing_schedule_df, airplane_stream)
    scores = []

    # Repeat the following steps until no improvement is found.
    while True:
        # Get all neighboring landing schedules from the current schedule.
        neighbors = get_Hill_Tabu_successors(landing_schedule_df, airplane_stream)

        # Assume the next state is the same as the current state and track the highest score.
        next_state_df = landing_schedule_df
        next_score = current_score

        # Iterate over the neighboring landing schedules and find the one with the highest score.
        for neighbor_df in neighbors:
            score = evaluate_landing_schedule(neighbor_df, airplane_stream)
            if score > next_score:
                next_state_df = neighbor_df
                next_score = score

        # If the next score is equal to the current score, indicating no improvement, the search terminates.
        if next_score == current_score:
            break

        # Update the current state and score to the next state and score.
        landing_schedule_df = next_state_df
        current_score = next_score

    # Return the optimized landing schedule and an empty list of scores.
    return landing_schedule_df, scores


"""
Optimize the landing schedule for airplanes using simulated annealing.

This function applies the simulated annealing algorithm to improve the landing schedule for airplanes. Simulated annealing is a technique inspired by metallurgy annealing, gradually reducing temperature to explore the solution space while avoiding local optima.

The algorithm begins with an initial landing schedule generated by 'schedule_landings'. It iteratively adjusts the schedule to improve its score, considering both better and worse solutions based on a probability function and current temperature.

@param airplane_stream: A list of airplanes to schedule for landing.
@type airplane_stream: list[Airplane]
@return: A tuple containing the optimized landing schedule (DataFrame) and its score.
@rtype: tuple(pandas.DataFrame, float)
"""


def simulated_annealing_schedule_landings(airplane_stream):
    def calculate_score(schedule_df, airplane_stream):
        """
        Calculates the score for a given landing schedule based on the difference
        between expected and actual landing times and urgency of flights.
        
        Args:
        schedule_df (DataFrame): The schedule of landings.
        airplane_stream (iterable): A collection of airplanes with expected landing times and urgency.
        
        Returns:
        DataFrame: The updated schedule dataframe with scores.
        """
        for index, row in schedule_df.iterrows():
            airplane = next((ap for ap in airplane_stream if ap.id == row['Airplane ID']), None)
            if airplane:
                # Calculate time difference penalty
                time_diff = abs(airplane.expected_landing_time - row['Actual Landing Time'])
                # Add additional penalty for urgent landings
                urgency_penalty = 100 if airplane.is_urgent else 0
                # Score calculation: base score minus penalties
                score = 1000 - time_diff - urgency_penalty
                schedule_df.at[index, 'Score'] = score
        return schedule_df

    def get_schedule_neighbor(schedule_df):
        """
        Generates a neighboring schedule by swapping two landings.
        
        Args:
        schedule_df (DataFrame): The current landing schedule.
        
        Returns:
        DataFrame: A neighboring schedule dataframe.
        """
        neighbor_df = schedule_df.copy()
        # Randomly select two different rows to swap
        i, j = random.sample(range(len(neighbor_df)), 2)
        neighbor_df.iloc[i], neighbor_df.iloc[j] = neighbor_df.iloc[j].copy(), neighbor_df.iloc[i].copy()
        return neighbor_df

    # Initialize the landing schedule and calculate its score
    current_schedule = schedule_landings(airplane_stream)
    current_schedule = calculate_score(current_schedule, airplane_stream)
    current_score = current_schedule['Score'].sum()

    # Initialize the best schedule and score to the current ones
    best_schedule = current_schedule
    best_score = current_score

    T = 1.0  # Initial high temperature
    T_min = 0.001  # Minimum temperature to stop the algorithm
    alpha = 0.9  # Cooling rate

    # Main loop of simulated annealing
    while T > T_min:
        new_schedule = get_schedule_neighbor(current_schedule)
        new_schedule = calculate_score(new_schedule, airplane_stream)
        new_score = new_schedule['Score'].sum()
        
        # Accept new schedule based on the acceptance probability
        if new_score > current_score or math.exp((new_score - current_score) / T) > random.random():
            current_schedule = new_schedule
            current_score = new_score
            # Update best schedule and score if the new one is better
            if new_score > best_score:
                best_schedule = new_schedule
                best_score = new_score
        
        T *= alpha  # Cool down

    return best_schedule, best_score

"""
Optimize the landing schedule for airplanes using tabu search with early stopping and aspiration criteria.

This function applies the tabu search algorithm to improve the landing schedule for airplanes. It includes an early
stopping mechanism that halts the algorithm if there's no improvement after a certain number of iterations
(defined by the 'patience' parameter).

The algorithm begins by determining if an airplane is urgent based on its fuel level and remaining flying time. 
It then initializes variables, including the landing schedule, current score, a list of scores, and a tabu list.

The function iterates through the search process until it reaches the maximum number of iterations specified or until 
the 'patience' limit is reached. During each iteration, it generates neighboring solutions from the current solution 
and evaluates their scores. It selects the best solution among neighbors and considers it for the next iteration 
while also considering solutions in the tabu list.

An aspiration criteria is used to allow the search to return to previously visited solutions if they offer a 
significant improvement. This helps the algorithm to escape from local optima and explore new areas of the solution space. 
All neighbors are added to the tabu list, not just the ones that improve the score, to ensure a diverse search.

The search process continues until the maximum number of iterations or the 'patience' limit is reached. 
The function returns the best solution found and the list of scores recorded during the search process.

@param airplane_stream: A list of airplanes to schedule for landing.
@type airplane_stream: list[Airplane]
@param max_iterations: The maximum number of iterations for the tabu search algorithm. Default is 1000.
@type max_iterations: int, optional
@param max_tabu_size: The maximum size of the tabu list. Default is 10.
@type max_tabu_size: int, optional
@param patience: The number of iterations without improvement before the algorithm stops. Default is 5.
@type patience: int, optional
@return: A tuple containing the best landing schedule (DataFrame) and a list of scores recorded during the search process.
@rtype: tuple(pandas.DataFrame, list[float])
"""

def tabu_search_schedule_landings(airplane_stream, max_iterations=1000, max_tabu_size=10, patience=3):
    # Mark urgent airplanes based on their fuel levels and expected landing times.
    for airplane in airplane_stream:
        airplane.is_urgent = airplane.fuel_level_final < airplane.emergency_fuel or airplane.remaining_flying_time < airplane.expected_landing_time
    
    # Generate an initial landing schedule using the schedule_landings function.
    landing_schedule_df = schedule_landings(airplane_stream)
    current_score = evaluate_landing_schedule(landing_schedule_df, airplane_stream)
    scores = []
    tabu_set = set()
    it = 0
    no_improvement_count = 0
    best_score = float('-inf') 

    # Dictionary to store previously evaluated schedules
    evaluated_schedules = {}

    while it < max_iterations and no_improvement_count < patience:
        print(f"Iteration {it}")
        # Get all neighboring landing schedules from the current schedule.
        neighbors = get_Hill_Tabu_successors(landing_schedule_df, airplane_stream)
        next_state_df = landing_schedule_df
        scores.append(current_score)
        next_score = current_score

        best_solution_df = landing_schedule_df
        best_solution_score = evaluate_landing_schedule(landing_schedule_df, airplane_stream)

        # Iterate over the neighboring landing schedules and find the one with the highest score.
        for neighbor_df in neighbors:
            neighbor_hash = hash(neighbor_df.to_string())
            # If we've already evaluated this schedule, retrieve the score from the dictionary
            if neighbor_hash in evaluated_schedules:
                score = evaluated_schedules[neighbor_hash]
            else:
                score = evaluate_landing_schedule(neighbor_df, airplane_stream)
                evaluated_schedules[neighbor_hash] = score

            if score > best_solution_score:
                best_solution_df = neighbor_df
                best_solution_score = score
                # Add only improving solutions to the tabu list.
                if neighbor_hash not in tabu_set:
                    next_state_df = neighbor_df
                    next_score = score
                    tabu_set.add(neighbor_hash) 
                    if len(tabu_set) > max_tabu_size:
                        tabu_set.pop()

        # Aspiration criteria
        if hash(best_solution_df.to_string()) in tabu_set and best_solution_score > best_score:
            next_state_df = best_solution_df
            next_score = best_solution_score
            tabu_set.remove(hash(best_solution_df.to_string()))
            
        # Update the current state and score to the next state and score.
        landing_schedule_df = next_state_df
        current_score = next_score

        # If the best solution score is better than the best score so far, reset the no improvement count.
        # Otherwise, increment the no improvement count.
        if best_solution_score > best_score:
            best_score = best_solution_score
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Increment the iteration count.
        it += 1
    # Return the best solution found and the list of scores.
    return best_solution_df, scores