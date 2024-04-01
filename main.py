import random
import math
import pandas as pd
from simulation import (generate_airplane_stream, schedule_landings, evaluate_landing_schedule, get_successors, generate_initial_schedule, select_parents, crossover, mutate)


def get_input(prompt, type_=None, min_=None, max_=None, header=None):
    while True:
        if header:
            print("\n" + "-" * 72)
            print("|" + header.center(70) + "|")
            print("-" * 72)
        try:
            value = input("| " + prompt.ljust(68) + "| ")
            print("-" * 72)
            if type_ is not None:
                value = type_(value)
            if (min_ is not None and value < min_) or (max_ is not None and value > max_):
                raise ValueError(f"Value should be between {min_} and {max_}.")
            return value
        except ValueError as e:
            print("|" + f"Invalid input: {e}".center(70) + "|")
            print("-" * 72)



def select_algorithm():
    print("\n" + "-" * 72)
    print("|" + "Airport Landing Scheduler".center(70) + "|")
    print("|" + "Optimization Algorithm Selection".center(70) + "|")
    print("-" * 72)
    print("|" + "1 - Hill Climbing".ljust(69) + "|")
    print("|" + "2 - Simulated Annealing".ljust(69) + "|")
    print("|" + "3 - Tabu Search".ljust(69) + "|")
    print("|" + "4 - Genetic Algorithm".ljust(69) + "|")
    print("-" * 72)
    choice = get_input("Enter your choice (number): ", type_=int, min_=1, max_=4)  # Update max_=4
    return choice



"""
|----------------------------------------------------------------------------------------------------------|
    This function schedules landings for an airplane stream using the Hill Climbing optimization algorithm.
    
    Parameters:
        airplane_stream (list): List of Airplane objects representing the airplane stream.

    Returns:
        - DataFrame containing the optimized landing schedule.

|----------------------------------------------------------------------------------------------------------|
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
        #Get all neighboring landing schedules from the current schedule.
        neighbors = get_successors(landing_schedule_df, airplane_stream)

        # Assume the next state is the same as the current state and track the lowest score.
        next_state_df = landing_schedule_df
        next_score = current_score

        # Iterate over the neighboring landing schedules and find the one with the lowest score.
        for neighbor_df in neighbors:
            score = evaluate_landing_schedule(neighbor_df, airplane_stream)
            if score < next_score:
                next_state_df = neighbor_df
                next_score = score

        #If the next score is equal to the current score, the search is complete.
        if next_score == current_score:
            break

        # Update the landing schedule and the current score.
        landing_schedule_df = next_state_df
        current_score = next_score

    # Return the optimized landing schedule and the list of scores.
    return landing_schedule_df, scores


"""
    |----------------------------------------------------------------------------------------------|
    Schedules landings for an airplane stream using the Simulated Annealing optimization algorithm.
    
    Parameters: 
        - airplane_stream (list): List of Airplane objects representing the airplane stream.
        

    Returns:
        tuple: A tuple containing:
            - DataFrame containing the optimized landing schedule.
            - Final score of the optimized schedule indicating its performance.

    |----------------------------------------------------------------------------------------------|
    """
def simulated_annealing_schedule_landings(airplane_stream):
    def evaluate_adjusted_landing_schedule(schedule_df):
        landing_schedule_df = schedule_df.copy()
        total_score = 0
        for index, row in schedule_df.iterrows():
            airplane = next((ap for ap in airplane_stream if ap.id == row['Airplane ID']), None)
            if airplane:
                is_urgent = airplane.fuel_level_final < airplane.emergency_fuel or airplane.remaining_flying_time < row['Actual Landing Time']
                difference = abs(airplane.expected_landing_time - row['Actual Landing Time'])
                urgency_penalty = 100 if is_urgent else 0
                score = 1000 - difference - urgency_penalty
                landing_schedule_df.at[index, 'Score'] = score
        total_score = landing_schedule_df['Score'].sum()
        return total_score

    current_schedule = schedule_landings(airplane_stream)
    current_score = evaluate_adjusted_landing_schedule(current_schedule)
    #current_schedule['Efficiency Score'] = 1000
    T = 1.0  # Temperatura inicial alta
    T_min = 0.001  # Temperatura mínima
    alpha = 0.9  # Taxa de resfriamento

    while T > T_min:
        i = 0
        while i <= 100:
            new_schedule = current_schedule.copy()
            # Esta é uma maneira simplificada de gerar um vizinho.
            new_schedule = get_successors(new_schedule, airplane_stream)[0]  # Assumindo que get_successors retorna uma lista de DFs
            new_score = evaluate_adjusted_landing_schedule(new_schedule)
            delta = new_score - current_score
            if delta < 0 or math.exp(-delta / T) > random.uniform(0, 1):
                current_schedule = new_schedule
                current_score = new_score
            i += 1
        T = T * alpha

    return current_schedule, current_score


"""
|----------------------------------------------------------------------------------------------|
    Schedules landings for an airplane stream using the Tabu Search optimization algorithm.

    Parameters:
        airplane_stream (list): List of Airplane objects representing the airplane stream.
        max_iterations (int): Maximum number of iterations to perform.
        max_tabu_size (int): Maximum size of the tabu list.

    Returns:
        tuple: A tuple containing:
            - DataFrame containing the optimized landing schedule.
            - List of scores for each iteration, indicating the performance over time.

|----------------------------------------------------------------------------------------------|
"""

def tabu_search_schedule_landings(airplane_stream, max_iterations=1000, max_tabu_size=10):
    # Verificar se um avião é urgente
    for airplane in airplane_stream:
        airplane.is_urgent = airplane.fuel_level_final < airplane.emergency_fuel or airplane.remaining_flying_time < airplane.expected_landing_time
    
    # Declaração de variáveis
    landing_schedule_df = schedule_landings(airplane_stream)
    current_score = evaluate_landing_schedule(landing_schedule_df, airplane_stream)
    scores = []
    tabu_list = []
    it = 0

    # Ciclo while que efetua a busca tabu até atingir o número máximo de iterações
    while it < max_iterations:
        neighbors = get_successors(landing_schedule_df, airplane_stream)
        next_state_df = landing_schedule_df
        scores.append(current_score)
        next_score = current_score

        best_solution_df = landing_schedule_df
        best_solution_score = evaluate_landing_schedule(landing_schedule_df, airplane_stream)

        for neighbor_df in neighbors:
            neighbor_string = neighbor_df.to_string()
            score = evaluate_landing_schedule(neighbor_df, airplane_stream)
            if score < best_solution_score:
                best_solution_df = neighbor_df
                best_solution_score = score
            if neighbor_string not in tabu_list:
                if score < next_score:
                    next_state_df = neighbor_df
                    next_score = score
                tabu_list.append(neighbor_string)  # Add the neighbor to the tabu list as soon as it's generated
                if len(tabu_list) > max_tabu_size:
                    tabu_list.pop(0)

        if next_score >= current_score:
            if random.random() < 0.1:  # 10% chance to choose a random neighbor
                next_state_df = random.choice(neighbors)
                next_score = evaluate_landing_schedule(next_state_df, airplane_stream)
            else:
                next_state_df = best_solution_df
                next_score = best_solution_score

        landing_schedule_df = next_state_df
        current_score = next_score
        it += 1
    
    return landing_schedule_df, scores

def genetic_algorithm_schedule_landings(airplane_stream, population_size=50, generations=50, crossover_rate=0.8, mutation_rate=0.1):
    population = [generate_initial_schedule(airplane_stream) for _ in range(population_size)]
    best_schedule = None
    best_score = float('inf')
    best_individuals = []  # To store the best individuals from each generation

    for generation in range(generations):
        fitness_scores = [evaluate_landing_schedule(schedule, airplane_stream) for schedule in population]

        parents = select_parents(population, fitness_scores, population_size // 2)

        offspring = crossover(parents, crossover_rate)

        offspring = [mutate(child, mutation_rate, airplane_stream) for child in offspring]

        population = parents + offspring

        current_best_score = min(fitness_scores)
        if current_best_score < best_score:
            best_score = current_best_score
            best_schedule = population[fitness_scores.index(best_score)]
            best_individuals.append(best_schedule)  # Implementing elitism

        print(f"Generation {generation}: Best Score - {best_score}")

    # After the main genetic algorithm loop
    best_individuals = sorted(best_individuals, key=lambda x: evaluate_landing_schedule(x, airplane_stream))[:population_size]  # Select the top individuals
    population = best_individuals + population[len(best_individuals):]  # Replace the worst individuals with the best ones from previous generations

    return best_schedule, best_score

#pertence ao sa
def calculate_efficiency_score(schedule_df, airplane_stream):
    max_score_per_plane = 1000
    efficiency_scores = []

    for index, row in schedule_df.iterrows():
        airplane = next((ap for ap in airplane_stream if ap.id == row['Airplane ID']), None)
        if airplane:
            # Calcula o desvio do tempo previsto
            time_deviation = abs(airplane.expected_landing_time - row['Actual Landing Time'])

            # Calcula a eficiência
            # A eficiência será o score máximo menos o desvio do tempo, a menos que o avião seja urgente.
            efficiency_score = max(0, max_score_per_plane - time_deviation)
            if row['Urgent']:
                efficiency_score = max(0, efficiency_score - 100)  # Penalidade de 50 pontos para urgência

            efficiency_scores.append(efficiency_score)
        else:
            efficiency_scores.append(None)  # Caso não encontre o avião correspondente

    schedule_df['Score'] = efficiency_scores
    return schedule_df

    # score inicial: 100
    # desvio : 30 min dif (100-30=70)
    # urgencia 70-50 = 20
    # efic final: 20%
    


def main():
    print("\n" + "=" * 72)
    print("=" + "Welcome to the Airport Landing Scheduler".center(70) + "=")
    print("=" * 72)
    num_airplanes = get_input("Enter the number of airplanes for the simulation (between 1-1440): ", 
                              type_=int, min_=1, max_=1440, 
                              header="Simulation Setup - Number of Airplanes")
    min_fuel = get_input("Enter the minimum fuel level (in liters, between 1000-5000): ", 
                         type_=float, min_=1000, max_=5000, 
                         header="Simulation Setup - Minimum Fuel Level")
    max_fuel = get_input("Enter the maximum fuel level (in liters, between 1000-5000): ", 
                         type_=float, min_=min_fuel, max_=5000, 
                         header="Simulation Setup - Maximum Fuel Level")
    min_arrival_time = get_input("Enter the minimum expected arrival time (in minutes, between 10-1440): ", 
                                 type_=float, min_=10, max_=1440, 
                                 header="Simulation Setup - Minimum Arrival Time")
    max_arrival_time = get_input("Enter the maximum expected arrival time (in minutes, between 10-1440): ", 
                                 type_=float, min_=min_arrival_time, max_=1440, 
                                 header="Simulation Setup - Maximum Arrival Time")

    airplane_stream = generate_airplane_stream(num_airplanes, min_fuel, max_fuel, min_arrival_time, max_arrival_time)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    df_initial = pd.DataFrame(
        [(airplane.id, airplane.fuel_level, airplane.fuel_level_final, airplane.emergency_fuel,
          airplane.fuel_consumption_rate, airplane.expected_landing_time)
         for airplane in airplane_stream],
        columns=["Airplane ID", "Initial Fuel", "Final Fuel", "Emergency Fuel Level", "Consumption Rate",
                 "Expected Landing Time"])

    print("\nGenerated Airplane Stream DataFrame:")
    print(df_initial.to_string(index=False))


    while True:
        algorithm_choice = select_algorithm()

        if algorithm_choice == 1:
            print("Running Hill Climbing algorithm...")
            landing_schedule_df, scores = hill_climbing_schedule_landings(airplane_stream)
            print("Hill Climbing algorithm finished.")
            print("Final landing schedule:")
            print(landing_schedule_df.to_string(index=False))
            average_score = landing_schedule_df['Score'].mean()
            print("\nAverage Score: {:.2f}".format(average_score))
        elif algorithm_choice == 2:
            print("Running Simulated Annealing algorithm...")
            landing_schedule_df, _ = simulated_annealing_schedule_landings(airplane_stream)
            landing_schedule_df = calculate_efficiency_score(landing_schedule_df, airplane_stream)
            print("Simulated Annealing algorithm finished.")
            print("Final landing schedule and score:")
            print(landing_schedule_df.to_string(index=False))
            average_score = landing_schedule_df['Score'].mean()
            print("\nAverage Score: {:.2f}".format(average_score))
        elif algorithm_choice == 3:
            max_iterations = get_input("Enter the maximum number of iterations for the Tabu Search algorithm (between 100-1000): ", type_=int, min_=100, max_=1000)
            max_tabu_size = get_input("Enter the maximum size of the tabu list for the Tabu Search algorithm (between 5-15): ", type_=int, min_=5, max_=15)

            print("Running Tabu Search algorithm...")
            landing_schedule_df, scores = tabu_search_schedule_landings(airplane_stream, max_iterations, max_tabu_size)
            print("Tabu Search algorithm finished.")
            print("Final landing schedule:")
            print(landing_schedule_df.to_string(index=False))
            average_score = landing_schedule_df['Score'].mean()
            print("\nAverage Score: {:.2f}".format(average_score))
        elif algorithm_choice == 4:
            print("Running Genetic Algorithm...")
            best_schedule, best_score = genetic_algorithm_schedule_landings(airplane_stream)
            print("Genetic Algorithm finished.")
            print(f"Best landing schedule score: {best_score}")
            print(best_schedule)
            average_score = landing_schedule_df['Score'].mean()
            print("\nAverage Score: {:.2f}".format(average_score))

        continue_choice = input("Would you like to run another algorithm? (Y/N): ").strip().upper()
        if continue_choice != 'Y':
            break

if __name__ == "__main__":
    main()


