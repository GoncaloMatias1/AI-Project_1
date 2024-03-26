import random
import math
import pandas as pd
from plotting import plot_scores
import time
from simulation import (generate_airplane_stream, schedule_landings, evaluate_landing_schedule, get_successors, get_tabu_successors, generate_initial_schedule, select_parents, crossover, mutate)


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
    start_time = time.time()  # Start timing the algorithm
    times = []  # To store elapsed times of each score calculation
    scores = []  # To store scores of each iteration

    # Mark urgent airplanes based on their fuel levels and expected landing times.
    for airplane in airplane_stream:
        airplane.is_urgent = (airplane.fuel_level_final < airplane.emergency_fuel or
                              airplane.remaining_flying_time < airplane.expected_landing_time)

    # Generate an initial landing schedule using the schedule_landings function.
    landing_schedule_df = schedule_landings(airplane_stream)

    # Initialize the current score and store the initial score.
    current_score = evaluate_landing_schedule(landing_schedule_df, airplane_stream)
    scores.append(current_score)
    times.append(time.time() - start_time)  # Record the time for the initial score

    # Repeat the following steps until no improvement is found.
    while True:
        # Get all neighboring landing schedules from the current schedule.
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

        # If the next score is equal to the current score, the search is complete.
        if next_score == current_score:
            break

        # Update the landing schedule and the current score, and record the time.
        landing_schedule_df = next_state_df
        current_score = next_score
        scores.append(current_score)
        times.append(time.time() - start_time)  # Record the time for this score

    # Return the optimized landing schedule, the times, and the list of scores.
    return landing_schedule_df, times, scores



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
        total_score = 0
        for index, row in schedule_df.iterrows():
            airplane = next((ap for ap in airplane_stream if ap.id == row['Airplane ID']), None)
            if airplane:
                is_urgent = airplane.fuel_level_final < airplane.emergency_fuel or airplane.remaining_flying_time < row['Actual Landing Time']
                difference = abs(airplane.expected_landing_time - row['Actual Landing Time'])
                urgency_penalty = 100 if is_urgent else 0
                score = difference + urgency_penalty
                total_score += score
        return total_score

    current_schedule = schedule_landings(airplane_stream)
    current_score = evaluate_adjusted_landing_schedule(current_schedule)
    T = 1.0  # Temperatura inicial alta
    T_min = 0.001  # Temperatura mínima
    alpha = 0.9  # Taxa de resfriamento
    scores = []  # Armazenar os scores a cada iteração
    times = []  # Armazenar os tempos a cada iteração
    start_time = time.time() 

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
        scores.append(current_score)  # Adiciona o score atual à lista de scores
        times.append(time.time() - start_time)  # Adiciona o tempo atual à lista de tempos
        T = T * alpha
        return current_schedule, scores, times

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
        # Aqui geram-se os vizinhos da solução/estado atual e guarda-se o score da solução inicial na lista de scores
        # (como melhor solução encontrada até ao momento)
        neighbors = get_tabu_successors(landing_schedule_df, airplane_stream)
        next_state_df = landing_schedule_df
        scores.append(current_score)
        next_score = current_score

        best_solution_df = landing_schedule_df
        best_solution_score = evaluate_landing_schedule(landing_schedule_df, airplane_stream)


        # Iteração entre os vizinhos para encontrar a melhor solução entre eles
        for neighbor_df in neighbors:
            score = evaluate_landing_schedule(neighbor_df, airplane_stream)
            if score < best_solution_score:
                best_solution_df = neighbor_df
                best_solution_score = score
            if neighbor_df.to_string() not in tabu_list and score < next_score:
                next_state_df = neighbor_df
                next_score = score

        if next_score >= current_score:
            next_state_df = best_solution_df
            next_score = best_solution_score

        landing_schedule_df = next_state_df
        current_score = next_score
        tabu_list.append(next_state_df.to_string())
        if len(tabu_list) > max_tabu_size:
            tabu_list.pop(0)
        it += 1
    
    return landing_schedule_df, scores

def genetic_algorithm_schedule_landings(airplane_stream, population_size=50, generations=50, crossover_rate=0.8, mutation_rate=0.05):
    population = [generate_initial_schedule(airplane_stream) for _ in range(population_size)]
    best_schedule = None
    best_score = float('inf')

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

        print(f"Generation {generation}: Best Score - {best_score}")

    return best_schedule, best_score


def calculate_efficiency_score(schedule_df, airplane_stream):
    max_score_per_plane = 100
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
                efficiency_score = max(0, efficiency_score - 50)  # Penalidade de 50 pontos para urgência

            efficiency_scores.append(efficiency_score)
        else:
            efficiency_scores.append(None)  # Caso não encontre o avião correspondente

    schedule_df['Efficiency Score'] = efficiency_scores
    return schedule_df

    # score inicial: 100
    # desvio : 30 min dif (100-30=70)
    # urgencia 70-50 = 20
    # efic final: 20%
    s


def main():
    print("\n" + "=" * 72)
    print("=" + "Welcome to the Airport Landing Scheduler".center(70) + "=")
    print("=" * 72)
    num_airplanes = get_input("Enter the number of airplanes for the simulation (between 1-40): ", 
                              type_=int, min_=1, max_=40, 
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
            landing_schedule_df, times, scores = hill_climbing_schedule_landings(airplane_stream)
            print("Hill Climbing algorithm finished.")
            print("Final landing schedule:")
            print(landing_schedule_df.to_string(index=False))
            plot_scores(times, scores, algorithm_name='Hill Climbing', filename='hill_climbing_performance.png')
        elif algorithm_choice == 2:
            print("Running Simulated Annealing algorithm...")
            landing_schedule_df, scores, times = simulated_annealing_schedule_landings(airplane_stream)
            landing_schedule_df = calculate_efficiency_score(landing_schedule_df, airplane_stream)
            print("Simulated Annealing algorithm finished.")
            print("Final landing schedule and score:")
            print(landing_schedule_df.to_string(index=False))
            plot_scores(times, scores, algorithm_name='Simulated Annealing', filename='simulated_annealing_performance.png')

        elif algorithm_choice == 3:
            max_iterations = get_input("Enter the maximum number of iterations for the Tabu Search algorithm (between 100-10000): ", type_=int, min_=100, max_=10000)
            max_tabu_size = get_input("Enter the maximum size of the tabu list for the Tabu Search algorithm (between 5-20): ", type_=int, min_=5, max_=20)

            print("Running Tabu Search algorithm...")
            landing_schedule_df, scores = tabu_search_schedule_landings(airplane_stream, max_iterations, max_tabu_size)
            print("Tabu Search algorithm finished.")
            print("Final landing schedule:")
            print(landing_schedule_df.to_string(index=False))
        elif algorithm_choice == 4:
            print("Running Genetic Algorithm...")
            best_schedule, best_score = genetic_algorithm_schedule_landings(airplane_stream)
            print("Genetic Algorithm finished.")
            print(f"Best landing schedule score: {best_score}")
            print(best_schedule)

        continue_choice = input("Would you like to run another algorithm? (Y/N): ").strip().upper()
        if continue_choice != 'Y':
            break

if __name__ == "__main__":
    main()


