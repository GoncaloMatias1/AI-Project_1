import random
import math
import pandas as pd
from simulation import (generate_airplane_stream, schedule_landings,evaluate_landing_schedule, get_successors)

def get_input(prompt, type_=None, min_=None, max_=None):
    while True:
        try:
            value = input(prompt)
            if type_ is not None:
                value = type_(value)
            if min_ is not None and value < min_:
                raise ValueError(f"Value should not be less than {min_}.")
            if max_ is not None and value > max_:
                raise ValueError(f"Value should not be greater than {max_}.")
            return value
        except ValueError as e:
            print(f"Invalid input: {e}")
            continue

def select_algorithm():
    print("\nSelect an optimization algorithm to run the simulation:")
    print("1. Hill Climbing")
    print("2. Simulated Annealing")
    print("3. Tabu Search")
    choice = get_input("Enter your choice (number): ", type_=int, min_=1, max_=3)
    return choice

def hill_climbing_schedule_landings(airplane_stream):
    for airplane in airplane_stream:
        airplane.is_urgent = airplane.fuel_level_final < airplane.emergency_fuel or airplane.remaining_flying_time < airplane.expected_landing_time

    landing_schedule_df = schedule_landings(airplane_stream)
    current_score = evaluate_landing_schedule(landing_schedule_df, airplane_stream)
    scores = []
    while True:
        neighbors = get_successors(landing_schedule_df, airplane_stream)
        next_state_df = landing_schedule_df
        scores.append(current_score)
        next_score = current_score

        for neighbor_df in neighbors:
            score = evaluate_landing_schedule(neighbor_df, airplane_stream)
            if score < next_score:
                next_state_df = neighbor_df
                next_score = score

        if next_score == current_score:
            break

        landing_schedule_df = next_state_df
        current_score = next_score

    return landing_schedule_df, scores

def simulated_annealing_schedule_landings(airplane_stream):
    def evaluate_adjusted_landing_schedule(schedule_df):
        total_score = 0
        for index, row in schedule_df.iterrows():
            airplane = next((ap for ap in airplane_stream if ap.id == row['Airplane ID']), None)
            if airplane:
                # Calculando urgência aqui baseado na lógica do seu problema
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

    while T > T_min:
        i = 0
        while i <= 100:
            new_schedule = current_schedule.copy()
            # Esta é uma maneira simplificada de gerar um vizinho; adapte conforme necessário.
            new_schedule = get_successors(new_schedule, airplane_stream)[0]  # Assumindo que get_successors retorna uma lista de DFs
            new_score = evaluate_adjusted_landing_schedule(new_schedule)
            delta = new_score - current_score
            if delta < 0 or math.exp(-delta / T) > random.uniform(0, 1):
                current_schedule = new_schedule
                current_score = new_score
            i += 1
        T = T * alpha

    return current_schedule, current_score

def calculate_efficiency_score(schedule_df, airplane_stream):
    efficiency_scores = []  # Inicializa uma lista vazia para armazenar os scores de eficiência

    for index, row in schedule_df.iterrows():
        airplane = next((ap for ap in airplane_stream if ap.id == row['Airplane ID']), None)
        if airplane:
            # Supõe-se que o 'Urgent' seja um booleano e que o 'Actual Landing Time' esteja em minutos
            urgency_score = 100 if row['Urgent'] else 0
            deviation = abs(row['Actual Landing Time'] - airplane.expected_landing_time)
            efficiency_score = 100 - deviation + urgency_score  # Exemplo: 100 pontos base - desvio + urgência
            efficiency_scores.append(efficiency_score)  # Adiciona o score calculado à lista
        else:
            efficiency_scores.append(None)  # Se não encontrar o avião correspondente, adiciona um valor None

    schedule_df['Efficiency Score'] = efficiency_scores  # Adiciona a lista como uma nova coluna no DataFrame
    return schedule_df


def main():
    print("Welcome to the Airport Landing Scheduler.")
    num_airplanes = get_input("Enter the number of airplanes for the simulation (between 1-40): ", type_=int, min_=1,max_=40)
    min_fuel = get_input("Enter the minimum fuel level (in liters, between 1000-5000): ", type_=float, min_=1000,max_=5000)
    max_fuel = get_input("Enter the maximum fuel level (in liters, between 1000-5000): ", type_=float, min_=min_fuel,max_=5000)
    min_arrival_time = get_input("Enter the minimum expected arrival time (in minutes, between 10-1440): ", type_=float,min_=10, max_=1440)
    max_arrival_time = get_input("Enter the maximum expected arrival time (in minutes, between 10-1440): ", type_=float,min_=min_arrival_time, max_=1440)

    airplane_stream = generate_airplane_stream(num_airplanes, min_fuel, max_fuel, min_arrival_time, max_arrival_time)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Generate DataFrame without scheduling to print the initial state
    df_initial = pd.DataFrame(
        [(airplane.id, airplane.fuel_level, airplane.fuel_level_final, airplane.emergency_fuel,
          airplane.fuel_consumption_rate, airplane.expected_landing_time)
         for airplane in airplane_stream],
        columns=["Airplane ID", "Initial Fuel", "Final Fuel", "Emergency Fuel Level", "Consumption Rate",
                 "Expected Landing Time"])

    print("\nGenerated Airplane Stream DataFrame:")
    print(df_initial.to_string(index=False))

    algorithm_choice = select_algorithm()

    if algorithm_choice == 1:
        print("Running Hill Climbing algorithm...")
        landing_schedule_df, scores = hill_climbing_schedule_landings(airplane_stream)
        print("Hill Climbing algorithm finished.")
        print("Final landing schedule:")
        print(landing_schedule_df.to_string(index=False))
    elif algorithm_choice == 2:
        print("Running Simulated Annealing algorithm...")
        landing_schedule_df, _ = simulated_annealing_schedule_landings(airplane_stream)
        landing_schedule_df = calculate_efficiency_score(landing_schedule_df, airplane_stream)
        print("Simulated Annealing algorithm finished.")
        print("Final landing schedule and score:")
        print(landing_schedule_df.to_string(index=False))
    elif algorithm_choice == 3:
        print("Running Tabu Search algorithm...")
        # Add tabu search logic here if applicable


if __name__ == "__main__":
    main()

