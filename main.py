from simulation import generate_airplane_stream, schedule_landings
import pandas as pd
from simulation import (generate_airplane_stream, schedule_landings,
                        evaluate_landing_schedule, get_successors)

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
    # Initial state
    landing_schedule_df = schedule_landings(airplane_stream)
    current_score = evaluate_landing_schedule(landing_schedule_df, airplane_stream)
    scores = []
    while True:
        neighbors = get_successors(landing_schedule_df, airplane_stream)
        next_state_df = landing_schedule_df
        scores.append(current_score)
        next_score = current_score
        

        # Examine all neighbors to find the best one
        for neighbor_df in neighbors:
            score = evaluate_landing_schedule(neighbor_df, airplane_stream)
            if score < next_score:
                next_state_df = neighbor_df
                next_score = score

        # If no better neighboring state is found, we've reached a local maximum
        if next_score == current_score:
            break

        # Move to the neighboring state
        landing_schedule_df = next_state_df
        current_score = next_score

    return landing_schedule_df, scores

def main():
    print("Welcome to the Airport Landing Scheduler.")
    num_airplanes = get_input("Enter the number of airplanes for the simulation (between 1-40): ", type_=int, min_=1, max_=40)
    min_fuel = get_input("Enter the minimum fuel level (in liters, between 1000-5000): ", type_=float, min_=1000, max_=5000)
    max_fuel = get_input("Enter the maximum fuel level (in liters, between 1000-5000): ", type_=float, min_=min_fuel, max_=5000)
    min_arrival_time = get_input("Enter the minimum expected arrival time (in minutes, between 10-1440): ", type_=float, min_=10, max_=1440)
    max_arrival_time = get_input("Enter the maximum expected arrival time (in minutes, between 10-1440): ", type_=float, min_=min_arrival_time, max_=1440)

    airplane_stream = generate_airplane_stream(num_airplanes, min_fuel, max_fuel, min_arrival_time, max_arrival_time)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    landing_schedule_df = schedule_landings(airplane_stream)
    df = pd.DataFrame(
        [(airplane.id, airplane.fuel_level, airplane.fuel_level_final,airplane.emergency_fuel, airplane.fuel_consumption_rate, airplane.expected_landing_time)
         for airplane in airplane_stream],
        columns=["Airplane ID", "Initial Fuel","Final Fuel","Emergency Fuel Level", "Consumption Rate", "Expected Landing Time"])
    df = df.merge(landing_schedule_df.rename(columns={"Airplane": "Airplane ID", "Landing Time": "Actual Landing Time", "Landing Strip": "Landing Strip"}), on="Airplane ID", how="left")
    df = df.sort_values("Actual Landing Time")
    print("\nGenerated Airplane Stream DataFrame:")
    print(df.to_string(index=False))

    algorithm_choice = select_algorithm()
    if algorithm_choice == 1:
        print("Running Hill Climbing algorithm...")
        landing_schedule_df, scores = hill_climbing_schedule_landings(airplane_stream)
        print("Hill Climbing algorithm finished.")
        print("Final landing schedule:")
        print(landing_schedule_df.to_string(index=False))
        print("Scores at each iteration:")
        print(scores)
    elif algorithm_choice == 2:
        print("Running Simulated Annealing algorithm...")
    elif algorithm_choice == 3:
        print("Running Tabu Search algorithm...")



if __name__ == "__main__":
    main()
