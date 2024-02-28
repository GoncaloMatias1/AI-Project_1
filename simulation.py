import random
from utils import get_input, validate_airplane_data, calculate_statistics, save_simulation_results, log_airplane_data
from landing_schedule import generate_airplane_stream, hill_climbing, simulated_annealing, tabu_search
import pandas as pd


def select_algorithm():
    print("\nSelect an optimization algorithm to run the simulation:")
    print("1. Hill Climbing")
    print("2. Simulated Annealing")
    print("3. Tabu Search")
    choice = get_input("Enter your choice (number): ", type_=int, min_=1, max_=3)
    return choice


def run_simulation(algorithm_choice, airplane_stream):
    if algorithm_choice == 1:
        print("Running Hill Climbing algorithm...")
        optimized_schedule, unable_to_land = hill_climbing(airplane_stream)
    elif algorithm_choice == 2:
        print("Running Simulated Annealing algorithm...")
        optimized_schedule, unable_to_land = simulated_annealing(airplane_stream)
    elif algorithm_choice == 3:
        print("Running Tabu Search algorithm...")
        optimized_schedule, unable_to_land = tabu_search(airplane_stream)
    else:
        raise ValueError("Invalid algorithm choice")
    return optimized_schedule, unable_to_land


def main():
    print("Welcome to the Airport Landing Scheduler.")
    print("Fuel levels should be between 1000 and 5000 liters.")
    print("Expected arrival times should be between 10 and 120 minutes.")

    num_airplanes = get_input("Enter the number of airplanes for the simulation: ", type_=int, min_=1)
    min_fuel = get_input("Enter the minimum fuel level (in liters): ", type_=float, min_=1000, max_=5000)
    max_fuel = get_input("Enter the maximum fuel level (in liters): ", type_=float, min_=min_fuel, max_=5000)
    min_arrival_time = get_input("Enter the minimum expected arrival time (in minutes): ", type_=float, min_=10,
                                 max_=120)
    max_arrival_time = get_input("Enter the maximum expected arrival time (in minutes): ", type_=float,
                                 min_=min_arrival_time, max_=120)

    airplane_stream = generate_airplane_stream(num_airplanes)
    first_airplane = True  # Inicia a lógica para o primeiro avião
    for airplane in airplane_stream:
        airplane.fuel_level = random.uniform(min_fuel, max_fuel)
        airplane.expected_landing_time = random.uniform(min_arrival_time, max_arrival_time)
        validate_airplane_data(airplane)
        log_airplane_data(airplane, first_airplane)  # Loga os dados do avião, controlando a sobrescrita do arquivo
        first_airplane = False  # Ajusta para os próximos aviões adicionarem ao arquivo

    df = pd.DataFrame(
        [(airplane.airplane_id, airplane.fuel_level, airplane.fuel_consumption_rate, airplane.expected_landing_time)
         for airplane in airplane_stream],
        columns=["Airplane ID", "Fuel Level", "Fuel Consumption Rate", "Expected Arrival Time"])

    print(df)

    algorithm_choice = select_algorithm()
    optimized_schedule, unable_to_land = run_simulation(algorithm_choice, airplane_stream)
    optimized_schedule.sort(key=lambda x: x[0])

    print("\nOptimized Landing Schedule:")
    total_fuel_remaining = 0
    for airplane_id, landing_time in optimized_schedule:
        airplane = next(plane for plane in airplane_stream if plane.airplane_id == airplane_id)
        wait_time = max(0, landing_time - airplane.expected_landing_time)
        fuel_consumed = airplane.fuel_consumption_rate * wait_time
        fuel_remaining = airplane.fuel_level - fuel_consumed
        if fuel_remaining < 0:
            print(f"Airplane {airplane_id} could not land due to insufficient fuel.")
        else:
            print(
                f"Airplane {airplane_id} landed after {round(landing_time)} minutes with {round(fuel_remaining)} liters of fuel remaining.")
            total_fuel_remaining += fuel_remaining

    average_fuel_remaining = round(total_fuel_remaining / len(airplane_stream), 2) if airplane_stream else 0
    print("\nSimulation Summary:")
    print(f"Total landings: {len(airplane_stream)}")
    print(f"Average fuel remaining: {average_fuel_remaining} liters")

    save_simulation_results([(airplane_id, round(landing_time)) for airplane_id, landing_time in optimized_schedule],
                            'optimized_landing_schedule.txt')

    statistics = calculate_statistics([(airplane_id, landing_time) for airplane_id, landing_time in optimized_schedule])
    print("\nSimulation Statistics:")
    for stat, value in statistics.items():
        print(f"{stat.capitalize()}: {value:.2f}")


if __name__ == "__main__":
    main()
