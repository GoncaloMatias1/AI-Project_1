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
                                 min_=min_arrival_time, max_=1440)

    airplane_stream = generate_airplane_stream(num_airplanes)
    for airplane in airplane_stream:
        validate_airplane_data(airplane)

    df = pd.DataFrame(
        [(airplane.airplane_id, airplane.fuel_level, airplane.fuel_consumption_rate, airplane.expected_landing_time)
         for airplane in airplane_stream],
        columns=["Airplane ID", "Fuel Level", "Fuel Consumption Rate", "Expected Arrival Time"])
    print(df)

    algorithm_choice = select_algorithm()
    optimized_schedule, unable_to_land = run_simulation(algorithm_choice, airplane_stream)

    print("\nOptimized Landing Schedule:")
    optimized_landing_data = []
    for airplane_id, landing_time in optimized_schedule:
        airplane = next((plane for plane in airplane_stream if plane.airplane_id == airplane_id), None)
        if airplane is not None:
            wait_time = max(0, landing_time - airplane.expected_landing_time)  # em minutos
            fuel_consumed = (wait_time / 60) * airplane.fuel_consumption_rate  # convertendo tempo de espera para horas
            airplane.fuel_level = max(airplane.fuel_level - fuel_consumed, 0)  # evita combustível negativo

    # Cria a lista de pouso otimizada com os níveis de combustível atualizados
    optimized_landing_data = []
    for airplane in airplane_stream:
        landing_info = next((item for item in optimized_schedule if item[0] == airplane.airplane_id), None)
        if landing_info:
            optimized_landing_data.append({
                "Airplane ID": airplane.airplane_id,
                "Landing Time (min)": landing_info[1],
                "Fuel Remaining (liters)": round(airplane.fuel_level, 2)
            })

    # Converte a lista otimizada para DataFrame
    optimized_landing_df = pd.DataFrame(optimized_landing_data)
    print("\nFinal Optimized Landing Schedule DataFrame:")
    print(optimized_landing_df.to_string(index=False))

    total_fuel_remaining = sum(item["Fuel Remaining (liters)"] for item in optimized_landing_data)
    average_fuel_remaining = total_fuel_remaining / len(optimized_landing_data)
    print("\nSimulation Summary:")
    print(f"Total landings: {len(optimized_landing_data)}")
    print(f"Average fuel remaining: {average_fuel_remaining:.2f} liters")


    save_simulation_results([(data["Airplane ID"], data["Landing Time (min)"]) for data in optimized_landing_data], 'optimized_landing_schedule.txt')
    statistics = calculate_statistics([(data["Airplane ID"], data["Landing Time (min)"]) for data in optimized_landing_data])
    print("\nSimulation Statistics:")
    for stat, value in statistics.items():
        print(f"{stat.capitalize()}: {value:.2f}")

if __name__ == "__main__":
    main()
