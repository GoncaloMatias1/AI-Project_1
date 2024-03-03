from simulation import generate_airplane_stream, schedule_landings
import pandas as pd

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

    df = pd.DataFrame(
        [(airplane.id, airplane.fuel_level, airplane.fuel_consumption_rate, airplane.expected_landing_time)
         for airplane in airplane_stream],
        columns=["Airplane ID", "Fuel Level", "Fuel Consumption Rate", "Expected Landing Time"])
    print("\nGenerated Airplane Stream DataFrame:")
    print(df.to_string(index=False))

    algorithm_choice = select_algorithm()
    if algorithm_choice == 1:
        print("Running Hill Climbing algorithm...")
        # Placeholder for Hill Climbing algorithm logic
    elif algorithm_choice == 2:
        print("Running Simulated Annealing algorithm...")
        # Placeholder for Simulated Annealing algorithm logic
    elif algorithm_choice == 3:
        print("Running Tabu Search algorithm...")
        # Placeholder for Tabu Search algorithm logic



if __name__ == "__main__":
    main()
