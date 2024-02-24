import random
from utils import get_input, validate_airplane_data, calculate_statistics, save_simulation_results
from landing_schedule import schedule_landings, generate_airplane_stream


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
    for airplane in airplane_stream:
        airplane.fuel_level = random.uniform(min_fuel, max_fuel)
        airplane.expected_arrival_time = random.uniform(min_arrival_time, max_arrival_time)
        validate_airplane_data(airplane)

    landing_schedule = schedule_landings(airplane_stream)

    print("\nLanding Schedule:")
    total_fuel_remaining = 0
    for airplane_id, landing_time in landing_schedule:
        airplane = next(plane for plane in airplane_stream if plane.airplane_id == airplane_id)
        wait_time = max(0, landing_time - airplane.expected_arrival_time)
        fuel_consumed = airplane.fuel_consumption_rate * wait_time
        fuel_remaining = airplane.fuel_level - fuel_consumed
        print(
            f"Airplane {airplane_id} landed after {round(landing_time)} minutes with {round(fuel_remaining)} liters of fuel remaining.")
        total_fuel_remaining += fuel_remaining

    average_fuel_remaining = round(total_fuel_remaining / len(airplane_stream)) if airplane_stream else 0
    print("\nSimulation Summary:")
    print(f"Total landings: {len(airplane_stream)}")
    print(f"Average fuel remaining: {average_fuel_remaining} liters")

    # Optionally save the results to a file
    save_simulation_results(landing_schedule, 'landing_schedule.txt')

    # If you want to see the statistics:
    statistics = calculate_statistics(landing_schedule)
    print("\nSimulation Statistics:")
    for stat, value in statistics.items():
        if stat.endswith("_wait_time"):
            print(f"{stat.capitalize()}: {round(value)} minutes")
        else:
            print(f"{stat.capitalize()}: {value:.2f}")


if __name__ == "__main__":
    main()
