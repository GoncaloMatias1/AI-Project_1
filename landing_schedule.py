from airplane import Airplane


def generate_airplane_stream(num_airplanes):
    """
    Generates a list of Airplane objects with random attributes.

    Args:
        num_airplanes (int): The number of airplanes to generate.

    Returns:
        list: A list of Airplane objects.
    """
    return [Airplane(airplane_id=i) for i in range(1, num_airplanes + 1)]

def schedule_landings(airplane_stream):
    """
    Calculates the landing schedule for a list of airplanes based on priority.

    Args:
        airplane_stream (list): A list of Airplane objects.

    Returns:
        list: A list of tuples representing the landing schedule, where each tuple contains the airplane ID and landing time.
    """
    for airplane in airplane_stream:
        airplane.calculate_priority()

    emergencies = [plane for plane in airplane_stream if plane.emergency]
    non_emergencies = [plane for plane in airplane_stream if not plane.emergency]
    priority_queue = [(plane.priority, plane.expected_arrival_time, plane) for plane in airplane_stream]
    heapq.heapify(priority_queue)

    landing_schedule = []
    current_time = 0
    runway_availability = [0] * 3

    for airplane in emergencies:
        runway_index = runway_availability.index(min(runway_availability))
        landing_time = max(current_time, runway_availability[runway_index])
        landing_schedule.append((airplane.airplane_id, landing_time))
        runway_availability[runway_index] = landing_time + 3
        current_time = landing_time
        
    while priority_queue:
        priority, expected_landing, airplane = heapq.heappop(priority_queue)
        runway_index = runway_availability.index(min(runway_availability))
        landing_time = max(current_time, expected_landing, runway_availability[runway_index])

        landing_schedule.append((airplane.airplane_id, landing_time))
        runway_availability[runway_index] = landing_time + 3
        current_time = landing_time

    landing_schedule.sort(key=lambda x: x[1])
    return landing_schedule
