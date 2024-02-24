def calculate_landing_priority(airplane):
    """
    Calculates the landing priority for an airplane based on configurable criteria.

    Args:
        airplane (Airplane): The airplane object for which to calculate priority.

    Returns:
        float: The calculated landing priority score.

    Raises:
        ValueError: If any required attribute is missing from the airplane object.
    """

    required_attributes = ["fuel_level", "expected_arrival_time"]
    for attr in required_attributes:
        if not hasattr(airplane, attr):
            raise ValueError(f"Missing required attribute: {attr}")

    # Define weights for different criteria (adjust as needed)
    fuel_weight = 0.7
    arrival_time_weight = 0.2
    emergency_weight = 1.0

    # Calculate priority based on weighted criteria
    priority = (
        fuel_weight * (airplane.fuel_level / 1000)  # Normalize fuel level
        - arrival_time_weight * airplane.expected_arrival_time / 60  # Normalize time to minutes
    )

    # Add bonus for emergencies
    if airplane.emergency:
        priority += emergency_weight

    return priority
