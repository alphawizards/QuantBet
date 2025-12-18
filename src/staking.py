def calculate_bet_size(bankroll: float, assessed_probability: float, decimal_odds: float, kelly_fraction: float = 0.25) -> float:
    """
    Calculates the specific dollar amount to wager based on a Fractional Kelly approach.

    Args:
        bankroll: Total bankroll available.
        assessed_probability: The model's estimated probability of winning (0.0 to 1.0).
        decimal_odds: The odds offered by the bookmaker (e.g., 1.90).
        kelly_fraction: The fraction of the full Kelly to bet (default 1/4 or 0.25).

    Returns:
        float: The dollar amount to wager. Returns 0.0 if the bet has no value (EV <= 0).
    """
    if assessed_probability <= 0 or assessed_probability >= 1:
        raise ValueError("Probability must be between 0 and 1 exclusive of extremes for safety.")

    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1.")

    # b = Net odds received on the wager (odds - 1)
    b = decimal_odds - 1

    # p = Probability of winning
    p = assessed_probability

    # q = Probability of losing (1 - p)
    q = 1 - p

    # Full Kelly Formula: f* = (bp - q) / b
    # Equivalent to: f* = p - (q / b)
    full_kelly_percentage = (b * p - q) / b

    # Apply safety fraction
    bet_percentage = full_kelly_percentage * kelly_fraction

    # Ensure non-negative bet size
    if bet_percentage <= 0:
        return 0.0

    bet_size = bankroll * bet_percentage

    return round(bet_size, 2)

# Usage Example:
if __name__ == "__main__":
    bankroll = 10000.0
    prob = 0.55 # 55% chance of winning
    odds = 1.91 # Standard -110 juice

    bet = calculate_bet_size(bankroll, prob, odds)
    print(f"Bankroll: ${bankroll}")
    print(f"Prob: {prob}, Odds: {odds}")
    print(f"Recommended Wager (1/4 Kelly): ${bet}")
