"""Elo rating system for tracking agent skill across training."""

from __future__ import annotations


class EloRating:
    """Standard Elo rating tracker."""

    def __init__(self, k: float = 32, initial_rating: float = 1000) -> None:
        self.ratings: dict[str, float] = {}
        self.k = k
        self.initial_rating = initial_rating

    def get_rating(self, name: str) -> float:
        return self.ratings.get(name, self.initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, winner_name: str, loser_name: str) -> None:
        """Call after each episode. Winner is the agent that won."""
        ra = self.get_rating(winner_name)
        rb = self.get_rating(loser_name)
        ea = self.expected_score(ra, rb)
        eb = self.expected_score(rb, ra)
        self.ratings[winner_name] = ra + self.k * (1 - ea)
        self.ratings[loser_name] = rb + self.k * (0 - eb)

    def draw(self, name_a: str, name_b: str) -> None:
        """Call for draw (prey survives full episode = draw)."""
        ra = self.get_rating(name_a)
        rb = self.get_rating(name_b)
        ea = self.expected_score(ra, rb)
        eb = self.expected_score(rb, ra)
        self.ratings[name_a] = ra + self.k * (0.5 - ea)
        self.ratings[name_b] = rb + self.k * (0.5 - eb)
