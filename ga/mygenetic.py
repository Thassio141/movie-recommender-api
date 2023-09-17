

from ga.algorithm import Algorithm
from sqlalchemy.orm import Session
from fastapi import Depends
import numpy as np
import math 

from db.database import get_db
from db.repositories import UserRepository, MovieRepository, RatingsRepository

class MyGeneticAlgorithm(Algorithm):

    def __init__(self, query_search, individual_size, population_size, p_crossover, p_mutation, all_ids, max_generations=100, size_hall_of_fame=1, fitness_weights=(1.0, ), seed=42, db=None) -> None:


        super().__init__(
            individual_size, 
            population_size, 
            p_crossover, 
            p_mutation, 
            all_ids, 
            max_generations, 
            size_hall_of_fame, 
            fitness_weights, 
            seed)
        
        self.db = db
        self.all_ids = all_ids
        self.query_search = query_search
        


    def evaluate(self, individual):
        unique_individual = list(set(individual))
        if len(unique_individual) != len(individual):
            return (0.0, )

        if any(movie_id not in self.all_ids for movie_id in unique_individual):
            return (0.0, )
        
        ratings_movies = RatingsRepository.find_by_movieid_list(self.db, unique_individual)

        if not ratings_movies:
            return (0.0, )

        ratings = [obj.rating for obj in ratings_movies]
        mean_rating = np.mean(ratings)
        return (mean_rating, )

