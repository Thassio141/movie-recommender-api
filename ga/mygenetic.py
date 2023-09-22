

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
        if len(individual) != len(set(individual)):
            return (0.0, )

        if len(list(set(individual) - set(self.all_ids))) > 0:
            return (0.0, )

        ratings_movies = RatingsRepository.find_by_movieid_list(self.db, individual)

        if len(ratings_movies) > 0:
            mean_rating = np.mean([obj.rating for obj in ratings_movies])
        else:
            mean_rating = 0.0

        filtered_movies = [movie for movie in individual if mean_rating > 4.0]

        genre_counts = {}
        for movie_id in filtered_movies:
            movie = MovieRepository.find_by_id(self.db, movie_id)
            if movie:
                genres = movie.genres.split("|")
                for genre in genres:
                    if genre in genre_counts:
                        genre_counts[genre] += 1
                    else:
                        genre_counts[genre] = 1

        most_common_genre = max(genre_counts, key=genre_counts.get, default=None)

        genre_filtered_movies = [movie_id for movie_id in filtered_movies if MovieRepository.find_by_id(self.db, movie_id).genres.startswith(most_common_genre)]

        release_year_filter = {}
        for movie_id in genre_filtered_movies:
            movie = MovieRepository.find_by_id(self.db, movie_id)
            if movie:
                release_year = movie.year
                if release_year not in release_year_filter:
                    release_year_filter[release_year] = 0
                release_year_filter[release_year] += 1

        most_common_release_year = max(release_year_filter, key=release_year_filter.get, default=None)
        
        final_score = mean_rating + 0.5 * release_year_filter.get(most_common_release_year, 0)

        return (final_score, )