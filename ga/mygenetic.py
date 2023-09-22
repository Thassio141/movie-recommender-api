

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
        
        total_weighted_rating = 0.0
        total_weight = 0.0

        movie_years = []
        
        user_liked_genres = set()  # Conjunto de gêneros que o usuário gosta
        
        # Recupere os gêneros que o usuário gosta a partir de suas avaliações
        user_ratings = RatingsRepository.find_by_userid(self.db, self.query_search)
        for rating in user_ratings:
            if rating.rating >= 3.5:
                movie = rating.movie
                if movie.genres:
                    if movie.genres not in user_liked_genres:
                        user_liked_genres.update(movie.genres.split("|"))
                
                if movie.year:
                    movie_years.append(movie.year)
        
        for movie_id in individual:
            # Recupere as avaliações do filme
            ratings_movies = RatingsRepository.find_by_movieid_list(self.db, [movie_id])

            if ratings_movies:
                # Calcula a média ponderada das avaliações para o filme
                weighted_rating = np.mean([obj_.rating for obj_ in ratings_movies])
                
                # Verifique se o filme pertence a um gênero que o usuário gosta
                movie = MovieRepository.find_by_id(self.db, movie_id)
                if movie.genres:
                    movie_genres = set(movie.genres.split("|"))
                    common_genres = user_liked_genres.intersection(movie_genres)
                    if common_genres:
                        # Atribua um peso maior ao filme se ele tiver gêneros em comum com o usuário
                        weighted_rating *= len(common_genres)  # Pode ajustar o peso conforme necessário

                if movie.year:
                    #VERIFICA SE O ANO DO FILME ESTA DENTRO DE UM RANGE DE ANOS QUE O USUARIO GOSTA PEGANDO O MAXIMO E O MINIMO
                    min_year = math.floor(min(movie_years))
                    max_year = math.ceil(max(movie_years))
                    if movie.year >= min_year and movie.year <= max_year:
                        weighted_rating *= 1.5

                total_weighted_rating += weighted_rating
                total_weight += 1.0

        if total_weight > 0:
            mean_weighted_rating = total_weighted_rating / total_weight
        else:
            mean_weighted_rating = 0.0

        return (mean_weighted_rating, )