-- Select data with user_id < user_num (a integer being specified) 
-- 
WITH users_target AS (
    SELECT 
        user_id
    FROM 
        ml_10m.users
    WHERE
        user_id <  :user_num
),
movies AS (
    SELECT 
        movie_id, 
        substring(title FROM '\((\d{4})\)$') AS movie_year, 
        TRIM(SUBSTRING(title FROM '^(.*)\s\(\d{4}\)$')) AS movie_title, 
        genres
    FROM
        ml_10m.movies
), 
tags_of_movie AS (
    SELECT 
        movie_id, 
        string_agg(tag, '|') AS tags
    FROM 
        ml_10m.tags
    GROUP BY
        movie_id
),
ratings AS (
    SELECT 
        users_target.user_id as user_id, 
        movie_id, 
        rating, 
        timestamp
    FROM
        users_target
    INNER JOIN
        (
            SELECT 
                user_id, 
                movie_id, 
                rating, 
                timestamp
            FROM
                ml_10m.ratings
        )  AS temp_ratings
    ON
        users_target.user_id = temp_ratings.user_id  --filtering off user_id>user_num
),  
movies_with_tags AS (
    SELECT 
        movies.movie_id, 
        movie_year, 
        movie_title, 
        genres, 
        tags_of_movie.tags AS tags
    FROM 
        movies
    INNER JOIN
        tags_of_movie
    ON 
        movies.movie_id = tags_of_movie.movie_id
)
SELECT 
    user_id, 
    ratings.movie_id, 
    rating, 
    movie_title, 
    movie_year,
    genres, 
    tags, 
    timestamp
FROM
    ratings
INNER JOIN
    movies_with_tags
ON 
    ratings.movie_id = movies_with_tags.movie_id; 