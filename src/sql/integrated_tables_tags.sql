-- Select data with user_id < user_num (a integer being specified) 
WITH movies_with_years AS (
    SELECT 
        movie_id, 
        substring(title FROM '\((\d{4})\)$') AS movie_year, 
        TRIM(SUBSTRING(title FROM '^(.*)\s\(\d{4}\)$')) AS movie_title, 
        genres
    FROM
        ml_10m.movies
), 
tags AS (
    SELECT 
        movie_id,
        STRING_AGG(tag, '|') AS concatenated_tags
    FROM 
        ml_10m.tags  
    GROUP BY 
        movie_id 
), 
all_data AS (
SELECT 
    movies_with_years.movie_id, 
    movie_title, 
    movie_year,
    genres, 
    COALESCE(concatenated_tags, '') AS tags  -- Replace NULL with an empty string
FROM 
    movies_with_years
LEFT JOIN
    tags
ON 
    movies_with_years.movie_id = tags.movie_id
)

SELECT
    movie_id,
    movie_title,
    movie_year,
    genres, 
    tags
FROM 
    all_data
ORDER BY 
    movie_id