-- Select data with user_id < user_num (a integer being specified) 
-- Filter the data with limited user number
WITH users_target AS (
    SELECT 
        user_id
    FROM 
        ml_10m.users
    WHERE
        user_id <  :user_num
),
-- Split the movie title into the title and the year
movies_with_years AS (
    SELECT 
        movie_id, 
        substring(title FROM '\((\d{4})\)$') AS movie_year, 
        TRIM(SUBSTRING(title FROM '^(.*)\s\(\d{4}\)$')) AS movie_title, 
        genres
    FROM
        ml_10m.movies
), 
-- Filter the rating data with the limited user number
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
-- connect the rating data to the movie year information
all_data as (
SELECT 
    user_id, 
    ratings.movie_id, 
    rating, 
    movie_title, 
    movie_year,
    genres, 
    timestamp
FROM
    ratings
INNER JOIN
    movies_with_years 
ON 
    ratings.movie_id = movies_with_years.movie_id
),
-- split the data into test and training data
ranked_data AS (
    SELECT 
        user_id, 
        movie_id, 
        rating, 
        movie_title, 
        movie_year, 
        genres, 
        timestamp, 
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp desc) AS rank
    FROM 
        all_data
), split_data as (
SELECT
    user_id,
    movie_id,
    rating,
    movie_title, 
    movie_year, 
    genres, 
    timestamp,
    CASE
        WHEN rank <= 5 THEN 'test'
        ELSE 'train'
    END AS label
FROM
    ranked_data
ORDER BY 
    user_id, timestamp
)
SELECT
    user_id,
    movie_id,
    rating,
    movie_title, 
    movie_year, 
    genres, 
    timestamp,
    label
FROM
    split_data
ORDER BY 
    user_id, timestamp 



-- rating   count
-- integer  bigint
-- 1	1	5536
-- 2	2	11202
-- 3	3	34217
-- 4	4	47597
-- 5	5	29082
