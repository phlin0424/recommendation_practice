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
)
SELECT * FROM ratings; 