WITH users_target AS (
    SELECT 
        user_id
    FROM 
        ml_10m.users
    WHERE
        user_id <  :user_num
),
-- filter the rating data by the specified user_num 
filterd_ratings_by_user_num as(
    select 
        r.user_id, 
        movie_id, 
        rating, 
        timestamp
    from 
        users_target as u
    inner join 
        ml_10m.ratings as r
    on 
        r.user_id = u.user_id
), 
-- divide data into training data and testing data
split_ratings as (
    SELECT 
        ranked.user_id, 
        ranked.movie_id,
        ranked.rating, 
        CASE
            WHEN rank <= 5 THEN 'train'
            ELSE 'test'
        END AS label
    from (
        select 
            user_id, 
            movie_id, 
            rating,
            timestamp,  
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp) AS rank
        from 
            filterd_ratings_by_user_num
    ) as ranked
    order by 
        user_id
),
-- ==================================
--              train
-- ==================================
-- averaged ratings of the trainning data
ratings_avg_train as (
    select 
        movie_id, 
        avg(rating) as ave_rating
    from 
        split_ratings
    where 
        label = 'train'
    group by movie_id
), 
-- add title column on the averaged rating data table
movie_title_ave_rating_train as (
    select 
        a.movie_id, 
        title, 
        ave_rating
    from 
        ratings_avg_train as a 
    inner join 
        ml_10m.movies as m  
    on 
        a.movie_id = m.movie_id
    order by ave_rating desc
), 
-- Filter: filter the data with only ratings_num > {threshold}
movie_rating_counts as (
    select
        movie_id,
        count(rating) as rated_movies_count
    from
        split_ratings
    where 
        label = 'train'
    group by movie_id
    having count(rating) >= :threshold
),
-- This is the training data: 
-- movies that are most popular among all the users
filtered_movie_rating as (
    select 
        mt.movie_id, 
        title, 
        ave_rating, 
        rated_movies_count
    from  
        movie_title_ave_rating_train as mt
    inner join 
        movie_rating_counts as mr
    on 
        mt.movie_id = mr.movie_id
    order by ave_rating desc
)

select  * from filtered_movie_rating; 


-- 1	1	5536
-- 2	2	11202
-- 3	3	34217
-- 4	4	47597
-- 5	5	29082