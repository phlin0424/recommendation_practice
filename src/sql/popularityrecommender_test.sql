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
--             test
-- ==================================
-- get the "rated" movie list for every single users in test data
test_data as (
    select 
        user_id, 
        r.movie_id,
        title,
        rating
    from
        split_ratings as r
    inner join ( 
        select
            movie_id,
            title
        from 
            ml_10m.movies 
        ) as m
    on 
        m.movie_id = r.movie_id
    where 
        label = 'test'
    order by
        user_id
    
)

select * from test_data; 