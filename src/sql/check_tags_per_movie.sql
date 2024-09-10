SELECT STRING_AGG(tag, ' | ') AS concatenated_tags FROM ml_10m.tags  group by movie_id 

