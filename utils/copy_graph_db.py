from arango import ArangoClient


def init_new_db(dbname):
    client = ArangoClient(hosts='http://ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:8529')
    sys_db = client.db('_system', username='root', password='nebula')

    if not sys_db.has_database(dbname):
        sys_db.create_database(dbname,users=[{'username': 'nebula', 'password': 'nebula', 'active': True}])

    db = client.db(dbname, username='nebula', password='nebula')
    
    if db.has_collection('Nodes'):
        ssn = db.collection('Nodes')
    else:
        ssn = db.create_collection('Nodes')

    if db.has_collection('Edges'):
        sse = db.collection('Edges')
    else:
        sse = db.create_collection('Edges',  edge=True)
    
    if db.has_collection('MovieToStory'):
        sse = db.collection('MovieToStory')
    else:
        sse = db.create_collection('MovieToStory',  edge=True)

    if db.has_graph('StoryGraph'):
        nebula_graph_storyg = db.graph('StoryGraph')
    else:
        nebula_graph_storyg = db.create_graph('StoryGraph')

    if not nebula_graph_storyg.has_edge_definition('Edges'):
        actors2asset = nebula_graph_storyg.create_edge_definition(
            edge_collection='Edges',
            from_vertex_collections=['Nodes'],
            to_vertex_collections=['Nodes']
        )

    
    if not nebula_graph_storyg.has_edge_definition('MovieToStory'):
        movie2actors = nebula_graph_storyg.create_edge_definition(
            edge_collection='MovieToStory',
            from_vertex_collections=['Movies'],
            to_vertex_collections=['Nodes']
        )

client = ArangoClient(hosts='http://ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:8529')
init_new_db("nebula_holywood")
db_new = client.db("nebula_holywood", username='nebula', password='nebula')
db_old = client.db("nebula_dev", username='nebula', password='nebula')
movie_id = ""
query_get_movies = 'FOR doc IN Movies FILTER doc.split == \'0\' AND doc.splits_total == \'30\' RETURN doc'
input()

cursor_movies = db_old.aql.execute(query_get_movies)
for i, movie in enumerate(cursor_movies):
    movie_id = movie['movie_id']
    query_get_semantic_nodes = 'FOR doc IN SemanticStoryNodes FILTER doc.movie_id == "{}" RETURN doc'.format(movie_id)
    query_get_semantic_edges = 'FOR doc IN SemanticStoryEdges FILTER doc.movie_id == "{}" RETURN doc'.format(movie_id)
    query_get_actors = 'FOR doc IN Actors FILTER doc.movie_id == "{}" RETURN doc'.format(movie_id)
    query_get_actions = 'FOR doc IN Actions FILTER doc.movie_id == "{}" RETURN doc'.format(movie_id)
    query_get_actor_to_actions = 'FOR doc IN ActorToAction FILTER doc.movie_id == "{}" RETURN doc'.format(movie_id)
    query_get_movie_to_story = 'FOR doc IN MovieToStory FILTER doc.movie_id == "{}" RETURN doc'.format(movie_id)
    
    insert_edge = 'INSERT @edge INTO Edges'
    insert_node = 'INSERT @node INTO Nodes'
    insert_movie = 'INSERT @movie INTO Movies'
    insert_actor = 'INSERT @actor INTO Nodes'
    insert_action = 'INSERT @action INTO Nodes'
    insert_movie_to_story = 'INSERT @edge INTO MovieToStory'
    
    bind_vars = {'movie': movie}
    db_new.aql.execute(insert_movie, bind_vars=bind_vars)
    print(i, ") Movie: ", movie['_id'] ," ",movie['movie_id'])
    cursor_actors = db_old.aql.execute(query_get_actors)
    for actor in cursor_actors:
        #print(actor)
        actor['actor_id'] = actor['_id']
        bind_vars = {'actor': actor}
        db_new.aql.execute(insert_actor, bind_vars=bind_vars)
    
    cursor_actions = db_old.aql.execute(query_get_actions)
    for action in cursor_actions:
        #print(action)
        action['action_id'] = action['_id']
        bind_vars = {'action': action}
        db_new.aql.execute(insert_action, bind_vars=bind_vars)
      
    cursor_actors_to_actions = db_old.aql.execute(query_get_actor_to_actions)
    for edge in cursor_actors_to_actions:
        _from = edge['_from'].replace("Actors/", "Nodes/")
        _to = edge['_to'].replace("Actions/", "Nodes/")
        edge['_from'] = _from
        edge['_to'] = _to
        bind_vars = {'edge': edge}
        db_new.aql.execute(insert_edge, bind_vars=bind_vars)
       
    s_nodes = db_old.aql.execute(query_get_semantic_nodes)
    for node in s_nodes:
        bind_vars = {'node': node}
        db_new.aql.execute(insert_node, bind_vars=bind_vars)
    
    s_edges = db_old.aql.execute(query_get_semantic_edges)
    for edge in s_edges:
        _from = edge['_from'].replace("SemanticStoryNodes/", "Nodes/")
        __to = edge['_to'].replace("Actors/", "Nodes/")
        _to = __to.replace("SemanticStoryNodes/", "Nodes/")
        edge['_from'] = _from
        edge['_to'] = _to
        bind_vars = {'edge': edge}
        db_new.aql.execute(insert_edge, bind_vars=bind_vars)

    cursor_movie_to_story = db_old.aql.execute(query_get_movie_to_story)
    for edge in cursor_movie_to_story:
        print(edge)
        _to = edge['_to'].replace("SemanticStoryNodes/", "Nodes/")
        edge['_from'] = _from
        edge['_to'] = _to
        print(edge)
        bind_vars = {'edge': edge}
        db_new.aql.execute(insert_movie_to_story, bind_vars=bind_vars)
         

    #     print(edge)
    #input()
