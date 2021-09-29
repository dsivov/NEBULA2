from arango import ArangoClient
from config_nebula.config import NEBULA_CONF

class DatabaseConnector():
    def __init__(self):
        config = NEBULA_CONF()
        self.arango_host = config.get_database_host()

    def connect_db(self, dbname):
        #client = ArangoClient(hosts='http://ec2-18-219-43-150.us-east-2.compute.amazonaws.com:8529')
        #client = ArangoClient(hosts='http://localhost:8529')
        client = ArangoClient(hosts=self.arango_host)
        db = client.db(dbname, username='nebula', password='nebula')
        return (db)
    
    def init_new_db(self, dbname):
        client = ArangoClient(hosts=self.arango_host)
        sys_db = client.db('_system', username='root', password='nebula')

        if not sys_db.has_database(dbname):
            sys_db.create_database(
                dbname, users=[{'username': 'nebula', 'password': 'nebula', 'active': True}])

        db = client.db(dbname, username='nebula', password='nebula')

        #StoryLine
        if db.has_collection('StoryLine'):
            ssn = db.collection('StoryLine')
        else:
            ssn = db.create_collection('StoryLine')

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

    # def init_nebula_db(self, dbname):
    #     # Initialize the ArangoDB client.
    #     client = ArangoClient(hosts='http://localhost:8529')

    #     # Connect to "_system" database as root user.
    #     # This returns an API wrapper for "_system" database.
    #     sys_db = client.db('_system', username='root', password='nebula')

    #     if not sys_db.has_database(dbname):
    #         sys_db.create_database(dbname,users=[{'username': 'nebula', 'password': 'nebula', 'active': True}])
      
    #     db = client.db(dbname, username='nebula', password='nebula')
    #     #Temp


    #     if db.has_collection('Actors'):
    #         actors = db.collection('Actors')
    #     else:
    #         actors = db.create_collection('Actors')

    #     if db.has_collection('Shots'):
    #         shots = db.collection('Shots')
    #     else:
    #         shots = db.create_collection('Shots')

    #     if db.has_collection('Positions'):
    #         positions = db.collection('Positions')
    #     else:
    #         positions = db.create_collection('Positions', edge=True)

    #     if db.has_collection('MovieToActors'):
    #         mov2act = db.collection('MovieToActors')
    #     else:
    #         mov2act = db.create_collection('MovieToActors', edge=True)

    #     if db.has_collection('MovieToStory'):
    #         mov2act = db.collection('MovieToStory')
    #     else:
    #         mov2act = db.create_collection('MovieToStory', edge=True)

    #     if db.has_collection('Movies'):
    #         movies = db.collection('Movies')
    #     else:
    #         movies = db.create_collection('Movies')

    #     if db.has_collection('Frames'):
    #         time_frames = db.collection('Frames')
    #     else:
    #         time_frames = db.create_collection('Frames')
        
    #     if db.has_collection('Properties'):
    #         time_frames = db.collection('Properties')
    #     else:
    #         time_frames = db.create_collection('Properties')
        
    #     if db.has_collection('Actions'):
    #         time_frames = db.collection('Actions')
    #     else:
    #         time_frames = db.create_collection('Actions')
        
    #     if db.has_collection('Relations'):
    #         time_frames = db.collection('Relations')
    #     else:
    #         time_frames = db.create_collection('Relations')

    #     if db.has_collection('SemanticStoryNodes'):
    #         ssn = db.collection('SemanticStoryNodes')
    #     else:
    #         ssn = db.create_collection('SemanticStoryNodes')
        
    #     if db.has_collection('SemanticStoryEdges'):
    #         sse = db.collection('SemanticStoryEdges')
    #     else:
    #         sse = db.create_collection('SemanticStoryEdges',  edge=True)

    #     if db.has_graph('AnnotationGraph'):
    #         nebula_graph_kl = db.graph('AnnotationGraph')
    #     else:
    #         nebula_graph_kl = db.create_graph('AnnotationGraph')
    #         if not nebula_graph_kl.has_edge_definition('ActorsToFrames'):
    #             actors2positions = nebula_graph_kl.create_edge_definition(
    #                 edge_collection='Positions',
    #                 from_vertex_collections=['Actors'],
    #                 to_vertex_collections=['Frames']
    #             )
    #         if not nebula_graph_kl.has_edge_definition('MovieToActors'):
    #             movie2actors = nebula_graph_kl.create_edge_definition(
    #                 edge_collection='MovieToActors',
    #                 from_vertex_collections=['Movies'],
    #                 to_vertex_collections=['Actors']
    #             )
    #         if not nebula_graph_kl.has_edge_definition('ActorToAsset'):
    #             actors2asset = nebula_graph_kl.create_edge_definition(
    #                 edge_collection='ActorToAsset',
    #                 from_vertex_collections=['Actors'],
    #                 to_vertex_collections=['Actors']
    #             )

    #     if db.has_graph('SceneGraph'):
    #         nebula_graph_sg = db.graph('SceneGraph')
    #     else:
    #         nebula_graph_sg = db.create_graph('SceneGraph')
    #         # if not nebula_graph_sg.has_edge_definition('ActorToAsset'):
    #         #     actors2asset = nebula_graph_sg.create_edge_definition(
    #         #         edge_collection='ActorToAsset',
    #         #         from_vertex_collections=['Actors'],
    #         #         to_vertex_collections=['Actors']
    #         #     )
    #         if not nebula_graph_sg.has_edge_definition('ActorToRelation'):
    #             actors2relations = nebula_graph_sg.create_edge_definition(
    #                  edge_collection='ActorToRelation',
    #                  from_vertex_collections=['Actors'],
    #                  to_vertex_collections=['Relations']
    #              )
    #         if not nebula_graph_sg.has_edge_definition('RealtionToProperty'):
    #             relations2propertys= nebula_graph_sg.create_edge_definition(
    #                  edge_collection='RelationToProperty',
    #                  from_vertex_collections=['Relations'],
    #                  to_vertex_collections=['Properties']
    #              )
    #         if not nebula_graph_sg.has_edge_definition('ActorToAction'):
    #             actors2actions = nebula_graph_sg.create_edge_definition(
    #                 edge_collection='ActorToAction',
    #                 from_vertex_collections=['Actors'],
    #                 to_vertex_collections=['Actions']
    #             )
    #         if not nebula_graph_sg.has_edge_definition('MovieToActors'):
    #             movie2actors = nebula_graph_sg.create_edge_definition(
    #                 edge_collection='MovieToActors',
    #                 from_vertex_collections=['Movies'],
    #                 to_vertex_collections=['Actors']
    #             )
        
    #     if db.has_graph('SceneGraphFull'):
    #         nebula_graph_sgf = db.graph('SceneGraphFull')
    #     else:
    #         nebula_graph_sgf = db.create_graph('SceneGraphFull')
    #         if not nebula_graph_sgf.has_edge_definition('Positions'):
    #             actors2positions = nebula_graph_sgf.create_edge_definition(
    #                 edge_collection='Positions',
    #                 from_vertex_collections=['Actors'],
    #                 to_vertex_collections=['Frames']
    #             )
    #         if not nebula_graph_sgf.has_edge_definition('ActorToRelation'):
    #             actors2relations = nebula_graph_sgf.create_edge_definition(
    #                 edge_collection='ActorToRelation',
    #                 from_vertex_collections=['Actors'],
    #                 to_vertex_collections=['Relations']
    #             )
    #         if not nebula_graph_sgf.has_edge_definition('RealtionToProperty'):
    #             relations2propertys = nebula_graph_sgf.create_edge_definition(
    #                 edge_collection='RelationToProperty',
    #                 from_vertex_collections=['Relations'],
    #                 to_vertex_collections=['Properties']
    #             )
    #         if not nebula_graph_sgf.has_edge_definition('ActorToAction'):
    #             actors2actions = nebula_graph_sgf.create_edge_definition(
    #                 edge_collection='ActorToAction',
    #                 from_vertex_collections=['Actors'],
    #                 to_vertex_collections=['Actions']
    #             )
    #         if not nebula_graph_sgf.has_edge_definition('MovieToActors'):
    #             movie2actors = nebula_graph_sgf.create_edge_definition(
    #                 edge_collection='MovieToActors',
    #                 from_vertex_collections=['Movies'],
    #                 to_vertex_collections=['Actors']
    #             )
        
    #     if db.has_graph('StoryGraph'):
    #         nebula_graph_storyg = db.graph('StoryGraph')
    #     else:
    #         nebula_graph_storyg = db.create_graph('StoryGraph')
    #         if not nebula_graph_storyg.has_edge_definition('SemanticStoryEdges'):
    #             actors2asset = nebula_graph_storyg.create_edge_definition(
    #                 edge_collection='SemanticStoryEdges',
    #                 from_vertex_collections=['SemanticStoryNodes'],
    #                 to_vertex_collections=['SemanticStoryNodes']
    #             )
    #         # if not nebula_graph_storyg.has_edge_definition('ActorToRelation'):
    #         #     actors2relations = nebula_graph_storyg.create_edge_definition(
    #         #          edge_collection='ActorToRelation',
    #         #          from_vertex_collections=['Actors'],
    #         #          to_vertex_collections=['Relations']
    #         #      )
    #         # if not nebula_graph_storyg.has_edge_definition('RealtionToProperty'):
    #         #     relations2propertys= nebula_graph_storyg.create_edge_definition(
    #         #          edge_collection='RelationToProperty',
    #         #          from_vertex_collections=['Relations'],
    #         #          to_vertex_collections=['Properties']
    #         #      )
    #         if not nebula_graph_storyg.has_edge_definition('ActorToAction'):
    #             actors2actions = nebula_graph_storyg.create_edge_definition(
    #                 edge_collection='ActorToAction',
    #                 from_vertex_collections=['Actors'],
    #                 to_vertex_collections=['Actions']
    #             )
    #         if not nebula_graph_storyg.has_edge_definition('MovieToStory'):
    #             movie2actors = nebula_graph_storyg.create_edge_definition(
    #                 edge_collection='MovieToStory',
    #                 from_vertex_collections=['Movies'],
    #                 to_vertex_collections=['SemanticStoryNodes']
    #             )

    #     return(db)

    def delete_db(self, dbname):
        client = ArangoClient(hosts='http://localhost:8529')
        # Connect to "_system" database as root user.
        # This returns an API wrapper for "_system" database.
        sys_db = client.db('_system', username='root', password='nebula')
        if not sys_db.has_database(dbname):
            print("NEBULADB not exist")
        else:
            sys_db.delete_database(dbname)

#For testing
def main():
    print()
    #
    vtdb = DatabaseConnector()
    #vtdb.delete_db()
    vtdb.init_nebula_db('nebula_dev')
    #db = vtdb.connect_db()

if __name__ == '__main__':
    main()
