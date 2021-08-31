from arango import ArangoClient
from gdb.databaseconnect import DatabaseConnector as dbc

# def connect_db(dbname):
#     client = ArangoClient(hosts='http://ec2-18-159-140-240.eu-central-1.compute.amazonaws.com:8529')
#     # client = ArangoClient(hosts='http://18.159.140.240:8529')
#     db = client.db(dbname, username='nebula', password='nebula')
#     return (db)

def connect_db(dbname):

    # client = ArangoClient(hosts='http://ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:8529')
    # # client = ArangoClient(hosts='http://35.158.120.92:8529')
    # db = client.db(dbname, username='nebula', password='nebula')

    arangodb = dbc()
    db = arangodb.connect_db(dbname)

    return db


def get_all_movies(db):
    nebula_movies = {}
    query = "FOR doc in Movies RETURN {movie: doc}"
    cursor = db.aql.execute(query)
    for i, doc in enumerate(cursor):
        nebula_movies[i] = doc
    return nebula_movies


def get_all_stories(db):
    nebula_movies = {}
    query = "FOR doc in StoryLine RETURN {movie: doc}"
    cursor = db.aql.execute(query)
    for i, doc in enumerate(cursor):
        nebula_movies[i] = doc
    return nebula_movies

def nebula_connect(db_name: str) -> dict:
    conn = {'dbName': db_name,
           'username': 'nebula',
           'password': 'nebula',
           'hostname': '18.159.140.240',
           'protocol': 'http',
           'port': 8529}
    return conn