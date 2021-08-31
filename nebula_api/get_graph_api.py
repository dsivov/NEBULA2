import json
import networkx as nx
from io import BytesIO

from nebula_api.cfg import Cfg

from nebula_api.databaseconnect import DatabaseConnector as dbc


def get_graph_as_dict(movie_id, db_name=None):
    if db_name is None:
        db_name = Cfg(['graphdb']).get('graphdb', 'name')

    db = dbc().connect_db(db_name)
    query = "FOR node,edge IN 1..1000 OUTBOUND @movie_id GRAPH 'SceneGraph' RETURN { node, edge } "
    bind_vars = {'movie_id': movie_id}
    edges = {}
    nodes = {}


    full_graph = db.aql.execute(
        query,
        bind_vars=bind_vars
    )
    if not full_graph.empty():
        nodes[movie_id] = {'_id': movie_id, 'description': movie_id}


    for line in full_graph:
        nodes[line['node']['_id']] = line['node']
        edges[line['edge']['_id']] = line['edge']

    graph = {
        'Nodes': nodes,
        'Edges': edges
    }
    return graph


def get_many_movies_graph_as_dict(movies_ids, db_name=None):
    if db_name is None:
        db_name = Cfg(['graphdb']).get('graphdb', 'name')

    full_graph = {
        'Nodes': {},
        'Edges': {}
    }

    for movie_id in movies_ids:
        movie_graph = get_graph_as_dict(movie_id, db_name)
        full_graph['Nodes'].update(movie_graph['Nodes'])
        full_graph['Edges'].update(movie_graph['Edges'])

    return full_graph


def get_nx_graph_from_dict(graph_dict):
    # The GML spec doesn't allow underscores in attribute names.
    # So we can't use graph_dict directly
    graph = nx.Graph()
    accessible_nodes_data = [
        (
            node_id,
            {key.replace('_', ''): value for key, value in node_data.items()}
        )
        for node_id, node_data in graph_dict['Nodes'].items()
    ]
    graph.add_nodes_from(accessible_nodes_data)

    accessible_edges_data = [
        (
            edge_data['_from'],
            edge_data['_to'],
            {key.replace('_', ''): value for key, value in edge_data.items()}
        )
        for edge_id, edge_data in graph_dict['Edges'].items()
    ]
    graph.add_edges_from(accessible_edges_data)

    return graph


def graph_dict_to_ml_file_data(graph_dict):
    ml_file = BytesIO()
    nx_graph = get_nx_graph_from_dict(graph_dict)
    nx.write_gml(nx_graph, ml_file)
    return ml_file.getvalue()

