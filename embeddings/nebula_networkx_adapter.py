from adbnx_adapter.arangodb_networkx_adapter_base import Networkx_Adapter_Base
import networkx as nx
import numpy as np
from arango import ArangoClient
from networkx.classes.function import is_empty


class Nebula_Networkx_Adapter(Networkx_Adapter_Base):

    def __init__(self, conn):
        if self.is_valid_conn(conn):
            url = conn["hostname"]
            user_name = conn["username"]
            password = conn["password"]
            dbName = conn["dbName"]
            if 'port' in conn:
                port = str(conn['port'])
            else:
                port = '8529'
            if 'protocol' in conn:
                protocol = conn['protocol']
            else:
                protocol = "https"
            con_str = protocol + "://" + url + ":" + port
            client = ArangoClient(hosts=con_str)
            self.db = client.db(dbName, user_name, password)
        else:
            print(
                "The connection information you supplied is invalid, please check and try again!")

        return

    def is_valid_conn(self, conn):
        valid_con_info = True

        if not "hostname" in conn:
            print("hostname is missing in connection")
        if not "username" in conn:
            print("Username is missing in connection")
            valid_con_info = False
        if not "password" in conn:
            print("Password is missing in connection")
            valid_con_info = False
        if not "dbName" in conn:
            print("Database is missing in connection")
            valid_con_info = False

        return valid_con_info

    def is_valid_graph_attributes(self, graph_config):
        valid_config = True

        if not 'vertexCollections' in graph_config:
            print("Graph attributes do not contain vertex collections")
            valid_config = False
        if not 'edgeCollections' in graph_config:
            print("Graph attributes do not contain edge collections")
            valid_config = False

        return valid_config
    
    def convert_lables(self, nmbr):
        return(str(int(round(nmbr))))
    
    def convert_lables_to_word(self, labels):
        word_labels = []
        if labels[1] > 0.9:
            word_labels.append("accurate")
        else:
            word_labels.append("inaccurate")
        if (labels[3] - labels[2]) < 10:
            word_labels.append("shot")
        else:
            word_labels.append("long")
        return(word_labels)

    def nebula_get_all_movies(self):
        nebula_movies={}
        query = "FOR doc in Movies RETURN {movie: doc}"
        cursor = self.db.aql.execute(query)
        for i, doc in enumerate(cursor): 
            nebula_movies[i] = doc
        return(nebula_movies)  
            
    def create_nebula_graph(self,graph_name, graph_attributes, _filter = ""):
        nebula_labels = {}
        nebula_metadata = {}
        i = 0
        graph = nx.DiGraph()
        if self.is_valid_graph_attributes(graph_attributes):         
            for k, v in graph_attributes['vertexCollections'].items():
                if not _filter:
                    query = "FOR doc in %s " % (k)
                else:
                    query = "FOR doc in %s FILTER doc.`movie_id` == \'%s\' " % (k, _filter) 
                cspl = [s + ':' + 'doc.' + s for s in v]
                cspl.append('_id: doc._id')
                csps = ','.join(cspl)
                query = query + " RETURN { " + csps + "}"
                cursor = self.db.aql.execute(query)
                for doc in cursor:
                    if "With" in doc['description']:
                        doc['description'] = "With"
                    if "Then" in doc['description']:
                        doc['description'] = "Then" 
                    # if "With" in doc['description'] or "Then" in doc['description']:
                    #     _class = doc['description']
                    # else:
                    _class = ''.join([i for i in doc['description'] if not i.isdigit()])
                    doc['_class'] = _class
                    doc['_object'] = doc['description']
                    graph.add_node(doc['_id'], attr_dict=doc)
                    #print(doc['labels'])
                    nebula_metadata[i] = (doc['description'], doc['_id'])
                    #print(doc['description'])
                    #curr_lable = self.convert_lables_to_word(doc['labels'])
                    if "Then" in doc['description']  or  "With" in doc['description']:
                         nebula_labels[i] = []
                    else:
                        nebula_labels[i]=list(map(self.convert_lables,doc['labels']))  
                        nebula_labels[i].pop(0) 
                        #_prefix = ''.join([i for i in doc['description'] if not i.isdigit()])
                        #nebula_labels[i] = [_prefix + "_" + sub for sub in nebula_labels[i]]
                        #nebula_labels[i] = curr_lable             
                    i = i + 1
            for k, v in graph_attributes['edgeCollections'].items():
                if not _filter:
                    query = "FOR doc in %s " % (k)
                else:
                    query = "FOR doc in %s FILTER doc.`movie_id` == \'%s\' " % (k, _filter)
                cspl = [s + ':' + 'doc.' + s for s in v]
                cspl.append('_id: doc._id')
                csps = ','.join(cspl)
                query = query + "RETURN { " + csps + "}"
                #print("QUERY: ", query)
                cursor = self.db.aql.execute(query)
                for doc in cursor:
                    graph.add_edge(doc['_from'], doc['_to'])
        
        return graph, nebula_labels, nebula_metadata
    

