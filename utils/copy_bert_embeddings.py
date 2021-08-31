from arango import collection
import milvus
from numpy import delete
from nebula_api.databaseconnect import DatabaseConnector as dbc
from milvus import Milvus, IndexType, MetricType, Status
from nebula_api.milvus_api import MilvusAPI
_HOST = '172.31.7.226'
_PORT = '19530'  # default value
# _PORT = '19121'  # default http value

# Vector parameters
# dimension of vector
milvus_api = MilvusAPI(
    'milvus', 'bert_embeddings_development', 'nebula_development', 768)

_INDEX_FILE_SIZE = 320  # max file size of stored index
arangodb = dbc()
db = arangodb.connect_db("nebula_development")
milvus = Milvus(_HOST, _PORT)
status, idx = milvus.list_id_in_segment('bert_embeddings_0808', '1628501948192090000')
for id in idx:
    status, vector = milvus.get_entity_by_id('bert_embeddings_0808', [0, id])
    query = 'FOR doc IN milvus_bert_embeddings_0808 \
         FILTER doc.milvus_key == @milvus_key RETURN doc'
    bind_vars = {'milvus_key': str(id)}
    cursor = db.aql.execute(query, bind_vars=bind_vars)
    for meta in cursor:
        print([meta])
        milvus_api.insert_vectors([vector[1]],[meta])
