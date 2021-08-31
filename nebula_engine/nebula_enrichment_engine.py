
import threading
from arango import ArangoClient
from nebula_api.nebula_enrichment_api import NRE_API
import time

class NRE:
    def __init__(self):
        self.connect_db("nebula_development")
        self.engines = []
        self.new_plugin_registration()
        self.nre_api = NRE_API()
    
   
    def get_plugins(self):
        self.experts = []
        query = 'FOR doc IN nebula_experts RETURN doc'
        cursor = self.db.aql.execute(query)
        for data in cursor: 
            #print(data)
            self.experts.append(data)

    def connect_db(self, dbname):
        #client = ArangoClient(hosts='http://ec2-18-219-43-150.us-east-2.compute.amazonaws.com:8529')
        #client = ArangoClient(hosts='http://35.158.120.92:8529')
        client = ArangoClient(hosts='http://ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:8529')
        db = client.db(dbname, username='nebula', password='nebula')
        self.db = db

    def new_plugin_registration(self):
        self.get_plugins()
        for _expert in self.experts:
            #print(_expert)
            try:
                #port = _expert['port']
                mod = __import__(_expert['Name'], fromlist=[_expert['Class']])
                klass = getattr(mod, _expert['Class'])
                #klass().run_reconciliation()
                #self.socket.connect ("tcp://localhost:%s" % port)
                engine = {'filter': _expert['Filter'], 'klass': klass()}
                self.engines.append(engine)
            except ModuleNotFoundError:
                print ("Reconciliation Module ", _expert['Name'], " not implemented yet...")

    def start_plugins(self):
        print("Registered engines: ", self.engines)
        for engine in self.engines:
            klass =  engine['klass']
            _plugin = threading.Thread(target=klass.run_reconciliation, args=(), daemon=True)
            #klass.run_reconciliation()
            _plugin.start()
            #print ("Got reload request for plugin: ") 
    
    def main_loop(self):
        #Process 5 updates
        while True:
            self.nre_api.wait_for_finish(['actors', 'actions', 'SceneGraph'])
            self.start_plugins()
           

nre = NRE()
nre.start_plugins()
while True:
   time.sleep(3)
#nre.main_loop()        