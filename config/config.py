class NEBULA_CONF:
    def __init__(self) -> None:
        self.ARANGO_HOST = 'http://ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:8529'
        self.ARANGO_DB = "nebula_development"
        self.ELASIC_HOST = 'http://tnnb2_master:NeBuLa_2@ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:9200/'
        self.ELASTIC_INDEX = "datadriven"
        self.S3BUCKET = "nebula-frames"
        self.MLV_SERVER = '172.31.7.226'
        self.MLV_PORT = '19530'  # default value

    def get_database_name(self):
        return(self.ARANGO_DB)
    
    #http://tnnb2_master:NeBuLa_2@ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:9200/
    def get_elastic_host(self):
        return(self.ELASIC_HOST)

    def get_elastic_index(self):
        return(self.ELASTIC_INDEX)

    def get_database_host(self):
        return(self.ARANGO_HOST)
    
    def get_s3_bucket(self):
        return(self.S3BUCKET)
        
    def get_milvus_server(self):
        return(self.MLV_SERVER, self.MLV_PORT)
