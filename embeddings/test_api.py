import milvus
from embeddings.nebula_embeddings_api import EmbeddingsLoader
import sys
import time


#milvus = index.create_milvus_database()
if len(sys.argv) < 2:
    print("Usage: ", sys.argv[0], "text/movie/scene")
    exit()
if sys.argv[1] == "movie":
    # index = EmbeddingsLoader('clip4string', debug=True)
    index = EmbeddingsLoader('clip4string', debug=True)
    #index.get_story_labels()
    while (True):
            _key = input("Enter movie id: ")
            #Key -> movie id (Movies/1111111), Top -> number of similar movies
            #sims = index.get_similar_movies(_key, 10) #Normal processinfg
            start_time = time.time()
            sims = index.get_similar_movies(_key, 10) 
            print("--- %s seconds ---" % (time.time() - start_time))
            if sims:
                print("----------------------") 
                print("Top 10 Sims for Movie: " + _key)
                for sim in sims.values():
                    print(sim)
                print("----------------------")  
            else:
                print("Please rebuild index")
elif (sys.argv[1] == "text"):
    index = EmbeddingsLoader('clip2bert')
  
    while (True):
        _query = []
        #input("Enter text query, ype end to stop: ")
        _query_ = input("Enter text query, type end to stop: ")
        _query.append(_query_)
        while (_query_ != 'end'):
            _query_ = input("Then? ")
            _query.append(_query_)
        

        vector = index.encoder.encode_with_transformers(_query)
        sims = index.index.search_vector(10, vector.tolist()[0]) 
        similars = []
        print("----------------------") 
        print("Top 10 Sims for Text: ")
        for dist, data in sims:
            print(dist, " ", data)
           #print(data)

elif (sys.argv[1] == "scene"):
    index = EmbeddingsLoader('clip4scene')
    while (True):
        _query = input("Enter movie id: ")
        scenes = index.get_scenes(_query)
        print("----------------------") 
        print("Scenes: ")
        print(scenes)
        #for scene in scenes:
        scene_query = input("Enter scene number: ")
        vectors = index.get_scene_vector(_query, scene_query)
        sims = index.get_similar_scenes(10, vectors)
        print(sims)
        #for sim in sims:
        #    print(sim)

        # for scin sims.values():
        #     print(sim)
        # print("----------------------") 
