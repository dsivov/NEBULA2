#     NEBULA2 - Next generation nebula 
#       Project structure
1) experts - code for external engines: input->movie or frames, output-> labels, vectors or bboxes
2) nebula_api - code for any shared resources (databases, storage)
3) nebula_engine - scheduler with plugins, each plugin must be under "plugins" directory, see examples
4) webui2 - WebUI code of nebula project

