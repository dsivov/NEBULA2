#!/bin/bash
#!/bin/bash

# fileid="1latFtkP7Xsg5z-ww_g2PO44r39SRBdb9"
# filename="weights.zip"
# curl --insecure -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
# curl --insecure -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
# rm -f cookie
# unzip weights.zip


gdown https://drive.google.com/uc?id=1latFtkP7Xsg5z-ww_g2PO44r39SRBdb9
unzip weights.zip