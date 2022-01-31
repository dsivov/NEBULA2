#! /bin/bash

#wget https://nebula-volume.s3.eu-central-1.amazonaws.com/backup.tgz -O /tmp/backup.tgz
docker run -v nebula_vol:/dbdata --name dbstore ubuntu /bin/bash
docker run --rm --volumes-from dbstore -v /tmp/:/backup ubuntu bash -c "cd /dbdata && tar zxvf /backup/backup.tgz --strip 1"


