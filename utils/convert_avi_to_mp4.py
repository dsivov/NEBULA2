import glob
from posixpath import basename
import os
def convert_avi_to_mp4(avi_file_path, output_name):
    os.system("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True

_files = glob.glob('/movies/*avi')
    #Example usage
for _file in _files:
    file_name = basename(_file)
    filename, file_extension = os.path.splitext(file_name)
    mp_file_name = "/dataset/hollywood/" + filename
    avi_file_name = _file
    print(avi_file_name + " TO " + mp_file_name)
    convert_avi_to_mp4(avi_file_name, mp_file_name)
