import os
from PIL import Image

p_dir = r'###'
for c_dir in os.listdir(p_dir):
    c_dir = os.path.join(p_dir, c_dir)

    for filename in os.listdir(c_dir):
        try :
            with Image.open(c_dir + "/" + filename) as im:
                print('ok')
        except :
            print(c_dir + "/" + filename)
            os.remove(c_dir + "/" + filename)
