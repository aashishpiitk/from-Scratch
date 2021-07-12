import os
import shutil

def create_folder(folder_name):
    folderpath = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        if os.path.exists(folderpath):
            print('Folder Created', folderpath)
    else:
        delete_folder(folder_name)
        create_folder(folder_name)
    return folderpath

#create_folder('yoyo')


def delete_folder(folder_name):
    folderpath = os.path.join(os.getcwd(),folder_name)
    if(os.path.exists(folderpath)):
        shutil.rmtree(folderpath)
        if not os.path.exists(folderpath):
            print('Folder Removed', folderpath)

#delete_folder('yoyo')

# import re
# s = '/content/folder/132.jpg'
# num = re.compile('\w{1,}')
# print(num.findall(os.path.split(s)[1])[0])
