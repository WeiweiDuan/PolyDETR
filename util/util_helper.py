import os

def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        return "create a folder in %s"%(path)
    else:
        return "%s alread exist."%(path)

def remove_files(path):
    for root, directory, files in os.walk(path):
        for fname in files:
            os.remove(os.path.join(root, fname))
    return "remove all files in %a"%(path)