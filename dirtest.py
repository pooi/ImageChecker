import os

dirname = "./"
# for dir in os.listdir(dirname):
#     if dir.startswith("."):
#         continue
#     if os.path.isdir(os.path.join(dirname, dir)):
#         print("Dir: ", os.path.join(dirname, dir))
#         for fn in os.listdir(os.path.join(dirname, dir)):
#             print(fn)
#         print()
#     else:
#         print(dir)
list = []

def checkDir(dir_name, list):
    # print("Dir: ", dir_name)
    for dir in os.listdir(dir_name):
        if dir.startswith("."):
            continue

        path = os.path.join(dir_name, dir)
        if os.path.isdir(path):
            checkDir(path, list)
        else:
            fname, ext = os.path.splitext(path)
            ext = ext.lower()
            if ext.endswith("jpeg") or ext.endswith("jpg") or ext.endswith("png") or ext.endswith("gif"):
                list.append(path)
    # print("Finish")

# checkDir(dirname, list)
# print(list)


def collectImage(dir_name, data):
    # print("Dir: ", dir_name)
    list = []
    for dir in os.listdir(dir_name):
        if dir.startswith("."):
            continue

        path = os.path.join(dir_name, dir)
        if os.path.isdir(path):
            collectImage(path, data)
        else:
            fname, ext = os.path.splitext(path)
            ext = ext.lower()
            if ext.endswith("jpeg") or ext.endswith("jpg") or ext.endswith("png") or ext.endswith("gif"):
                list.append(path)

    dir_name = dir_name.replace("\\", "/")
    folder = dir_name.split("/").pop()
    if len(list) > 0 and (not folder.startswith(".")):
        data[folder] = list

data = {}
collectImage(dirname, data)
print(data)
for key in data.keys():
    print(key, data[key])