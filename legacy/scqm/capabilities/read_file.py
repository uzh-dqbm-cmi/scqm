import json


# Checks file exist and wheter its extension is correct
def check_file(file: str = None, ext: str = None):
    if file is None:
        print('no file found')
    if not file.endswith(ext):
        print('wrong file type')


def read_file(file: str = None):
    if file is None:
        print('no file found')

    if not file.endswith('ext'):
        print('wrong file type')
    try:
        f = open(file)
        # Load content of the the json file
        data = json.load(f)
    except IOError:
        print("File not accessible")
    finally:
        f.close()

    return data

