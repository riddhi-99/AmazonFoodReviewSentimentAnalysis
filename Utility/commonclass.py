import pickle
class CommonClass:

    def pickle_dump(file_name, file_object):
        obj1 = open(file_name,'wb')
        pickle.dump(file_object, obj1)
        obj1.close()

    def pickle_load(file_name):
        obj1 = open(file_name,'rb')
        file_object = pickle.load(obj1)
        obj1.close()
        return file_object