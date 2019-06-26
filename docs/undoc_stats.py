import pickle
import sys

undoc = pickle.load(open('./build/coverage/undoc.pickle', "rb"))

classes_count = 0
methods_count = 0
functions_count = 0

for module, entities in undoc[0].items():
    for class_, methods in entities['classes'].items():
        if len(methods):
            methods_count += len(methods)
        else:
            classes_count += 1

    functions_count += len(entities['funcs'])

print("Undocumented classes:   {}".format(classes_count)) 
print("Undocumented methods:   {}".format(functions_count))
print("Undocumented functions: {}".format(functions_count))