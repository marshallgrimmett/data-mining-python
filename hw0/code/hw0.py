import operator

def test_print():
    print "Hello world!"

def list_set_length():
    items_list= [2, 1, 2, 3, 4, 3, 2, 1]
    items_set = {2, 1, 2, 3, 4, 3, 2, 1}

    print(len(items_list))
    print(len(items_set))

def set_intersect():
    S = {1,4,7}
    T = {1,2,3,4,5,6}
    return {x for x in S for y in T if x == y}

def three_tuples():
    S = {-4, -2, 1, 2, 5, 0}
    return [(i, j, k) for i in S for j in S for k in S if i+j+k==0]

def dict_init():
    mydict = {'Hopper':'Grace', 'Einstein':'Albert', 'Turing':'Alan', 'Lovelace':'Ada'}
    return mydict

def dict_find(dlist, k):
    return [dlist[k] for x in dlist if x == k]

def file_line_count():
    f = open("stories.txt", "r")
    count = 0
    for line in f:
        count += 1
    return(count)

def make_inverse_index(strlist):
    idx = {}
    for i, doc in enumerate(strlist):
        for word in doc.split():
            if word not in idx:
                idx.update({word : {i}})
            else:
                idx[word].add(i)
    return idx

def or_search(inverseIndex, query):
    docs = set()
    for word in query:
        if word in inverseIndex:
            docs = docs.union(inverseIndex[word])
    return docs

def and_search(inverseIndex, query):
    docs = []
    for word in query:
        if word in inverseIndex:
            docs.append(inverseIndex[word])
    final = set()
    for d in docs[0]:
        count = 0
        for i in docs:
            if d in i:
                count += 1
        if count == len(query):
            final.add(d)
    return final

def most_similar(inverseIndex, query):
    docs = []
    for word in query:
        if word in inverseIndex:
            docs.append(inverseIndex[word])
    final = {}
    for i in docs:
        for j in i:
            if j not in final:
                final[j] = 1
            else:
                final[j] += 1
    finalSorted = sorted(final.items(), key=operator.itemgetter(1), reverse=True)
    return [i[0] for i in finalSorted]


if __name__ == '__main__':
    test_print()
    list_set_length()
    print(set_intersect())
    print(three_tuples())
    print(dict_init())
    dlist = dict_init()
    print(dict_find(dlist, 'Hopper'))
    print(file_line_count())

    # Search Engine
    f = open("stories.txt", "r")
    inverseIndex = make_inverse_index(f)
    query = ['the', 'a', 'supply', 'burden', 'tent']
    print(or_search(inverseIndex, query))
    query = ['A', 'prep', 'course', 'for', 'the']
    print(and_search(inverseIndex, query))
    query = ['A', 'prep', 'course', 'for', 'the', 'a', 'lodging']
    print(most_similar(inverseIndex, query))