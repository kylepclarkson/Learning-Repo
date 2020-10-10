from trees.Tree import Tree

class SuffixTree(Tree):

    def __init__(self, string=None):
        Tree.__init__(self, string)

    def naive_construction(self, string):
        """ A O(n^2) algorithm to construct a suffix tree for string with
        length n. Creates a new tree
        """



def get_tree_1():
    '''
              -- a --
         -b-     -c-      d-
       -e-
     -f--g-
    '''
    t = SuffixTree('a')
    root = t.root()
    b = t._add(root, 'b')
    c = t._add(root, 'c')
    d = t._add(root, 'd')
    e = t._add(b, 'e')
    f = t._add(e, 'f')
    g = t._add(e, 'g')
    return t

# Testing
def test1():
    t = get_tree_1()
    print(t.root())

def test2():
    t = SuffixTree()
    root = t.root()

    # insert axabx
    x1 = t._add(root, 'axabx')

    # insert xabx
    x2 = t._add(root, 'xabx')

    # TODO implement in _insert_between
    x3 = t._add(x1, 'xabx')
    x4 = t._add(x1, 'bx')
    t._replace(x1, 'a')

    print('bfs')
    for node in t.bfs():
        print(node.element())

    print('\ndfs')
    for node in t.dfs():
        print(node.element())

def test3():
    t = get_tree_1()
    print('\nBFS')
    for n in t.bfs():
        print(n.element())

    print('\nDFS')
    for n in t.dfs():
        print(n.element())

def test4():
    t = get_tree_1()
    t._delete_tree()
    t._add()

def test5():
    t = get_tree_1()
    print(t.height())
test5()

