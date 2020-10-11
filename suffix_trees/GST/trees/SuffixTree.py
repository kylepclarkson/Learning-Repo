from trees.Tree import Tree

class SuffixTree(Tree):

    class _SuffixNode:
        """
            _label: The _label associated with a traversal from the root to this node.
            _idx:   The starting index of the prefix associated to this node.
                    S[_idx:] returns the prefix.
        """
        __slots__ = '_label', '_idx'

        def __init__(self, label=None, idx=None):
            self._label = label
            self._idx = idx

        def __repr__(self):
            return f'{str(self._label)}:{self._idx}'

    def __init__(self):
        """ Suffix tree contains empty string at root node.
        """
        Tree.__init__(self, "")

    # ======== Accessor Functions ========
    def find_matching_node(self, string, pos=None, start_idx=0):
        """ Search through tree to find a node whose label matches string[start_idx:len(_label)].
        Returns node that matches string partially.
        """
        if pos is None:
            pos = self.root()

        for child in self.children(pos):
            # labels match
            if child.element()._label[0] == string[start_idx]:
                child_label = child.element()._label
                # Labels fully match, continue search with child node.
                if child_label == string[start_idx: len(child_label)]:
                    return self.find_matching_node(string, pos=child, start_idx=len(child_label-1))
                # Label partially matches, return child node.
                else:
                    return child

        # No match found.
        return pos

    def insert_prefix(self, prefix, idx):
        """ Insert prefix into current tree.
        """
        parent = self.find_matching_node(prefix)
        print('idx: ', idx, ' prefix: ', prefix, ' parent: ', parent)
        for child in self.children(parent):
            print('child label: ', child.element()._label)
            if child.element()._label[0] == prefix[0]:
                print('match found')
                # Intermediate node is added between parent and child.
                j = 0
                while j < len(child.element()._label) and \
                    child.element()._label[j] == prefix[j]:
                    j += 1
                # create intermediate node with label being matching parts between prefix and child's label.

                # Update tree structure
                intermediate_node = self._add(parent, self._SuffixNode(prefix[:j], -1))
                child._parent = intermediate_node
                intermediate_node._children[child] = child

                del parent._children[child]

                # Set label of child node to be unmatched part of child label.
                child.element()._label = child.element()._label[j:]
                # create new leaf node containing unmatched part of suffix.
                self._add(intermediate_node, self._SuffixNode(prefix[j+1:], idx))
                # break from for loop.
                break


        # New node is inserted as child of parent.
        self._add(parent, self._SuffixNode(prefix, idx))

    def naive_construction(self, string):
        """ A O(n^2) algorithm to construct a suffix tree for string with
        length n. Creates a new tree
        """
        if not self.is_empty():
            Tree.__init__(self, self._SuffixNode(None, -1))

        # ensure string ends with '$' character (termination.)
        if not string.endswith('$'):
            string += '$'

        # Insert all prefixes.
        for i in range(len(string)):
            self.insert_prefix(string[i:], i)
            print("After insert i: ", i, list(self.bfs()))



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

def test6():
    t = SuffixTree()
    t.naive_construction("xabxac")

    print(list(t.bfs()))

test6()

