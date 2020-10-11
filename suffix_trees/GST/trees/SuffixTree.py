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

    # ================ Accessor Functions ================
    def path_to_matching_prefix(self, string, pos=None, start_idx=0, path=None):
        """ Search through tree to find a node whose label matches string[start_idx:len(_label)].
        Returns Path traced from root to node node that matches .
        """
        if path is None:
            path = []
        if pos is None:
            pos = self.root()
            path.append(pos)

        for child in self.children(pos):
            # labels match
            if child.element()._label[0] == string[start_idx]:
                child_label = child.element()._label
                # Labels fully match, continue search with child node.
                if child_label == string[start_idx: len(child_label)]:
                    return path.append(self.path_to_matching_prefix(string, pos=child, start_idx=len(child_label)))
                # Label partially matches, return parent node.
                else:
                    return path


        # No match found.
        return path

    def insert_suffix(self, prefix, idx):
        """ Insert prefix into current tree.
        """
        parent_pos = self.path_to_matching_prefix(prefix)[-1]

        has_inserted = False
        for child_pos in self.children(parent_pos):
            if child_pos.element()._label[0] == prefix[0]:
                # Intermediate node is added between parent and child.
                j = 0
                while j < len(child_pos.element()._label) and \
                    child_pos.element()._label[j] == prefix[j]:
                    j += 1

                # Update tree structure
                intermediate_pos = self._add(parent_pos, self._SuffixNode(prefix[:j], -1))
                intermediate_node = self._validate(intermediate_pos)

                child_node = self._validate(child_pos)
                child_node._parent = intermediate_node
                intermediate_node._children[child_node] = child_node
                parent_node = self._validate(parent_pos)
                del parent_node._children[child_node]

                # Set label of child node to be unmatched part of child label.
                child_pos.element()._label = child_pos.element()._label[j:]
                # create new leaf node containing unmatched part of suffix.
                self._add(intermediate_pos, self._SuffixNode(prefix[j:], idx))
                # break from for loop.
                has_inserted = True
                break

        # New node is inserted as child of parent.
        if not has_inserted:
            self._add(parent_pos, self._SuffixNode(prefix, idx))

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
            self.insert_suffix(string[i:], i)



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
    print(t.height())

test6()

