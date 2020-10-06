class Tree:
    """Abstract class for Tree data structure. """

    # === Nest Position class (Container for node) ===
    class Position:
        """An abstracting representing a node of the tree. """

        def element(self):
            raise NotImplementedError()

        def __eq__(self, other):
            raise NotImplementedError()

        def __ne__(self, other):
            return not (self == other)

    # === Abstract methods to be implemented by subclass ===
    def root(self):
        raise NotImplementedError()

    def parent(self):
        raise NotImplementedError()

    def children(self, pos):
        # Generate an iteration of Positions representing pos' children.
        raise NotImplementedError()

    def num_children(self, pos):
        raise NotImplementedError()

    def __len__(self):
        """Number of elements in tree."""
        raise NotImplementedError()

    # === Class methods ===
    def is_root(self, pos):
        return self.root() == pos

    def is_leaf(self, pos):
        return self.num_children() == 0

    def is_empty(self):
        return len(self) == 0

    def height(self, pos):
        """ Return height of tree rooted at pos."""
        if pos is None:
            p = self.root()

        if self.is_leaf(pos):
            return 0
        else:
            return 1 + max(self.height(c) for c in self.children(pos))

