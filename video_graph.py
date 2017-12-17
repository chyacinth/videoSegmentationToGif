class VideoEdge:
    def __init__(self):
        self.n = self.m = self.w = 0
        self.r = False

    def __del__(self):
        pass


class VideoEdgeSorter:
    def __call__(self, g, h):
        return h.w>g.w


class VideoNode:
    def __init__(self):
        self.r = self.g = self.b = 0
        self.i = self.j = self.t = 0
        self.fx = self.fy = self.H = 0
        self.f = None
        self.max_w = 0.0
        self.l = self.id = 0
        self.n = 1
        self.h = []
        pass

    def __del__(self):
        pass


class VideoGraph:
    def __init__(self, N = None):
        if (N != None):
            self.__nodes = []
            for i in range(N):
                self.__nodes += [VideoNode()]
            # self.__nodes = [VideoNode()]*N
            # self.__edges= [VideoEdge()]*N
            self.__edges =[]
            # for i in range(N):
            #     self.__edges += [VideoEdge()]
        pass
        # type: int -> VideoGraph

    def __del__(self):
        pass

    def __eq__(self, graph):
        self.__nodes = graph.__nodes
        self.__edges = graph.__edges

    def setNode(self, n, node):
        self.__nodes[n] = node

    def addNode(self, node):
        self.__nodes.append(node)

    def addEdge(self, edge):
        self.__edges.append(edge)

    def getNode(self, n):
        assert (n>=0 and n<len(self.__nodes))
        return self.__nodes[n]

    def getEdge(self, n):
        assert (n>=0 and n<len(self.__edges))
        return self.__edges[n]

    def getNumNodes(self):
        return len(self.__nodes)

    def getNumEdges(self):
        return len(self.__edges)

    def sortEdges(self):
        self.__edges.sort(key = lambda e:e.w )

    def clearEdges(self):
        del self.__edges[:]

    def findNodeComponent(self, n):
        l = n.l
        id = n.id
        while (l != id):
            id = self.__nodes[l].id
            l = self.__nodes[l].l

        S = self.__nodes[l]
        assert (S.l == S.id)
        n.l = S.id
        return S

    def merge(self, S_n, S_m, e):
        S_m.l = S_n.id
        # Update histogram
        for i in range(S_n.H):
            S_n.h[i] += S_m.h[i]

        # Flow
        S_n.fx += S_m.fx
        S_n.fy += S_m.fy
        S_n.f += S_m.f

        # Update mean color.
        # S_n.r += S_m.r;
        # S_n.g += S_m.g;
        # S_n.b += S_m.b;

        # Update cound.
        S_n.n += S_m.n

        # Update maximum weight.
        S_n.max_w = max(max(S_n.max_w, S_m.max_w), e.w)

        # This edge has already been removed.
        e.r = True

