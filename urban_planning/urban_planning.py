


class TerrainMap:
    def __init__(self, size, max_residential, max_industrial, max_commercial):
        self.size = size
        self._map = self.make_map()
        self.max_residential = max_residential
        self.max_industrial = max_industrial
        self.max_commerial = max_commercial

    def make_map(self):
        _map = {}
        for x in range(1, self.size + 1):
            for y in range(1, self.size + 1):
                _map[(x, y)] = set()
        return _map

    @classmethod
    def from_file(cls, path):
        """ maximum number of industrial, commercial, and residential locations (respectively) """
        with open(path, "r") as file:
            lines = file.readlines()
        tmap = cls(size=len(lines[3]), max_residential=lines[2], max_commercial=lines[1], max_industrial=lines[0])

        for line in lines[3:]:
            print(line)
        return tmap

if __name__ == '__main__':
    path = "/home/kevin/Downloads/urban 1.txt"
    tmap = TerrainMap.from_file(path)
    print(tmap)
