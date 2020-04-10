import collections

import flowws
from flowws import Argument as Arg
import freud
import numpy as np

@flowws.add_stage_arguments
class SmoothBOD(flowws.Stage):
    """Compute and display Bond Orientational Order Diagrams (BOODs)"""
    ARGS = [
        Arg('num_neighbors', '-n', int, default=4,
            help='Number of neighbors to compute'),
        Arg('use_distance', '-d', bool, default=False,
            help='Use distance, rather than num_neighbors, to find bonds'),
        Arg('r_max', type=float, default=2,
            help='Maximum radial distance if use_distance is given'),
        Arg('average', type=bool, default=False,
            help='If True, average the BOOD'),
        Arg('average_keys', type=[str],
            help='List of scope keys to generate distinct series when averaging'),
    ]

    def __init__(self, *args, **kwargs):
        self._data_cache = collections.defaultdict(list)
        self._run_cache_keys = set()
        super().__init__(*args, **kwargs)

    def run(self, scope, storage):
        """Compute the bonds in the system"""
        box = freud.box.Box.from_box(scope['box'])
        positions = scope['position']

        aq = freud.AABBQuery(box, positions)
        args = dict(num_neighbors=self.arguments['num_neighbors'],
                    exclude_ii=True, r_guess=self.arguments['r_max'])
        if self.arguments['use_distance']:
            args['mode'] = 'ball'
            args['r_max'] = self.arguments['r_max']

        nlist = aq.query(positions, args).toNeighborList()
        rijs = positions[nlist.point_indices] - positions[nlist.query_point_indices]
        bonds = box.wrap(rijs)

        key_names = self.arguments.get('average_keys', [])
        self._last_data_key = tuple(scope[name] for name in key_names)
        if self.arguments['average']:
            if scope.get('cache_key', object()) not in self._run_cache_keys:
                self._data_cache[self._last_data_key].append(bonds)
            if 'cache_key' in scope:
                self._run_cache_keys.add(scope['cache_key'])
        else:
            self._data_cache[self._last_data_key] = [bonds]

        scope.setdefault('visuals', []).append(self)
        scope.setdefault('visual_link_rotation', []).append(self)

    def draw_plato(self):
        import plato, plato.draw as draw
        bonds = np.concatenate(self._data_cache[self._last_data_key], axis=0)
        prim = draw.SpherePoints(points=bonds, on_surface=True)
        scene = draw.Scene(prim, size=(3, 3), pixel_scale=100,
                           features=dict(additive_rendering=dict(invert=True)))
        return scene
