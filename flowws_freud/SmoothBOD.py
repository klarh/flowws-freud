import flowws
from flowws import Argument as Arg
import freud

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
    ]

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
        self.bonds = box.wrap(rijs)

        scope.setdefault('visuals', []).append(self)

    def draw_plato(self):
        import plato, plato.draw as draw
        prim = draw.SpherePoints(points=self.bonds, on_surface=True)
        scene = draw.Scene(prim, size=(3, 3), pixel_scale=100,
                           features=dict(additive_rendering=dict(invert=True)))
        return scene
