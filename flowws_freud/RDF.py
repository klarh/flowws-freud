import flowws
from flowws import Argument as Arg
import freud

@flowws.add_stage_arguments
class RDF(flowws.Stage):
    """Compute and plot the radial distribution function (RDF)"""
    ARGS = [
        Arg('bins', '-b', int, default=64,
            help='Number of bins to use'),
        Arg('r_min', type=float, default=0,
            help='Minimum radial distance'),
        Arg('r_max', type=float, required=True,
            help='Maximum radial distance'),
        Arg('bond_max', type=float, default=0,
            help='Render bonds that have length up to the given distance'),
        Arg('bond_width', type=float, default=.1,
            help='Width of drawn bonds, if enabled'),
    ]

    def run(self, scope, storage):
        """Compute and provide the RDF"""
        compute = freud.density.RDF(
            self.arguments['bins'], self.arguments['r_max'],
            self.arguments.get('r_min', 0))

        box = freud.box.Box.from_box(scope['box'])
        system = freud.locality.AABBQuery(box, scope['position'])
        compute.compute(system)
        self.r, self.rdf = compute.bin_centers, compute.rdf

        if self.arguments.get('bond_max', None):
            import plato.draw

            query_args = dict(
                mode='ball', r_max=self.arguments['bond_max'], exclude_ii=True)
            query = system.query(scope['position'], query_args)
            nlist = query.toNeighborList()

            start_points = scope['position'][nlist.query_point_indices]
            bonds = (scope['position'][nlist.point_indices] - start_points)
            bonds = box.wrap(bonds)
            end_points = start_points + bonds

            prim = plato.draw.Lines(
                start_points=start_points, end_points=end_points,
                widths=self.arguments['bond_width'], colors=(.1, .1, .1, 1),
                cap_mode=1)
            scope.setdefault('plato_primitives', []).append(prim)

        scope.setdefault('visuals', []).append(self)

    def draw_matplotlib(self, figure):
        ax = figure.add_subplot(111)
        ax.plot(self.r, self.rdf)
        ax.set_xlabel('r')
        ax.set_ylabel('RDF')
