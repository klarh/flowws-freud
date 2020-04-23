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
    ]

    def run(self, scope, storage):
        """Compute and provide the RDF"""
        compute = freud.density.RDF(
            self.arguments['bins'], self.arguments['r_max'],
            self.arguments.get('r_min', 0))

        box = freud.box.Box.from_box(scope['box'])
        compute.compute((box, scope['position']))
        self.r, self.rdf = compute.bin_centers, compute.rdf

        scope.setdefault('visuals', []).append(self)

    def draw_matplotlib(self, figure):
        ax = figure.add_subplot(111)
        ax.plot(self.r, self.rdf)
        ax.set_xlabel('r')
        ax.set_ylabel('RDF')
