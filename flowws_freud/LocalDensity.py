import flowws
from flowws import Argument as Arg
import freud

@flowws.add_stage_arguments
class LocalDensity(flowws.Stage):
    """Compute the local density of particles in the system"""
    ARGS = [
        Arg('histogram_bins', '-b', int, default=64,
            help='Number of bins to use in the histogram plot'),
        Arg('r_max', '-r', float, required=True,
            help='Maximum radial distance'),
        Arg('diameter', '-d', float, default=0.,
            help='Smoothing diameter to use in the density calculation'),
    ]

    def run(self, scope, storage):
        """Compute and provide the local density"""
        compute = freud.density.LocalDensity(
            self.arguments['r_max'],
            self.arguments['diameter'])

        box = freud.box.Box.from_box(scope['box'], scope.get('dimensions', 3))
        compute.compute((box, scope['position']), scope['position'])
        self.density = scope['local_density'] = compute.density

        scope.setdefault('color_scalars', []).append('local_density')
        scope.setdefault('visuals', []).append(self)

    def draw_matplotlib(self, figure):
        ax = figure.add_subplot(111)
        ax.hist(self.density, bins=self.arguments['histogram_bins'],
                density=True)
        ax.set_xlabel('Density')
        ax.set_ylabel('Probability')
