import flowws
from flowws import Argument as Arg
import freud

@flowws.add_stage_arguments
class Steinhardt(flowws.Stage):
    """Compute the Steinhardt order parameter of particles in the system"""
    ARGS = [
        Arg('histogram_bins', '-b', int, default=64,
            help='Number of bins to use in the histogram plot'),
        Arg('l', '-l', int, default=6,
            help='Spherical harmonic degree of the order parameter'),
        Arg('r_max', '-r', float,
            help='Maximum radial distance to consider for neighbors (if given)'),
        Arg('num_neighbors', '-n', int,
            help='Number of neighbors to use; overrules r_max if given'),
        Arg('r_guess', None, float, 2,
            help='Characteristic distance for finding num_neighbors neighboring particles'),
    ]

    def run(self, scope, storage):
        """Compute and provide the Steinhardt order parameter"""
        compute = freud.order.Steinhardt(self.arguments['l'])

        box = freud.box.Box.from_box(scope['box'], scope.get('dimensions', 3))
        query_options = dict(exclude_ii=True)
        if 'num_neighbors' in self.arguments:
            query_options['num_neighbors'] = self.arguments['num_neighbors']
            query_options['r_guess'] = self.arguments['r_guess']
        else:
            query_options['r_max'] = self.arguments.get('r_max', None)
        compute.compute((box, scope['position']), query_options)

        self.steinhardt = compute.ql
        name = 'steinhardt_q{}'.format(self.arguments['l'])
        scope[name] = self.steinhardt
        scope.setdefault('color_scalars', []).append(name)
        scope.setdefault('visuals', []).append(self)

    def draw_matplotlib(self, figure):
        ax = figure.add_subplot(111)
        ax.hist(self.steinhardt, bins=self.arguments['histogram_bins'],
                density=True)
        ax.set_xlabel('$Q_{}$'.format(self.arguments['l']))
        ax.set_ylabel('Probability')
