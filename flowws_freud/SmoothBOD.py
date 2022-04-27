import collections

import flowws
from flowws import Argument as Arg
import freud
import numpy as np

def fit_D(qs, lmax, seed=13):
    """Fit a Wigner D matrix according to some randomly-generated vectors."""
    import fsph
    import rowan

    qs = np.asarray(qs)
    dim = qs.ndim
    qs = np.atleast_2d(qs)

    lms = fsph.get_LMs(lmax, True)
    filt = lms[:, 0]%2 == 0
    filt[0] = False
    lms = lms[filt]
    N = len(lms)

    rng = np.random.default_rng(seed)
    nvecs = rng.normal(size=(N, 3))
    nvecs /= np.linalg.norm(nvecs, axis=-1, keepdims=True)

    sphs = []
    for rot in [False, True]:
        if rot:
            nvecs = rowan.rotate(qs[:, None, :], nvecs[..., None, :, :])
        phi = np.arccos(nvecs[..., 2])
        theta = np.arctan2(nvecs[..., 1], nvecs[..., 0])
        sphs.append(fsph.pointwise_sph(phi, theta, lmax, True)[..., filt])
    sphs[0] = sphs[0][None]
    D = np.linalg.solve(sphs[0], sphs[1])

    if dim == 1:
        D = D[0]
    return D

OptimizeResult = collections.namedtuple('OptimizeResult', ['op', 'q', 'sphs', 'history_best', 'history_25'])

def optimize_rotation(points, lmax, rotations=32, theta_min=1e-3, theta_max=np.pi/2,
                      max_steps=128, seed=13, patience=64, diffusive=False,
                      initial_rotation=None, normalize_every=32):
    import fsph
    import rowan

    lms = fsph.get_LMs(lmax)
    filt = lms[:, 0]%2 == 0
    filt[0] = False
    lms = lms[filt]

    norms = np.array(points)
    norms /= np.linalg.norm(norms, axis=-1, keepdims=True)
    phi = np.arccos(norms[:, 2])
    theta = np.arctan2(norms[:, 1], norms[:, 0])
    sphs = fsph.pointwise_sph(phi, theta, lmax)[:, filt]
    sphs = np.mean(sphs, axis=0)

    # (rotations, Nsphs)
    sphs = np.tile([sphs], (rotations, 1))
    # (rotations, 4)
    qaccums = np.tile([(1., 0, 0, 0)], (rotations, 1))

    rng = np.random.default_rng(seed)

    axes = rng.normal(size=(rotations, 3))
    angle_scales = np.geomspace(theta_min, theta_max, rotations)
    angles = rng.normal(scale=angle_scales, size=len(angle_scales))
    quats = rowan.from_axis_angle(axes, angles)
    halftheta = np.arccos(quats[:, 0])
    sortidx = np.argsort(np.abs(halftheta))
    quats = quats[sortidx]
    if initial_rotation is not None:
        initial_rotation = np.atleast_2d(initial_rotation)
        quats = rowan.multiply(quats, initial_rotation)
    # (rotations, Nsphs, Nsphs) for sphs@D
    Ds = fit_D(quats, lmax, seed + 1)

    get_op = lambda sph: np.max(sph.real, axis=-1)

    history_best = []
    history_25 = []
    best = OptimizeResult(-np.inf, qaccums[0], sphs[0], history_best, history_25)
    last_improvement = 0

    for step in range(max_steps):
        # rotate
        qaccums = rowan.multiply(quats, qaccums)
        sphs = np.einsum('ab,abc->ac', sphs, Ds, optimize=True)

        if normalize_every and step%normalize_every == 0:
            qaccums /= np.linalg.norm(qaccums, axis=-1, keepdims=True)
        # evaluate
        ops = get_op(sphs)
        # sort
        if diffusive:
            perm = rng.permutation(rotations)
            qaccums = qaccums[perm]
            sphs = sphs[perm]
            ops = ops[perm]
            sortidx = np.argsort(-ops)
        else:
            sortidx = np.argsort(-ops)
            qaccums = qaccums[sortidx]
            sphs = sphs[sortidx]

        best_i = sortidx[0] if diffusive else 0
        best_op = ops[sortidx[0]]
        history_best.append(best_op)
        history_25.append(np.mean(ops[sortidx[:len(sortidx)//4]]))
        if best[0] < best_op:
            best = OptimizeResult(
                best_op, qaccums[best_i], sphs[best_i], history_best, history_25)
            last_improvement = step
        elif step - last_improvement > patience:
            break

    best = best._replace(q=best.q/np.linalg.norm(best.q))
    return best

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
        Arg('on_surface', type=bool, default=True,
            help='Restrict the BOOD to be on the surface of a sphere'),
        Arg('average', type=bool, default=False,
            help='If True, average the BOOD'),
        Arg('average_keys', type=[str],
            help='List of scope keys to generate distinct series when averaging'),
        Arg('auto_rotate', None, bool, default=False,
            help='If True, try to automatically rotate the system to a symmetric orientation'),
        Arg('auto_rotate_lmax', None, int, default=6,
            help='Maximum spherical harmonic degree to detect for auto_rotate'),
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

        self.auto_rotation = getattr(self, 'auto_rotation', None)
        if self.arguments['auto_rotate']:
            kwargs = {}
            if self.auto_rotation is not None:
                kwargs['theta_min'] = 1e-5
                kwargs['theta_max'] = np.pi/16
                kwargs['diffusive'] = True
                kwargs['initial_rotation'] = self.auto_rotation
            self.auto_rotation = optimize_rotation(
                bonds, self.arguments['auto_rotate_lmax'], **kwargs).q
            scope['rotation'] = self.auto_rotation
        else:
            scope.pop('rotation', None)

        scope['SmoothBOD.bonds'] = bonds
        scope.setdefault('visuals', []).append(self)
        scope.setdefault('visual_link_rotation', []).append(self)

        self.gui_actions = [
            ('Auto-orient', self._auto_rotate),
        ]

    def _auto_rotate(self, scope, storage):
        plato_scene = scope['visual_objects'][self]
        bonds = scope['SmoothBOD.bonds']
        quat = optimize_rotation(bonds, self.arguments['auto_rotate_lmax'],
                                 initial_rotation=plato_scene.rotation).q

        plato_scene.rotation = quat
        for v in scope.get('visual_objects', {}).values():
            try:
                v.render()
            except AttributeError:
                pass

    def draw_plato(self):
        import plato, plato.draw as draw
        bonds = np.concatenate(self._data_cache[self._last_data_key], axis=0)
        prim = draw.SpherePoints(points=bonds, on_surface=self.arguments['on_surface'])
        scene = draw.Scene(prim, size=(3, 3), pixel_scale=100,
                           features=dict(additive_rendering=dict(invert=True)))
        if self.auto_rotation is not None:
            scene.rotation = self.auto_rotation
        return scene
