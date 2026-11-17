"""Small, inspectable registration experiments for CAMP 1.

Coordinates are (x, y); NumPy arrays are indexed [y, x]. Synthetic distances
are pixels (unit spacing). Point ICP maps moving -> fixed. Image transforms
phi map fixed -> moving and are used for backward sampling. Medical slices
are bundled locally; no external services are needed during the practical.
"""
from pathlib import Path
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def grid(shape):
    y, x = np.indices(shape, dtype=float)
    return np.stack((x, y), axis=-1)


def rotation(degrees):
    a = np.deg2rad(degrees)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def transform_points(points, R, t):
    return np.asarray(points) @ np.asarray(R).T + np.asarray(t)


def rigid_fit(source, target):
    """Proper least-squares rigid fit with known matches; no scaling."""
    a, b = np.asarray(source), np.asarray(target)
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a-ca).T @ (b-cb))
    D = np.eye(2)
    D[-1, -1] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def nearest_indices(source, target):
    """Each current source point chooses a nearest target point."""
    return cKDTree(target).query(source)[1]


def point_case():
    # An asymmetric closed contour: no outliers or noise in the core example.
    a = np.sort(np.random.default_rng(3).uniform(0, 2*np.pi, 85))
    radius = 18 + 3*np.cos(3*a) + 1.8*np.sin(2*a) + 1.2*np.cos(a)
    fixed = np.column_stack((radius*np.cos(a), .72*radius*np.sin(a)))
    R, t = rotation(28), np.array([6., -4.])
    moving = (fixed-t) @ R
    permutation = np.random.default_rng(7).permutation(len(moving))
    return dict(fixed=fixed, moving=moving[permutation],
                matched_fixed=fixed[permutation], true_R=R, true_t=t)


def run_icp(case, initial_angle=20, initial_translation=(4, -3),
            match_function=nearest_indices, max_iterations=60):
    R, t = rotation(initial_angle), np.array(initial_translation, dtype=float)
    q = transform_points(case['moving'], R, t)
    snapshots, nn_errors, true_errors = [], [], []

    def record():
        idx = nearest_indices(q, case['fixed'])
        snapshots.append(q.copy())
        nn_errors.append(float(np.sqrt(np.mean(np.sum((q-case['fixed'][idx])**2, axis=1)))))
        true_errors.append(float(np.sqrt(np.mean(np.sum((q-case['matched_fixed'])**2, axis=1)))))

    record()
    for _ in range(max_iterations):
        idx = np.asarray(match_function(q, case['fixed']))
        if idx.shape != (len(q),) or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError('Return one integer fixed-point index per moving point.')
        if (idx < 0).any() or (idx >= len(case['fixed'])).any():
            raise ValueError('Correspondence index outside the fixed point set.')
        dR, dt = rigid_fit(q, case['fixed'][idx])
        q = transform_points(q, dR, dt)
        R, t = dR @ R, dR @ t + dt
        record()
        if abs(nn_errors[-2]-nn_errors[-1]) < 1e-9:
            break
    return dict(R=R, t=t, points=q, snapshots=np.array(snapshots),
                nn_error=np.array(nn_errors), true_error=np.array(true_errors))


def plot_icp(case, runs):
    fig, axes = plt.subplots(len(runs), 3, figsize=(12, 3.5*len(runs)), squeeze=False)
    for row, (label, result) in enumerate(runs.items()):
        for col, step in enumerate((0, -1)):
            ax = axes[row, col]
            q = result['snapshots'][step]
            idx = nearest_indices(q, case['fixed'])
            for i in range(0, len(q), 5):
                ax.plot(*np.vstack((q[i], case['fixed'][idx[i]])).T, color='.65', lw=.6)
            ax.scatter(*case['fixed'].T, s=10, label='Fixed', c='#0065BD')
            ax.scatter(*q.T, s=8, label='Moving estimate', c='#D55E00')
            ax.set(title=f'{label}: {"initial" if step == 0 else "final"}',
                   xlabel='x [pixels]', ylabel='y [pixels]', aspect='equal')
            ax.legend(fontsize=8)
        ax = axes[row, 2]
        ax.plot(result['nn_error'], label='Nearest-neighbor RMSE')
        ax.plot(result['true_error'], label='Known-correspondence RMSE')
        ax.set(xlabel='ICP iteration', ylabel='Error [pixels]', title='Two ways to measure error')
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def bilinear(image, xy, gradient=False):
    """Bilinear sampling and its exact piecewise coordinate derivatives.

    All coordinates must be inside the image. No fill pixels enter metrics.
    At grid knots the derivative is the one-sided derivative of the selected
    interpolation cell. The final pixel uses the preceding cell.
    """
    h, w = image.shape
    xy = np.asarray(xy)
    if np.any(xy[..., 0] < 0) or np.any(xy[..., 0] > w-1) or np.any(xy[..., 1] < 0) or np.any(xy[..., 1] > h-1):
        raise ValueError('Sampling outside image support. Reduce the transform or evaluation region.')
    ix = np.minimum(np.floor(xy[..., 0]).astype(int), w-2)
    iy = np.minimum(np.floor(xy[..., 1]).astype(int), h-2)
    ax, ay = xy[..., 0]-ix, xy[..., 1]-iy
    a, b = image[iy, ix], image[iy, ix+1]
    c, d = image[iy+1, ix], image[iy+1, ix+1]
    values = (1-ay)*((1-ax)*a+ax*b) + ay*((1-ax)*c+ax*d)
    if not gradient:
        return values
    dx = (1-ay)*(b-a)+ay*(d-c)
    dy = (1-ax)*(c-a)+ax*(d-b)
    return values, np.stack((dx, dy), axis=-1)


def warp_for_display(image, xy):
    """Display-only warp with NaNs outside support; metrics use bilinear."""
    h, w = image.shape
    valid = ((xy[..., 0] >= 0) & (xy[..., 0] <= w-1) &
             (xy[..., 1] >= 0) & (xy[..., 1] <= h-1))
    out = np.full(xy.shape[:-1], np.nan)
    out[valid] = bilinear(image, xy[valid])
    return out


def rigid_map(xy, parameters, center):
    """Image backward map phi(p)=c+R(theta)(p-c)+(tx,ty)."""
    theta, tx, ty = parameters
    return (xy-center) @ rotation(theta).T + center + np.array([tx, ty])


def phantom(xy, size):
    """Analytic asymmetric image; continuous evaluation avoids a double warp."""
    x, y = xy[..., 0]/size, xy[..., 1]/size
    def blob(cx, cy, sx, sy, amp):
        return amp*np.exp(-.5*(((x-cx)/sx)**2+((y-cy)/sy)**2))
    z = (blob(.40,.39,.11,.15,.85)+blob(.64,.58,.09,.08,.7)
         +blob(.33,.66,.055,.07,.45)-blob(.41,.39,.035,.065,.32)
         +blob(.65,.31,.045,.035,.5))
    return z


def intensity_case(contrast='same'):
    size = 96
    p = grid((size, size)); center = np.array([(size-1)/2]*2)
    truth = np.array([9., 5., -4.])
    # m(q)=f(phi_true^{-1}(q)); m(phi_true(p))=f(p) in the continuous model.
    q_inverse = (p-center-truth[1:]) @ rotation(truth[0]) + center
    fixed = phantom(p, size)
    base = phantom(q_inverse, size)
    if contrast == 'same':
        moving = base
    elif contrast == 'scale_offset':
        moving = 1.7*base + .35
    elif contrast == 'inverted':
        moving = 1.6 - 1.7*base
    else:
        raise ValueError('Choose same, scale_offset, or inverted.')
    # Radius <= 27 and |translations| <= 10 imply valid samples for all angles.
    mask = np.sum((p-center)**2, axis=-1) <= 27**2
    landmarks = size*np.array([[.40,.39],[.64,.58],[.33,.66],[.65,.31],[.47,.57]])
    edges = (np.linspace(-.05, 1.05, 33), np.linspace(-.15, 2.15, 33))
    return dict(fixed=fixed, moving=moving, mask=mask, grid=p, center=center,
                truth=truth, fixed_landmarks=landmarks,
                moving_landmarks=rigid_map(landmarks, truth, center),
                edges=edges, contrast=contrast, bounds=[(-20,20),(-10,10),(-10,10)])


def similarity(x, y, metric, edges=None):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    if metric == 'SSD':
        return float(np.sum((x-y)**2))
    if metric == 'NCC':
        a, b = x-x.mean(), y-y.mean()
        denominator = np.linalg.norm(a)*np.linalg.norm(b)
        if denominator < 1e-14:
            raise ValueError('Signed NCC is undefined for a constant image sample.')
        return float(a@b/denominator)
    if metric == 'MI':
        raise ValueError('Mattes MI requires image geometry; use image_score(case, parameters, "MI").')
    raise ValueError('Metric must be SSD, NCC, or MI.')


def metric_sign(metric):
    return 1 if metric == 'SSD' else -1


def image_overlap(case, parameters):
    """All valid corresponding pixels; no anatomical/background exclusion.

    Medical cases use the whole fixed image. Only out-of-image samples and
    padding introduced when generating the moving image are excluded.
    Synthetic teaching cases retain their independently defined domains.
    """
    selected = case['mask'].copy()
    xy = rigid_map(case['grid'], parameters, case['center'])
    if case.get('full_image_overlap', False):
        h, w = case['moving'].shape
        selected &= ((xy[..., 0] >= 0) & (xy[..., 0] <= w-1) &
                     (xy[..., 1] >= 0) & (xy[..., 1] <= h-1))
        rows, cols = np.nonzero(selected)
        supported = bilinear(case['moving_valid'].astype(float), xy[selected]) >= 1.-1e-9
        selected[rows[~supported], cols[~supported]] = False
    if not np.any(selected):
        raise ValueError('No valid image overlap for these parameters.')
    return selected, xy[selected]


def image_scorer(case, metric):
    """Build an evaluator once per optimization; all transforms use pixel units."""
    if metric != 'MI':
        def score(parameters):
            selected, xy = image_overlap(case, parameters)
            return similarity(case['fixed'][selected], bilinear(case['moving'], xy), metric)
        return score
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError('Mattes MI requires SimpleITK. Rerun the first notebook setup cell or install requirements.txt.') from exc
    fixed = sitk.GetImageFromArray(np.asarray(case['fixed'], dtype=np.float64))
    moving = sitk.GetImageFromArray(np.asarray(case['moving'], dtype=np.float64))
    # Default spacing is one: this teaching API expresses angles in degrees,
    # translations and centers in pixels, including for the 2 mm CT/MR lattice.
    evaluator = sitk.ImageRegistrationMethod()
    evaluator.SetNumberOfThreads(2)
    evaluator.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    evaluator.SetMetricFixedMask(sitk.GetImageFromArray(case['mask'].astype(np.uint8)))
    evaluator.SetInterpolator(sitk.sitkLinear)
    transform = sitk.Euler2DTransform()
    transform.SetCenter(np.asarray(case['center'], dtype=float).tolist())
    def score(parameters):
        transform.SetAngle(float(np.deg2rad(parameters[0])))
        transform.SetTranslation(np.asarray(parameters[1:], dtype=float).tolist())
        evaluator.SetInitialTransform(transform)
        if case.get('full_image_overlap', False):
            selected, _ = image_overlap(case, parameters)
            evaluator.SetMetricFixedMask(sitk.GetImageFromArray(selected.astype(np.uint8)))
        # SimpleITK returns negative MI in nats. Expose positive MI in bits,
        # preserving the exercise's maximize-MI / minimize-negative-MI convention.
        return -float(evaluator.MetricEvaluate(fixed, moving))/np.log(2.)
    return score


def image_score(case, parameters, metric):
    return image_scorer(case, metric)(parameters)


def landmark_error(case, parameters):
    predicted = rigid_map(case['fixed_landmarks'], parameters, case['center'])
    return float(np.mean(np.linalg.norm(predicted-case['moving_landmarks'], axis=1)))


def register_intensity(case, metric, sign_function=metric_sign, initial=(0., 0., 0.),
                       model='rigid', max_iterations=90):
    if model not in ['translation', 'rigid']:
        raise ValueError('Model must be translation or rigid.')
    initial = np.asarray(initial, dtype=float).copy()
    if model == 'translation' and initial[0] != 0:
        raise ValueError('A translation-only model requires initial angle = 0.')
    if any(v < low or v > high for v, (low, high) in zip(initial, case['bounds'])):
        raise ValueError('Initial pose must lie inside the displayed search bounds.')
    start = time.perf_counter()
    history = []
    expand = (lambda p: np.r_[0., p]) if model == 'translation' else (lambda p: p)
    score = image_scorer(case, metric)
    def objective(par):
        return float(sign_function(metric))*score(expand(par))
    # Same optimizer and limits for every metric; MI uses SimpleITK Mattes estimation.
    result = minimize(objective, initial[1:] if model == 'translation' else initial,
                      method='Powell', bounds=case['bounds'][1:] if model == 'translation' else case['bounds'],
                      callback=lambda p: history.append((objective(p), landmark_error(case, expand(p)))),
                      options=dict(maxiter=max_iterations, maxfev=4500, xtol=1e-4, ftol=1e-7))
    parameters = expand(result.x)
    return dict(parameters=parameters, initial=initial, model=model, metric=metric,
                score=score(parameters),
                landmark_error=landmark_error(case, parameters), seconds=time.perf_counter()-start,
                success=bool(result.success), message=str(result.message), history=np.array(history),
                overlap_pixels=int(image_overlap(case, parameters)[0].sum()))


def print_intensity_results(results):
    print(f'{"Pair":<16} {"Metric":<5} {"Score":>11} {"Probe err px":>12} {"theta, tx, ty":>27}  Stop')
    for contrast, runs in results.items():
        for metric, result in runs.items():
            p = ', '.join(f'{v:.2f}' for v in result['parameters'])
            print(f'{contrast:<16} {metric:<5} {result["score"]:>11.4f} {result["landmark_error"]:>12.3f} {p:>27}  {result["message"]}')


def overlay(ax, fixed, warped, display_limits=None):
    # Normalize each image only for display, never for numerical metric values.
    def normalized(a, limits=None):
        low, high = (np.nanmin(a), np.nanmax(a)) if limits is None else limits
        return np.clip((a-low)/(high-low+1e-12), 0, 1)
    limits = (None, None) if display_limits is None else display_limits
    f, m = normalized(fixed, limits[0]), normalized(warped, limits[1])
    rgb = np.stack((m, f, f), axis=-1)
    ax.imshow(np.nan_to_num(rgb), origin='upper')
    ax.set(xlabel='x [pixels]', ylabel='y [pixels]')


def plot_intensity(case, results):
    labels = ['Selected initialization'] + list(results)
    first = next(iter(results.values()))
    parameters = [first.get('initial', np.zeros(3))] + [results[k]['parameters'] for k in results]
    fig, axes = plt.subplots(1, len(labels), figsize=(3.2*len(labels), 3.5))
    for ax, label, par in zip(axes, labels, parameters):
        warped = warp_for_display(case['moving'], rigid_map(case['grid'], par, case['center']))
        overlay(ax, case['fixed'], warped, case.get('display_limits'))
        error_label = case.get('error_label', 'Landmark error')
        ax.set_title(f'{label}\n{error_label}: {landmark_error(case, par):.2f} px', fontsize=9)
    fig.suptitle(f'{case["contrast"]}: cyan=fixed, red=warped (display normalized)', fontsize=10)
    fig.tight_layout()
    return fig


def plot_landscape(case, tx_values=None):
    if tx_values is None:
        tx_values = np.linspace(-9, 9, 91)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    for ax, metric in zip(axes, ['SSD','NCC','MI']):
        values = []
        for tx in tx_values:
            par = case['truth'].copy(); par[1] = tx
            values.append(metric_sign(metric)*image_score(case, par, metric))
        ax.plot(tx_values, values)
        ax.axvline(case['truth'][1], color='black', ls='--', label='Known tx')
        ax.set(xlabel='Backward-map tx [pixels]', ylabel='Objective (minimize)', title=metric)
        ax.legend(fontsize=8)
    fig.suptitle(f'{case["contrast"]}: theta and ty held at the known values', fontsize=10)
    fig.tight_layout()
    return fig


def inspect_sampler(case, tx=0., point=(40, 40)):
    par = np.array([0., tx, 0.]); q = rigid_map(np.array(point), par, case['center'])
    sampled = float(bilinear(case['moving'], q))
    print(f'Output p={point} samples moving q=({q[0]:.2f}, {q[1]:.2f}); value={sampled:.4f}')
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    warped = warp_for_display(case['moving'], rigid_map(case['grid'], par, case['center']))
    overlay(axes[0], case['fixed'], warped); axes[0].plot(*point, 'yo'); axes[0].set_title('Overlay and output pixel')
    axes[1].imshow(case['moving'], cmap='gray'); axes[1].plot(*q, 'ro'); axes[1].set_title('Moving sampling coordinate')
    row = int(point[1]); axes[2].plot(case['fixed'][row], label='Fixed row')
    axes[2].plot(warped[row], label='Warped row'); axes[2].set(xlabel='x [pixels]', ylabel='Intensity'); axes[2].legend()
    fig.tight_layout()
    return fig


def deformable_case(size=56):
    p = grid((size, size)); x, y = p[..., 0], p[..., 1]
    bump = np.exp(-((x-.43*size)**2+(y-.43*size)**2)/(2*(.19*size)**2))
    bump2 = np.exp(-((x-.66*size)**2+(y-.62*size)**2)/(2*(.12*size)**2))
    u = np.stack((1.2 + 2.5*bump-2.0*bump2, -.7-2.3*bump+1.6*bump2), axis=-1)
    moving = phantom(p, size)
    clean = phantom(p+u, size)
    rng = np.random.default_rng(11)
    # Appearance change is not geometric motion; it can lure a flexible warp.
    artifact = .18*np.exp(-((x-.55*size)**2+(y-.42*size)**2)/(2*(.035*size)**2))
    fixed = clean + artifact + rng.normal(0, .009, clean.shape)
    mask = np.zeros_like(fixed, dtype=bool); mask[8:-8, 8:-8] = True
    landmarks = size*np.array([[.35,.34],[.45,.38],[.55,.42],[.62,.53],[.65,.64],
                              [.34,.65],[.48,.57],[.63,.32],[.42,.47],[.53,.67]])
    true_at_landmarks = np.stack([bilinear(u[..., i], landmarks) for i in range(2)], axis=-1)
    return dict(fixed=fixed, moving=moving, clean=clean, artifact=artifact, grid=p,
                truth=u, mask=mask, fixed_landmarks=landmarks,
                moving_landmarks=landmarks+true_at_landmarks)


def regularizer(u):
    """Mean squared first differences, averaged over both axes/components.

    Unit pixel spacing; natural boundary differences (no wraparound).
    Return R and exact dR/du for the implemented discrete energy.
    """
    dx, dy = np.diff(u, axis=1), np.diff(u, axis=0)
    count = dx.size+dy.size
    value = (np.sum(dx*dx)+np.sum(dy*dy))/count
    derivative = np.zeros_like(u)
    derivative[:, :-1] -= 2*dx/count; derivative[:, 1:] += 2*dx/count
    derivative[:-1] -= 2*dy/count; derivative[1:] += 2*dy/count
    return float(value), derivative


def field_energy(case, flat, weight):
    u = flat.reshape(case['truth'].shape)
    mask = case['mask']; xy = (case['grid']+u)[mask]
    warped, spatial_gradient = bilinear(case['moving'], xy, gradient=True)
    residual = warped-case['fixed'][mask]
    mismatch = np.mean(residual**2)
    derivative = np.zeros_like(u)
    derivative[mask] = (2*residual[:, None]*spatial_gradient)/mask.sum()
    reg, reg_gradient = regularizer(u)
    return float(mismatch+weight*reg), (derivative+weight*reg_gradient).ravel()


def jacobian(u):
    """det Dphi for phi(x,y)=(x,y)+u(x,y), unit pixel spacing."""
    ux_y, ux_x = np.gradient(u[..., 0])
    uy_y, uy_x = np.gradient(u[..., 1])
    return (1+ux_x)*(1+uy_y)-ux_y*uy_x


def field_metrics(case, u):
    mask = case['mask']
    values = bilinear(case['moving'], (case['grid']+u)[mask])
    points = case['fixed_landmarks']
    predicted = points+np.stack([bilinear(u[..., i], points) for i in range(2)], axis=-1)
    det = jacobian(u)
    return dict(mse=float(np.mean((values-case['fixed'][mask])**2)),
                regularity=regularizer(u)[0],
                landmark_error=float(np.mean(np.linalg.norm(predicted-case['moving_landmarks'], axis=1))),
                field_error=float(np.mean(np.linalg.norm(u[mask]-case['truth'][mask], axis=1))),
                fold_percent=float(100*np.mean(det[mask] <= 0)), min_jacobian=float(det[mask].min()))


def rigid_for_deformation(case):
    center = np.array([(case['fixed'].shape[0]-1)/2]*2)
    def objective(par):
        xy = rigid_map(case['grid'][case['mask']], par, center)
        return np.mean((bilinear(case['moving'], xy)-case['fixed'][case['mask']])**2)
    result = minimize(objective, [0,0,0], method='Powell', bounds=[(-5,5),(-4,4),(-4,4)])
    return rigid_map(case['grid'], result.x, center)-case['grid']


def control_matrix(shape, nodes):
    """Bilinear control lattice prolongation B; not a cubic B-spline model."""
    if nodes < 2:
        raise ValueError('Use at least two nodes per axis.')
    h, w = shape
    p = grid(shape)*np.array([(nodes-1)/(w-1), (nodes-1)/(h-1)])
    ix = np.minimum(p[..., 0].astype(int), nodes-2)
    iy = np.minimum(p[..., 1].astype(int), nodes-2)
    dx, dy = p[..., 0]-ix, p[..., 1]-iy
    rows = np.arange(h*w)
    columns = [iy*nodes+ix, iy*nodes+ix+1, (iy+1)*nodes+ix, (iy+1)*nodes+ix+1]
    weights = [(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy]
    return coo_matrix((np.concatenate([a.ravel() for a in weights]),
                       (np.tile(rows, 4), np.concatenate([a.ravel() for a in columns]))),
                      shape=(h*w, nodes*nodes)).tocsr()


def optimize_field(case, weight=.01, initial=None, max_iterations=250, control_nodes=None):
    start = time.perf_counter()
    if initial is None:
        initial = rigid_for_deformation(case)
    history = []
    if weight < 0:
        raise ValueError('The regularization weight must be nonnegative.')
    if control_nodes is None:
        start_values = initial.ravel()
        expand = lambda x: x.reshape(initial.shape)
        objective = lambda x: field_energy(case, x, weight)
    else:
        B = control_matrix(initial.shape[:2], control_nodes)
        nodes = grid((control_nodes, control_nodes))*np.array([
            (initial.shape[1]-1)/(control_nodes-1), (initial.shape[0]-1)/(control_nodes-1)])
        start_values = np.stack([bilinear(initial[..., i], nodes) for i in range(2)], axis=-1).ravel()
        expand = lambda x: (B @ x.reshape(-1, 2)).reshape(initial.shape)
        def objective(x):
            value, derivative = field_energy(case, expand(x).ravel(), weight)
            return value, (B.T @ derivative.reshape(-1, 2)).ravel()
    result = minimize(objective, start_values, jac=True,
                      method='L-BFGS-B', bounds=[(-6,6)]*len(start_values),
                      callback=lambda x: history.append(objective(x)[0]),
                      options=dict(maxiter=max_iterations, ftol=1e-12, gtol=1e-9, maxls=35))
    u = expand(result.x)
    return dict(field=u, weight=float(weight), history=np.array(history),
                iterations=int(result.nit), success=bool(result.success), message=str(result.message),
                seconds=time.perf_counter()-start, **field_metrics(case, u))


def folding_field(case):
    """Deliberately constructed diagnostic; never presented as an optimizer output."""
    p = case['grid']; size = case['fixed'].shape[0]
    ux = -7*np.exp(-((p[..., 0]-.5*size)/3)**2)
    return np.stack((ux, np.zeros_like(ux)), axis=-1)


def plot_grid(ax, u, mask, step=4):
    warped = grid(u.shape[:2])+u
    yy, xx = np.where(mask); lo_y, hi_y = yy.min(), yy.max(); lo_x, hi_x = xx.min(), xx.max()
    for y in range(lo_y, hi_y+1, step):
        ax.plot(warped[y, lo_x:hi_x+1, 0], warped[y, lo_x:hi_x+1, 1], color='#0065BD', lw=.6)
    for x in range(lo_x, hi_x+1, step):
        ax.plot(warped[lo_y:hi_y+1, x, 0], warped[lo_y:hi_y+1, x, 1], color='#0065BD', lw=.6)
    ax.set(xlim=(lo_x-6, hi_x+6), ylim=(hi_y+6, lo_y-6), aspect='equal')


def deformation_panels(axes, fixed, moving, warped, field, mask, grid_step=4):
    """Target, before/after images and residuals, overlays and deformation checks."""
    fig = axes[0, 0].figure
    low = float(min(np.nanmin(fixed), np.nanmin(moving)))
    high = float(max(np.nanmax(fixed), np.nanmax(moving)))
    for ax, data, title in zip(axes[0], [fixed, moving, warped],
                               ['Target (fixed)', 'Moving: before registration', 'Registered: after registration']):
        ax.imshow(np.ma.masked_invalid(data), cmap='gray', vmin=low, vmax=high)
        ax.set(title=title, xlabel='x [pixels]', ylabel='y [pixels]', facecolor='.85')
    overlay(axes[1, 0], fixed, moving); axes[1, 0].set_title('Overlay: before')
    overlay(axes[1, 1], fixed, warped); axes[1, 1].set_title('Overlay: after')
    plot_grid(axes[1, 2], field, mask, step=grid_step)
    axes[1, 2].set(title='Predicted sampling grid', xlabel='x [pixels]', ylabel='y [pixels]')
    # Anchor both residual maps to the unregistered pair, so later trials retain
    # the same scale. Values outside the range are marked by colorbar extensions.
    before = moving-fixed
    limit = max(float(np.nanmax(np.abs(before[mask]))), 1e-6)
    for ax, residual, title in zip(axes[2, :2], [before, warped-fixed],
                                   ['Difference to target: before', 'Difference to target: after']):
        im = ax.imshow(np.ma.masked_invalid(np.where(mask, residual, np.nan)),
                       cmap='coolwarm', vmin=-limit, vmax=limit)
        ax.set(title=title, xlabel='x [pixels]', ylabel='y [pixels]', facecolor='.85')
        fig.colorbar(im, ax=ax, shrink=.75, extend='both', label='Image − target [intensity]')
    det = np.where(mask, jacobian(field), np.nan)
    im = axes[2, 2].imshow(np.ma.masked_invalid(det), cmap='coolwarm', vmin=-.5, vmax=2.5)
    axes[2, 2].set(title='Sampled Jacobian det Dφ', xlabel='x [pixels]', ylabel='y [pixels]', facecolor='.85')
    fig.colorbar(im, ax=axes[2, 2], shrink=.75)


def plot_fields(case, fields):
    fig, axes = plt.subplots(3*len(fields), 3, figsize=(12, 10*len(fields)), squeeze=False)
    for i, (label, u) in enumerate(fields.items()):
        metrics = field_metrics(case, u)
        warped = warp_for_display(case['moving'], case['grid']+u)
        block = axes[3*i:3*i+3]
        deformation_panels(block, case['fixed'], case['moving'], warped, u, case['mask'])
        block[0, 1].set_title(f'{label}\nMoving: before registration')
        block[0, 2].set_title(f'Registered: after\nMSE {metrics["mse"]:.5f}; landmark {metrics["landmark_error"]:.2f} px')
        block[2, 2].set_title(f'Sampled Jacobian\n{metrics["fold_percent"]:.2f}% ≤ 0 in ROI')
    fig.suptitle('Overlays: cyan = target, red = moving/registered. Differences use the same signed scale.\nGray = outside evaluation ROI or unavailable samples.', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .97))
    return fig


def print_field_results(case, fields):
    print(f'{"Case":<20} {"MSE":>9} {"R":>9} {"Landmark px":>12} {"Field px":>9} {"Fold %":>8}')
    for label, u in fields.items():
        d = field_metrics(case, u)
        print(f'{label:<20} {d["mse"]:>9.6f} {d["regularity"]:>9.4f} {d["landmark_error"]:>12.3f} {d["field_error"]:>9.3f} {d["fold_percent"]:>8.2f}')


def load_deformation_results():
    with np.load(ROOT/'results'/'deformation_results.npz', allow_pickle=False) as z:
        return {name: z[name].copy() for name in z.files}


def medical_demo():
    """Saved SimpleITK CT demonstration; no SimpleITK needed to display it."""
    with np.load(ROOT/'data'/'lung_demo.npz', allow_pickle=False) as z:
        d = {key: z[key].copy() for key in z.files}
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0,0].imshow(d['fixed'], cmap='gray', vmin=-1000, vmax=200)
    axes[0,0].set_title('Target (fixed CT)')
    overlay(axes[0,1], d['fixed'], d['initial']); axes[0,1].set_title('Overlay: before registration')
    overlay(axes[0,2], d['fixed'], d['registered']); axes[0,2].set_title('Overlay: rigid + B-spline')
    axes[1,0].imshow(d['registered'], cmap='gray', vmin=-1000, vmax=200)
    axes[1,0].set_title('Registered CT')
    for ax, key, title in [(axes[1,1], 'initial', 'Difference to target: before'),
                            (axes[1,2], 'registered', 'Difference to target: after')]:
        im = ax.imshow(d[key]-d['fixed'], cmap='coolwarm', vmin=-300, vmax=300)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.75, extend='both', label='Image − target [HU]')
    fig.suptitle('Lung CT: qualitative demonstration; residual maps share a ±300 HU scale', fontsize=11)
    fig.tight_layout()
    metadata = json.loads((ROOT/'data'/'lung_metadata.json').read_text())
    print(json.dumps(metadata, indent=2))
    return fig


def medical_intensity_case(pair='ct_ct'):
    """Real anatomy, imposed 2D motion; CT-MR recovery is not anatomical truth."""
    if pair not in ['ct_ct', 'ct_mr']:
        raise ValueError('Choose ct_ct or ct_mr.')
    with np.load(ROOT/'data'/'abdominal_ct_mr.npz', allow_pickle=False) as data:
        ct, mr = data['ct'].astype(float), data['mr'].astype(float)
    # Registration retains the source intensity units. No clipping, range
    # rescaling, histogram matching or cross-modality intensity calibration.
    fixed = gaussian_filter(ct, .65)
    base = fixed if pair == 'ct_ct' else gaussian_filter(mr, .65)
    # Windowing is strictly for plotting and never enters the metric arrays.
    ct_display = (-150., 250.)
    mr_display = (0., float(np.percentile(mr[mr > 0], 99.5)))
    display_limits = (ct_display, ct_display if pair == 'ct_ct' else mr_display)
    p = grid(fixed.shape)
    center = np.array([(fixed.shape[1]-1)/2, (fixed.shape[0]-1)/2])
    truth = np.array([9., 5., -4.])
    inverse = (p-center-truth[1:]) @ rotation(truth[0]) + center
    generated = warp_for_display(base, inverse)
    valid = np.isfinite(generated)
    moving = np.nan_to_num(generated)
    mask = np.ones(fixed.shape, dtype=bool)  # Entire image; no circular ROI.
    # Virtual evaluation probes; these are not annotated anatomical landmarks.
    probes = center + np.array([[-30,-20],[30,-20],[-30,20],[30,20],[0,0]])
    return dict(fixed=fixed, moving=moving, reference_moving=base,
                moving_valid=valid, mask=mask, full_image_overlap=True, grid=p, center=center,
                truth=truth, fixed_landmarks=probes,
                moving_landmarks=rigid_map(probes, truth, center),
                display_limits=display_limits,
                intensity_processing='Source units, Gaussian sigma 0.65 pixels; no intensity rescaling or clipping for registration.',
                contrast='CT–CT' if pair == 'ct_ct' else 'CT–MR', pair=pair,
                error_label='Imposed-motion error', spacing_mm=2.,
                bounds=[(-20,20),(-10,10),(-10,10)])


def plot_image_inputs(case):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, key, title, limits in zip(axes[:2], ['fixed','moving'], ['Fixed CT','Moving '+('CT' if case['pair']=='ct_ct' else 'MR')], case['display_limits']):
        ax.imshow(case[key], cmap='gray', vmin=limits[0], vmax=limits[1])
        ax.set(title=title, xlabel='x [pixels]', ylabel='y [pixels]')
    overlay(axes[2], case['fixed'], case['moving'], case.get('display_limits'))
    axes[2].set_title('Before registration', fontsize=10)
    fig.tight_layout()
    return fig


def plot_manual_pose(case, parameters):
    warped = warp_for_display(case['moving'], rigid_map(case['grid'], parameters, case['center']))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    overlay(axes[0], case['fixed'], case['moving'], case.get('display_limits')); axes[0].set_title('Before')
    overlay(axes[1], case['fixed'], warped, case.get('display_limits')); axes[1].set_title(f'Your backward map: {tuple(parameters)}')
    fig.tight_layout()
    return fig


def plot_selected_landscape(case, result, metric, sign_function=metric_sign):
    values = np.linspace(-9, 9, 73)
    scores = []
    for tx in values:
        par = result['parameters'].copy(); par[1] = tx
        scores.append(sign_function(metric)*image_score(case, par, metric))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(values, scores)
    ax.axvline(result['parameters'][1], color='#D55E00', ls='--', label='Your final tx')
    ax.set(xlabel='Backward-map tx [pixels]', ylabel='Objective to minimize',
           title=f'{case["contrast"]}: {metric}; angle and ty fixed at your result')
    ax.legend(); fig.tight_layout()
    return fig
