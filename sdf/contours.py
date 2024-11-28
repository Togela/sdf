import numpy as np
import contourpy
import shapely as sh

SAMPLES = 2**18
BATCH_SIZE = 32

def generate(
    sdf,
    step=None,
    bounds=None,
    samples=SAMPLES,
    simplify=None,
    verbose=True,
):

    if bounds is None:
        bounds = sdf.bounds
    x0, x1, y0, y1 = bounds

    if step is None and samples is not None:
        area = (x1 - x0) * (y1 - y0)
        step = (area / samples) ** (1 / 2)

    try:
        dx, dy = step
    except TypeError:
        dx = dy = step

    if verbose:
        print("min %g, %g" % (x0, y0))
        print("max %g, %g" % (x1, y1))
        print("step %g, %g" % (dx, dy))

    # generate points for estimation
    X = np.arange(x0-5*dx, x1+5*dx, dx)
    Y = np.arange(y0-5*dx, y1+5*dy, dy)
    Xs, Ys = np.meshgrid(X, Y)
    P = np.vstack([Xs.flatten(),Ys.flatten()]).T

    # eval sdf at points
    Vflat = sdf(P)
    V = Vflat.reshape(Xs.shape)
    cont_gen = contourpy.contour_generator(Xs, Ys, V)    
    contourlines = cont_gen.lines(level=0)[0] # get lines at level set 0 == contour of sdf

    if simplify is not None:
        cont_sh = sh.linearrings(coords=contourlines)
        cont_sh_simple = cont_sh.simplify(tolerance=simplify)
        contourlines = np.asarray(cont_sh_simple.xy).T

    return contourlines 

if __name__ == "__main__":
    import sdf
    f = sdf.circle(5)

    contour = generate(f)