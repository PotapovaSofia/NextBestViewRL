import k3d

def illustrate_points(points, plot=None, size=0.1):
    if plot is None:
        plot = k3d.plot(name='points')
    plt_points = k3d.points(positions=points, point_size=size)
    plot += plt_points
    plt_points.shader='3d'
    return plot

def illustrate_mesh(vertices, faces, plot=None):
    if plot is None:
        plot = k3d.plot()
        
    plt_surface = k3d.mesh(vertices, faces,
                           color_map = k3d.colormaps.basic_color_maps.Blues,
                           attribute=vertices[:, 2])

    plot += plt_surface
    return plot

