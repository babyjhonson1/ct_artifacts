from filterd_back_projection import filterd_back_projection

def ReverseTask(p, N_xy, xy_max, Method_name, xi0):
    if "__filterd_back_projection__" in Method_name:
        mu = filterd_back_projection(p, N_xy, xy_max, xi0)

    return mu