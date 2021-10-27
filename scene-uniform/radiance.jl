# local coordinate
function local_(n::vec)
    w = unit(n)
    if abs(w.x) > 0.9
        a = vec(0, 1, 0)
    else
        a = vec(1, 0, 0)
    end
    v = unit(cross(w, a))
    u = cross(w, v)
    return u, v, w
end

# uniform sampling
function sample_direction(nl::vec)
    u, v, w = local_(nl)
    r = rand()
    s = rand()
    x = cos(2*π*s)*sqrt(1 - r*r)
    y = sin(2*π*s)*sqrt(1 - r*r)
    z = r
    return x*u + y*v + z*w
end

# sample lambertian reflection
function scatter(nl::vec)
    ω = sample_direction(nl)
    pdf = 1/(2*π)
    return ω, pdf
end

# radiance function
function radiance(r::ray, world::hitable, depth::Int)
    put = vec(1,1,1)
    local prev_paths, prev_cell_id, prev_patch_ind, prev_qvalues
    for i=1:depth
        hit_surface = hit(world, r, 0.0001, typemax(Float64))
        if !ismissing(hit_surface)
            if typeof(hit_surface.mater) == lambertian
                n = hit_surface.normal
                brdf = hit_surface.mater.color/(1.0*π)
                ω, pdf  = scatter(n)
                cosθ = dot(ω, n)
                put = put*brdf*cosθ/pdf
                r = ray(hit_surface.point, ω)
            else
                return put*hit_surface.mater.emit
            end
        else
            return vec(0,0,0)
        end
    end
    return vec(0,0,0)
end
