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
function sample_direction(nl::vec, nu::Int64, nv::Int64, i::Int64, j::Int64)
    u, v, w = local_(nl)
    r1 = rand()
    r2 = rand()
    r = (i - rand())/nu
    s = (j - rand())/nv
    x = cos(2*π*s)*sqrt(1 - r*r)
    y = sin(2*π*s)*sqrt(1 - r*r)
    z = r
    return x*u + y*v + z*w
end

# sample lambertian reflection
function scatter(nl::vec, qvalues::Array{Float64, 2})
    patch_ind, patch_qvalue = sample_patch(qvalues)
    i, j = Tuple(patch_ind)
    nu, nv = size(qvalues)
    ω = sample_direction(nl, nu, nv, i, j)
    pdf = (patch_qvalue*nu*nv)/(2*π)
    return ω, pdf, patch_ind
end

# max qvalue
function update_qvalues(qvalues::Array{Float64, 2})
    max_index = argmax(qvalues)
    nu = size(qvalues)[1]
    max_qvalues = qvalues[max_index]
    i, j = Tuple(max_index)
    cosθ = (i - rand())/nu
    return cosθ*max_qvalues
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
                curr_cell_id = hit_surface.cell_id
                curr_paths = hit_surface.paths
                curr_qvalues = hit_surface.qtable[curr_cell_id]
                brdf = hit_surface.mater.color/(1.0*π)
                ω, pdf, curr_patch_ind  = scatter(n, curr_qvalues)
                if i > 1
                    α = 1/(1+prev_paths[prev_cell_id][prev_patch_ind])
                    prev_qvalues[prev_patch_ind] = (1 .- α)*prev_qvalues[prev_patch_ind] + α*maxe(brdf)*update_qvalues(curr_qvalues)
                end
                cosθ = dot(ω, n)
                curr_paths[curr_cell_id][curr_patch_ind] += 1
                prev_cell_id = curr_cell_id
                prev_paths = curr_paths
                prev_qvalues = curr_qvalues
                prev_patch_ind = curr_patch_ind
                put = put*brdf*cosθ/pdf
                r = ray(hit_surface.point, ω)
            else
                if i > 1
                    α = 1/(1+prev_paths[prev_cell_id][prev_patch_ind])
                    prev_qvalues[prev_patch_ind] = (1 .- α)*prev_qvalues[prev_patch_ind] + α*maxe(hit_surface.mater.emit)
                end
                return put*hit_surface.mater.emit
            end
        else
            if i > 1
                α = 1/(1+prev_paths[prev_cell_id][prev_patch_ind])
                prev_qvalues[prev_patch_ind] = (1 .- α)*prev_qvalues[prev_patch_ind]
            end
            return vec(0,0,0)
        end
    end
    return vec(0,0,0)
end
