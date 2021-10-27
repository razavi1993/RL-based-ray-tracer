# xy rectangle
struct xy_rect <: hitable
    z::Float64
    x1::Float64
    x2::Float64
    y1::Float64
    y2::Float64
    normal::vec
    mater::material
    m::Int64
    n::Int64
    qtable::Dict{Int64,Array{Float64,2}}
    paths::Dict{Int64,Array{Int64,2}}
end

# yz rectangle
struct yz_rect <: hitable
    x::Float64
    y1::Float64
    y2::Float64
    z1::Float64
    z2::Float64
    normal::vec
    mater::material
    m::Int64
    n::Int64
    qtable::Dict{Int64,Array{Float64,2}}
    paths::Dict{Int64,Array{Int64,2}}
end

# xz rectangle
struct xz_rect <: hitable
    y::Float64
    x1::Float64
    x2::Float64
    z1::Float64
    z2::Float64
    normal::vec
    mater::material
    m::Int64
    n::Int64
    qtable::Dict{Int64,Array{Float64,2}}
    paths::Dict{Int64,Array{Int64,2}}
end

# sphere
struct sphere <: hitable
    radius::Float64
    center::vec
    mater::material
    m::Int64
    n::Int64
    qtable::Dict{Int64,Array{Float64,2}}
    paths::Dict{Int64,Array{Int64,2}}
end

# hitable list
struct hit_list <: hitable
    list::Vector{hitable}
end

# xy rectangle hit function
function hit(rect::xy_rect, r::ray, tmin::Float64, tmax::Float64)
    t = (rect.z - r.o.z)/r.d.z
    if (t < tmin || t > tmax)
        return missing
    end
    p = point(r, t)
    if (p.x < rect.x1 || p.x > rect.x2 || p.y < rect.y1 || p.y > rect.y2)
        return missing
    end
    i = ceil(Int64, rect.m*(p.x - rect.x1)/(rect.x2 - rect.x1))
    j = ceil(Int64, rect.n*(p.y - rect.y1)/(rect.y2 - rect.y1))
    cell_id = rect.n*(i-1) + j
    return record(t, p, rect.normal, rect.mater, cell_id, rect.qtable, rect.paths)
end

# xz rectangle hit function
function hit(rect::xz_rect, r::ray, tmin::Float64, tmax::Float64)
    t = (rect.y - r.o.y)/r.d.y
    if (t < tmin || t > tmax)
        return missing
    end
    p = point(r, t)
    if (p.x < rect.x1 || p.x > rect.x2 || p.z < rect.z1 || p.z > rect.z2)
        return missing
    end
    i = ceil(Int64, rect.m*(p.x - rect.x1)/(rect.x2 - rect.x1))
    j = ceil(Int64, rect.n*(p.z - rect.z1)/(rect.z2 - rect.z1))
    cell_id = rect.n*(i-1) + j
    return record(t, p, rect.normal, rect.mater, cell_id, rect.qtable, rect.paths)
end

# yz rectangle hit function
function hit(rect::yz_rect, r::ray, tmin::Float64, tmax::Float64)
    t = (rect.x - r.o.x)/r.d.x
    if (t < tmin || t > tmax)
        return missing
    end
    p = point(r, t)
    if (p.y < rect.y1 || p.y > rect.y2 || p.z < rect.z1 || p.z > rect.z2)
        return missing
    end
    i = ceil(Int64, rect.m*(p.y - rect.y1)/(rect.y2 - rect.y1))
    j = ceil(Int64, rect.n*(p.z - rect.z1)/(rect.z2 - rect.z1))
    cell_id = rect.n*(i-1) + j
    return record(t, p, rect.normal, rect.mater, cell_id, rect.qtable, rect.paths)
end

# sphere hit function
function hit(s::sphere, r::ray, tmin::Float64, tmax::Float64)
    a = dot(r.d, r.d)
    b = 2.0*dot(r.d, r.o - s.center)
    c = dot(r.o - s.center, r.o - s.center) - s.radius*s.radius
    Δ = b*b - 4.0*a*c
    if Δ <= 0.
        return missing
    end
    t1 = (-b-sqrt(Δ))/(2.0*a)
    t2 = (-b-sqrt(Δ))/(2.0*a)
    if t2 > 0.
        if t1 > 0.
            p = point(r, t1)
            normal = unit(p - s.center)
            u = acos(normal.y)/π
            v = (π + atan(normal.x, normal.z))/(2.0*π)
            i = ceil(Int64, s.m*u)
            j = ceil(Int64, s.n*v)
            cell_id = s.n*(i-1) + j
            return record(t1, p, normal, s.mater, cell_id, s.qtable, s.paths)
        else
            p = point(r, t2)
            normal = unit(s.center - p)
            u = acos(-1.0*normal.y)/π
            v = (π + atan(-1.0*normal.x, -1.0*normal.z))/(2.0*π)
            i = ceil(Int64, s.m*u)
            j = ceil(Int64, s.n*v)
            cell_id = s.n*(i-1) + j
            return record(t2, p, normal, s.mater, cell_id, s.qtable, s.paths)
        end
    end
    return missing
end

# nearest object
function hit(h::hit_list, r::ray, tmin::Float64, tmax::Float64)
    tc = tmax
    hit_surface = missing
    for surface in h.list
        temp_surface = hit(surface, r, tmin, tc)
        if !ismissing(temp_surface)
            hit_surface = temp_surface
            tc = hit_surface.t
        end
    end
    return hit_surface
end
