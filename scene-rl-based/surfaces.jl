# xy rectangle
struct xy_rect <: hitable
    z::Float64
    x1::Float64
    x2::Float64
    y1::Float64
    y2::Float64
    normal::vec
    mater::material
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
end

# sphere
struct sphere <: hitable
    radius::Float64
    center::vec
    mater::material
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
    return record(t, p, rect.normal, rect.mater)
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
    return record(t, p, rect.normal, rect.mater)
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
    return record(t, p, rect.normal, rect.mater)
end

# sphere hit function
function hit(s::sphere, r::ray, tmin::Float64, tmax::Float64)
    a = dot(r.d, r.d)
    b = 2.0*dot(r.d, r.o - s.center)
    c = dot(r.o - s.center, r.o - s.center) - s.radius*s.radius
    Δ = b*b - 4.0*a*c
    if Δ < 0.
        return missing
    else
        t1 = (-b-sqrt(Δ))/(2.0*a)
        t2 = (-b+sqrt(Δ))/(2.0*a)
        if t2 < 0. 
            return missing
        else
            t = (t1 > 0.) ? t1 : t2
            p = point(r, t)
            normal = unit(p - s.center)
            return record(t, p, normal, s.mater)
        end
    end
end

# nearest object
function hit(h::hit_list, r::ray, tmin::Float64, tmax::Float64)
    tc = tmax
    hit_surface = missing
    for surface in h.list
        temp_surface = hit(surface, r, tmin, tc)
        if !ismissing(temp_surface) && tmin < temp_surface.t < tc
            hit_surface = temp_surface
            tc = hit_surface.t
        end
    end
    return hit_surface
end
