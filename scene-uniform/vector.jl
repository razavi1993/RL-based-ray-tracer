import Base: +, -, /, *

# vec
struct vec
    x::Float64
    y::Float64
    z::Float64
end

# basic operations
+(u::vec, v::vec) = vec(u.x + v.x, u.y + v.y, u.z + v.z)
-(u::vec, v::vec) = vec(u.x - v.x, u.y - v.y, u.z - v.z)
/(u::vec, v::vec) = vec(u.x / v.x, u.y / v.y, u.z / v.z)
/(u::vec, v::Float64) = vec(u.x / v, u.y / v, u.z / v)
*(u::vec, v::vec) = vec(u.x * v.x, u.y * v.y, u.z * v.z)
*(u::Float64, v::vec) = vec(u * v.x, u * v.y, u * v.z)
*(u::vec, v::Float64) = vec(u.x * v, u.y * v, u.z * v)

# clamp
clamp(x::Float64) = x < 0. ? 0. : x < 1. ? x : 1.

# max vector element
maxe(u::vec) = max(u.x, u.y, u.z)
mine(u::vec) = min(u.x, u.y, u.z)

# dot product
dot(u::vec, v::vec) = u.x * v.x + u.y * v.y + u.z * v.z
# norm
norm(u::vec) = sqrt(dot(u, u))
unit(u::vec) = u/norm(u)
# cross product
cross(u::vec, v::vec) = vec(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x)

# ray
struct ray
    o::vec
    d::vec
end

# point
function point(r::ray, t::Float64)
    return r.o + t*r.d
end
