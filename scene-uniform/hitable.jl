# hitable
abstract type
    hitable
end

# material
abstract type
    material
end

# record
struct record
    t::Float64
    point::vec
    normal::vec
    mater::material
end

# lambertian
struct lambertian <: material
    color::vec
end

# light
struct light <: material
    emit::vec
end
