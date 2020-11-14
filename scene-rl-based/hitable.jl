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
    cell_id::Int64
    qtable::Dict{Int64,Array{Float64,2}}
    paths::Dict{Int64,Array{Int64,2}}
end

# lambertian
struct lambertian <: material
    color::vec
end

# light
struct light <: material
    emit::vec
end
