using Distributions

# normalize array
normalize_array(x::Array{Float64, 2}) = x/sum(x)

# create q-table
function create_qtable(num_cells::Int64, nu::Int64, nv::Int64)
    d = Dict{Int64,Array{Float64,2}}()
    for i=1:num_cells
        d[i] = ones(nu,nv)
    end
    return d
end

# create paths
function create_paths(num_cells::Int64, nu::Int64, nv::Int64)
    d = Dict{Int64,Array{Int64,2}}()
    for i=1:num_cells
        d[i] = zeros(Int64, nu,nv)
    end
    return d
end

# sample patch
function sample_patch(qvalues::Array{Float64, 2})
    normalized_qvalues = normalize_array(qvalues)
    flattened_qvalues = normalized_qvalues[:]
    patch_index = rand(Categorical(flattened_qvalues), 1)[1]
    return CartesianIndices(normalized_qvalues)[patch_index], flattened_qvalues[patch_index]
end
