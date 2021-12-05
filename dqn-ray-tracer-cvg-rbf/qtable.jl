using Distributions

# normalize array
normalize_array(x::Array{Float64, 2}) = x/sum(x)

# sample patch
function sample_patch(qvalues::Array{Float64, 2})
    normalized_qvalues = normalize_array(qvalues)
    normalized_qvalues[end] = 1 - sum(normalized_qvalues[1:end-1])
    flattened_qvalues = normalized_qvalues[:]
    patch_index = rand(Categorical(flattened_qvalues), 1)[1]
    return CartesianIndices(normalized_qvalues)[patch_index], flattened_qvalues[patch_index]
end
