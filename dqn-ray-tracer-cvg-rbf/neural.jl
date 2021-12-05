using Flux
using Statistics

# q-network
q_net = fmap(f64, Chain(Dense(30, 250, relu), Dense(250, 250, relu), Dense(250, 12*30, softplus)))
target_net = fmap(f64, q_net)

# calculate centroids
function create_centroids(kx, ky, kz, δp, pₘᵢₙ)
    inds = Base.vec(collect(Iterators.product(Base.OneTo(kx), Base.OneTo(ky), Base.OneTo(kz))))
    inds_arr = reshape(reinterpret(Int, inds), (3, kx*ky*kz))
    cents = δp.*inds_arr ./ reshape([kx+1, ky+1, kz+1], (3,1)) .+ pₘᵢₙ
    return cents
end

# rbf kernel
function rbf_kernel(cents, x)
    x = repeat(x, size(cents)[2])
    r2 =  reshape(sum(reshape(x .- Base.vec(cents), (3,:)).^2, dims=1), (size(cents)[2],:))
    return exp.(-0.5*r2)
end

# rbf layer
rbf_net = fmap(f64, Chain(Dense(3*6*6, 64, relu), Dense(64, 64, relu), Dense(64, 24)))

# model
function create_model(x, cents, network, rbf_network)
    x_rbf = rbf_network(rbf_kernel(cents, x[1:3,:]))
    return network(vcat(x, x_rbf))
end

centroids = create_centroids(6, 3, 6, reshape([MAX_X - MIN_X, MAX_Y - MIN_Y, MAX_Z - MIN_Z], (3,1)), reshape([MIN_X, MIN_Y, MIN_Z], (3,1)))

model(x) = create_model(x, centroids, q_net, rbf_net)
target_model(x) = create_model(x, centroids, target_net, rbf_net)

q_net_ps = Flux.params(q_net)
target_net_ps = Flux.params(target_net)
rbf_ps = Flux.params(rbf_net)
model_ps = Flux.params(rbf_ps[:], target_net_ps[:], q_net_ps[:])

# soft update
function soft_update(q_network_params, target_network_params)
    for (p1, p2) in zip(q_network_params, target_network_params)
        p2 .= 0.001*p1 .+ 0.999*p2
    end
end

# loss function
function loss(input, indices)
    x, target_values, mse_target_values = input[1:6,:], input[7:7,:], input[8:8,:]
    return mean(abs2, maximum(vcat(model(x)[indices] .- target_values, model(x)[indices] .- mse_target_values), dims=1))
end

# training steps
mutable struct TrainingSteps
    max_steps::Int
    current_step::Int
    loss_values::Array{Float64, 1}
end

# optimizer parameters
mutable struct OptimizerParams
    γ::Float64
    decay::Float64
end

# update optimizer parameters
function update_optimizer_params(opt::OptimizerParams)
    opt.γ *= 1 - opt.decay
end

# train neural network
function train_network(loss, q_network_params, target_network_params, model_params, inputs, indices, memory::Memory, opt::OptimizerParams, t_steps::TrainingSteps, batch_size::Int)
    Flux.train!(loss, model_params, [(inputs, indices)], ADAM(opt.γ))
    update_optimizer_params(opt)
    soft_update(q_network_params, target_network_params)
    t_steps.current_step += 1
    t_steps.loss_values = vcat(t_steps.loss_values, loss(inputs, indices))
end
