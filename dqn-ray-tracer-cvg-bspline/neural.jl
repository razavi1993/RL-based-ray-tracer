using Flux
using Statistics

# q-network
q_net = fmap(f64, Chain(Dense(30, 200, relu), Dense(200, 200, relu), Dense(200, 10*20, softplus)))
target_net = fmap(f64, q_net)

# spline encoder
sp_net = fmap(f64, Chain(Dense(3, 64, relu), Dense(64, 64, relu), Dense(64, 2*9*24)))

# calculate spline parameters
function calculate_spline_vars(sp_out, n, k)
    sp_out = reshape(sp_out, (:,n*size(sp_out)[2]))
    ω = softplus.(sp_out[1:k,:])
    δ = softmax(sp_out[k+1:2*k,:], dims=1)
    c = vcat(zeros(1,size(sp_out)[2]), cumsum(δ, dims=1)[1:end-1,:])
    return δ, c, ω
end

# quadratic b-spline
function quadbs(x)
    if all(0.0 .<= x .< 1.0)
        return 0.5.*x.^2
    elseif all(1.0 .<= x .< 2.0)
        return 0.5.*(-2.0.*x.^2 .+ 6.0.*x .- 3.0)
    elseif all(2.0 .<= x .< 3.0)
        return 0.5.*(3.0 .- x).^2
    else
        return 0.0.*x
    end
end

function populate(x, n, k)
    x′ = reshape(x, (1,length(x)))
    return repeat(x′, inner=(k,div(n,3)))
end

function interpolate(x, sp_net, n, k)
    sp_out = sp_net(x)
    δ, c, ω = calculate_spline_vars(sp_out, n, k)
    x′ = populate(x, n, k)
    out = sum(ω.*quadbs.((x′ .- c)./δ), dims=1)
    return reshape(out, (n,:))
end

# model
function create_model(x, network, spline_network, n, k)
    x_sp = interpolate(x[1:3,:], spline_network, n, k)
    return network(vcat(x, x_sp))
end

model(x) = create_model(x, q_net, sp_net, 24, 9)
target_model(x) = create_model(x, target_net, sp_net, 24, 9)

q_net_ps = Flux.params(q_net)
target_net_ps = Flux.params(target_net)
spline_ps = Flux.params(sp_net)
model_ps = Flux.params(spline_ps[:], target_net_ps[:], q_net_ps[:])

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
