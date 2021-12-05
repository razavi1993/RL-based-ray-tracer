using FileIO, ImageView, Plots, NPZ

include("vector.jl")
include("camera.jl")
include("hitable.jl")
include("surfaces.jl")
include("qtable.jl")
include("rl.jl")
include("neural.jl")
include("radiance.jl")

# render
function render(world::hitable, cam::camera, q_network, target_network, batch_size, loss, nu, nv, nx=300, ny=300, ns=100, depth=10)
    memory = Memory(120000, 100000, Quadruple[])
    opt = OptimizerParams(0.001, 0.005)
    t_steps = TrainingSteps(1000, 0, zeros(0))
    res = zeros(ny, nx, 3)
    image_history = zeros(ns, ny, nx, 3)
    for s in 1:ns
        for j in 1:ny
            for i in 1:nx
                r = ray(cam, (i+rand())/nx, (j+rand())/ny)
                cl = radiance(r, world, depth, memory, nu, nv, batch_size, loss, q_network, target_network, opt, t_steps)
                res[ny-j+1, nx-i+1, :] += [cl.x, cl.y, cl.z]
            end
        end
        image_history[s, :, :, 1:3] = sqrt.(clamp.(res./s))
    end
    return image_history[ns, :, :, 1:3], image_history, t_steps
end

# scene
function scene()
    surfaces = []
    push!(surfaces, yz_rect(-1.5, 0, 1, -1.5, 1.5, vec(1,0,0), lambertian(vec(0.75,0.25,0.25))))
    push!(surfaces, xz_rect(0, -1.5, 1.5, -1.5, 1.5, vec(0,1,0), lambertian(vec(0.75,0.75,0.75))))
    push!(surfaces, yz_rect(1.5, 0, 1, -1.5, 1.5, vec(-1,0,0), lambertian(vec(0.25,0.25,0.75))))
    push!(surfaces, xz_rect(1, -1.5, 1.5, -1.5, 1.5, vec(0,-1,0), lambertian(vec(0.75,0.75,0.75))))
    push!(surfaces, xy_rect(1.5, -1.5, 1.5, 0, 1, vec(0,0,-1), lambertian(vec(0.75,0.75,0.75))))
    push!(surfaces, xy_rect(-1.5, -1.5, 1.5, 0, 1, vec(0,0,1), lambertian(vec(0.75,0.75,0.75))))
    push!(surfaces, yz_rect(1.499, -1.0, 1.0, 0, 1, vec(1,0,0), light(vec(8,8,8))))
    push!(surfaces, sphere(0.25, vec(0.5,0.251,0.75), lambertian(vec(0.67,0.13,0.25))))
    push!(surfaces, sphere(0.25, vec(-0.5,0.251,0.75), lambertian(vec(0.23,0.65,0.49))))
    push!(surfaces, sphere(0.20, vec(0,0.201,0.2), lambertian(vec(0.57,0.15,0.79))))
    push!(surfaces, sphere(0.20, vec(-0.9,0.201,0.2), lambertian(vec(0.17,0.23,0.45))))
    push!(surfaces, sphere(0.20, vec(0.9,0.201,0.2), lambertian(vec(0.53,0.95,0.49))))
    push!(surfaces, sphere(0.20, vec(0.6,0.799,0.4), lambertian(vec(0.12,0.34,0.67))))
    push!(surfaces, sphere(0.20, vec(-0.6,0.799,0.4), lambertian(vec(0.42,0.54,0.17))))
    push!(surfaces, sphere(0.25, vec(0,0.749,0.1), lambertian(vec(0.13,0.54,0.37))))
    push!(surfaces, sphere(0.15, vec(-0.4,0.151,-0.2), lambertian(vec(0.13,0.54,0.37))))
    push!(surfaces, sphere(0.15, vec(0.4,0.151,-0.2), lambertian(vec(0.13,0.54,0.37))))
    push!(surfaces, sphere(0.15, vec(0.6,0.849,-0.55), lambertian(vec(0.53,0.54,0.27))))
    push!(surfaces, sphere(0.15, vec(-0.6,0.849,-0.55), lambertian(vec(0.23,0.54,0.77))))
    push!(surfaces, sphere(0.15, vec(0.6,0.151,-0.55), lambertian(vec(0.29,0.74,0.57))))
    push!(surfaces, sphere(0.15, vec(-0.6,0.151,-0.55), lambertian(vec(0.63,0.14,0.87))))
    return hit_list(surfaces)
end

nx = 192
ny = 192
ns = 32
world = scene()
cam = camera(vec(0.,0.85,-1.3), vec(0.,0.6,0), vec(0,1,0), 80.0, nx/ny)
img, img_history, t_steps = render(world, cam, q_net, target_net, 8192, loss, 12, 30, nx, ny, ns)
imshow(img)
save("path\\dqn-ray-tracer-cvg\\img.png", img)
loss_values = t_steps.loss_values
loss_values = reshape(mean(reshape(loss_values, 20,50), dims=1), 50)
plot(collect(1:50), loss_values)
savefig("path\\dqn-ray-tracer-cvg\\loss_values.png")

npzwrite("path\\dqn-ray-tracer-cvg\\img_history.npz", img_history)
npzwrite("path\\dqn-ray-tracer-cvg\\loss_values.npz", loss_values)