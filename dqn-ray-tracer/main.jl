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
function render(world::hitable, cam::camera, q_network, target_network, batch_size, loss, nu, nv, nx=300, ny=300, ns=100, depth=20)
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
    push!(surfaces, yz_rect(0, 0, 1, 0, 1, vec(1,0,0), lambertian(vec(0.75,0.25,0.25))))
    push!(surfaces, xz_rect(0.999, 0.25, 0.75, 0.25, 0.75, vec(0,1,0), light(vec(5,5,5))))
    push!(surfaces, xz_rect(0, 0, 1, 0, 1, vec(0,1,0), lambertian(vec(0.75,0.75,0.75))))
    push!(surfaces, yz_rect(1, 0, 1, 0, 1, vec(-1,0,0), lambertian(vec(0.25,0.25,0.75))))
    push!(surfaces, xz_rect(1, 0, 1, 0, 1, vec(0,-1,0), lambertian(vec(0.75,0.75,0.75))))
    push!(surfaces, xy_rect(1, 0, 1, 0, 1, vec(0,0,-1), lambertian(vec(0.75,0.75,0.75))))
    push!(surfaces, sphere(0.1, vec(0.3,0.101,0.55), lambertian(vec(0.75,0.75,0.75))))
    push!(surfaces, sphere(0.1, vec(0.7,0.101,0.35), lambertian(vec(0.75,0.75,0.75))))
    return hit_list(surfaces)
end

nx = 256
ny = 256
ns = 32
world = scene()
cam = camera(vec(0.5,0.5,-1.3), vec(0.5,0.5,0), vec(0,1,0), 40.0, nx/ny)
img, img_history, t_steps = render(world, cam, q_net, target_net, 8192, loss, 9, 12, nx, ny, ns)
imshow(img)
save("path\\dqn-ray-tracer\\img.png", img)
loss_values = t_steps.loss_values
loss_values = reshape(mean(reshape(loss_values, 20,50), dims=1), 50)
plot(collect(1:50), loss_values)
savefig("path\\dqn-ray-tracer\\loss_values.png")

npzwrite("path\\dqn-ray-tracer\\img_history.npz", img_history)
npzwrite("path\\dqn-ray-tracer\\loss_values.npz", loss_values)