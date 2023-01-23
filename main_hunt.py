import torch, pdb
import torch.nn as nn
import matplotlib.pyplot as plt


num_particles = 1000
half = num_particles // 2

# Define the number of training iterations
num_iterations = 9000

# Define the number of nearest neighbors to consider
k = 10

# Global variables for positions and velocities
positions = torch.rand(num_particles, 2).cuda() * 100.0
velocities = torch.zeros(num_particles, 2).cuda()

def generate_batch(positions, velocities, k):
    num_particles = positions.shape[0]
    first_half_positions = positions[:half]
    second_half_positions = positions[half:]
    # calculate the distance matrix
    distances = torch.norm(positions[:, None] - positions, dim=2)
    #Find the k nearest neighbors for each particle in the first half
    _, nearest_neighbors = torch.topk(-distances[:,:half], k, dim=1)
    nearest_neighbors = nearest_neighbors.T
    # Find the k nearest neighbors for each particle in the second half
    _, nearest_neighbors_2 = torch.topk(-distances[:, half:], k, dim=1)
    nearest_neighbors_2 = nearest_neighbors_2.T + half
    # Compute the relative positions of the nearest neighbors
    relative_positions = positions[nearest_neighbors] - positions[None, :].repeat(k, 1, 1)
    relative_positions_2 = positions[nearest_neighbors_2] - positions[None, :].repeat(k, 1, 1)
    # Compute the relative positionsy of the goals
    first_half_goals = torch.mean(second_half_positions, dim=0)
    second_half_goals = torch.mean(first_half_positions, dim=0)
    relative_goals = torch.cat([first_half_goals - first_half_positions, second_half_goals - second_half_positions])
    # Concatenate positions, velocities, relative positions, and relative goals
    batch_input = torch.cat([velocities[nearest_neighbors].reshape(num_particles, 2 * k), relative_positions.reshape(num_particles, 2 * k), relative_positions_2.reshape(num_particles, 2 * k), relative_goals], dim=1)
    return batch_input

def loss_function(positions, velocities, accelerations, first_half = True):
    # Check for collisions
    distances = torch.norm(positions[:half, None] - positions[half:], dim=2)
    collision_ratio = 1 - torch.sigmoid(distances-0.1) # changed here
    indices = (collision_ratio > 0.0).nonzero()
    if not first_half: return -collision_ratio.mean(), positions, velocities, indices
    # Compute the mean position of the predators
    predator_mean_position = positions[half:].mean(dim=0)
    # Compute the mean distance of preys to the mean position of the predators
    mean_predator_distance = torch.mean(torch.norm(positions[:half]-predator_mean_position, dim=1))
    return (collision_ratio + mean_predator_distance).mean(), positions, velocities, indices

class AccelerationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, size=256):
        super(AccelerationNetwork, self).__init__()
        def block(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat), nn.Mish()]
        self.model = nn.Sequential(*block(input_size, size), nn.Dropout(0.5),
             *block(size, size//2), nn.Dropout(0.5), *block(size//2, size//4), nn.Dropout(0.5), *block(size//4, output_size))

        self.model[-1] = nn.Tanh()

    def forward(self, x):
        
        return self.model(x)

dt = 0.1

# Define the prey and predator acceleration networks
prey_accel_net = AccelerationNetwork(3 + 6 * k, 10, 2).cuda()
predator_accel_net = AccelerationNetwork(3 + 6 * k, 10, 2).cuda()

# Define a single optimizer for both models
opt_prey = torch.optim.Adam(prey_accel_net.parameters())
opt_pred = torch.optim.Adam(predator_accel_net.parameters())

max_velocities = torch.rand(num_particles).cuda() + 1

for i in range(num_iterations):
    # Generate batch for prey and predator agents
    batch = generate_batch(positions, velocities, k).detach()
    batch = torch.cat([batch, max_velocities.unsqueeze(1)], dim=1)

    # Predict acceleration for prey and predator agents
    prey_accel = prey_accel_net(batch[:half])
    predator_accel = predator_accel_net(batch[half:])
    accelerations = torch.cat([prey_accel, predator_accel], dim=0)
    
    velocities += accelerations * dt

    # Diversify the maximum velocities of them
    #velocities = max_velocities[:, None] * velocities / torch.norm(velocities, dim=1, keepdim=True)

    # Multiply accelerations of the predators by 2
    velocities = velocities / torch.norm(velocities, dim=1, keepdim=True)
    velocities[half:] *= 2

    positions += velocities * dt

    if (i % 2) == 0:
        opt_prey.zero_grad()
        loss, pos, vel, col = loss_function(positions, velocities, prey_accel, True)
        loss.backward()
        opt_prey.step()
        string = 'Prey'
    
    else:
        opt_pred.zero_grad()
        loss, pos, vel, col = loss_function(positions, velocities, predator_accel, False)
        loss.backward()   
        opt_pred.step() 
        string = 'Predator'

    positions = pos.detach()
    velocities = vel.detach()
    
    print("Iteration: %i %s Loss: %f" % (i, string, loss.item()))
    
    diff = positions - velocities
    ind = torch.cat((torch.zeros(num_particles//2),torch.ones(num_particles//2)), dim=0)

    plt.clf()
    plt.quiver(diff[:, 0].cpu(), diff[:, 1].cpu(), velocities[:, 0].cpu(), 
        velocities[:, 1].cpu(), ind.float().cpu().numpy(), cmap ='cool')
    plt.savefig('%i.png' % (i+1000))
