import torch, pdb, math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12.8,9.6)

num_particles = 1000

# Define the number of training iterations
num_iterations = 9000

# Define the number of nearest neighbors to consider
k = 10

# Generate random angles between 0 and 2*pi
angles = torch.rand(1000) * 2 * math.pi

# Convert angles to cartesian coordinates
x = radius * torch.cos(angles) * 100.0
y = radius * torch.sin(angles) * 100.0

# Combine x and y coordinates into a single tensor
positions = torch.stack((x, y), dim=1).cuda()

initial_positions = positions.clone()
goals = -positions.clone()


velocities = torch.zeros(num_particles, 2).cuda()
#goals = torch.rand(num_particles, 2).cuda() * 100.0

def generate_batch(positions, velocities, k):
    num_particles = positions.shape[0]
    # Find the k nearest neighbors for each particle
    distances = torch.norm(positions[:, None] - positions, dim=2)
    _, nearest_neighbors = torch.topk(-distances, k, dim=1)
    nearest_neighbors = nearest_neighbors.T
    # Compute the relative positions of the nearest neighbors
    relative_positions = positions[nearest_neighbors] - positions[None, :].repeat(k, 1, 1)
    # Compute the relative positionsy of the goals
    relative_goals = goals - positions
    # Concatenate positions, velocities, relative positions, and relative goals
    batch_input = torch.cat([velocities[nearest_neighbors].reshape(num_particles, 2 * k), relative_positions.reshape(num_particles, 2 * k), relative_goals], dim=1)
    return batch_input

def loss_function(positions, velocities, accelerations):
    dt = 0.1

    initial_distance = torch.norm(positions - goals, dim=1)
    velocities += accelerations * dt
    positions += velocities * dt
    after_distance = torch.norm(positions - goals, dim=1)
    distance_loss = (after_distance/initial_distance).mean()

    # Check for collisions
    distances = torch.norm(positions[:, None] - positions, dim=2)
    #distances = torch.triu(distances, diagonal=1)
    collision_ratio = torch.sigmoid(distances-1.0).mean()
    
    # Compute total loss
    total_loss = distance_loss + collision_ratio
    print('%f %f' % (distance_loss, collision_ratio))
    return total_loss, positions, velocities, distances

size = 128

class AccelerationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AccelerationNetwork, self).__init__()
        def block(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat), nn.Mish()]
        self.model = nn.Sequential(*block(input_size, size), nn.Dropout(0.5),
             *block(size, size//2), nn.Dropout(0.5), *block(size//2, output_size))

        self.model[-1] = nn.Tanh()

    def forward(self, x):
        
        return self.model(x)

# Define the network, loss function, and optimizer
net = AccelerationNetwork(input_size=2+4*k, hidden_size=10, output_size=2).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 1e-2)

for i in range(num_iterations):

    # Generate a batch of input for the network
    batch_input = generate_batch(positions, velocities, k)
    # Forward pass
    accelerations = net(batch_input.detach())

    accelerations = accelerations / torch.norm(accelerations, dim=1, keepdim=True)
    # Simulate particle movement
    loss, pos, vel, dis = loss_function(positions, velocities, accelerations)
    # Backward pass and optimizer step
    loss.backward(retain_graph = True)
    if (i % 10) == 0:
        optimizer.step()
        optimizer.zero_grad()
    #print('Iteration:', i, 'Loss:', loss.item())    

    positions = pos.detach()
    velocities = vel.detach()
    distances = dis.detach()

    dd = torch.norm(positions - goals, dim=1)
    positions[dd < 0.1] = initial_positions[dd < 0.1].clone()

    if (i % 500) == 0:
        positions = initial_positions.clone()
    else:
        dd = torch.norm(positions - goals, dim=1)
        positions[dd < 0.1] = initial_positions[dd < 0.1].clone()



    #if i < 1000: continue


    diff = positions - velocities
    distances.fill_diagonal_(1.0)
    ind = (distances < 1.0).any(dim=1)
    print(ind.float().mean())

    plt.clf()
    plt.quiver(diff[:, 0].cpu(), diff[:, 1].cpu(), velocities[:, 0].cpu(), 
        velocities[:, 1].cpu(), ind.float().cpu().numpy(), cmap ='cool')
    plt.savefig('%i.png' % (i+1000))