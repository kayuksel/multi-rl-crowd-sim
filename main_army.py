import random
import torch, pdb
import torch.nn as nn
import matplotlib.pyplot as plt

num_soldiers = 200
half = num_soldiers // 2

# Define the number of training iterations
num_iterations = 90000

# Define the number of nearest neighbors to consider
k = 5

# Global variables for positions, velocities, and healths
positions = torch.rand(num_soldiers, 2).cuda()
positions[:half] *= -50
positions[half:] *= 50
velocities = torch.zeros(num_soldiers, 2).cuda()
healths = torch.ones(num_soldiers).cuda().requires_grad_()

field_of_view = torch.tensor(2.0).cuda() #radians

def generate_batch(positions, velocities, healths, k):
    first_half_positions = positions[:half]
    second_half_positions = positions[half:]
    # calculate the distance matrix
    distances = torch.norm(positions[:, None] - positions, dim=2)

    # Compute dot product between velocity and position difference
    dot_product = (positions[:, None] - positions) * velocities[None, :]
    dot_product = torch.sum(dot_product, dim=2)

    # Find angle between velocity and position difference
    cos_angle = dot_product / (torch.norm(positions[:, None] - positions, dim=2) * torch.norm(velocities, dim=1))
    #cos_angle = dot_product / (torch.norm(positions[:, None] - positions, dim=2) * torch.norm(velocities[:, None], dim=2))
    angle = torch.acos(cos_angle)

    # Set distances to infinity for particles outside the field of view
    distances[(angle > field_of_view) | (healths <= 0)] += distances.max()

    #Find the k nearest neighbors for each particle in the first half
    _, nearest_neighbors = torch.topk(-distances[:,:half], k, dim=1)
    nearest_neighbors = nearest_neighbors.T
    # Find the k nearest neighbors for each particle in the second half
    _, nearest_neighbors_2 = torch.topk(-distances[:, half:], k, dim=1)
    nearest_neighbors_2 = nearest_neighbors_2.T + half
    # Compute the relative positions of the nearest neighbors
    relative_positions = positions[nearest_neighbors] - positions[None, :].repeat(k, 1, 1)
    relative_positions_2 = positions[nearest_neighbors_2] - positions[None, :].repeat(k, 1, 1)

    mean_first_half_pos = torch.mean(first_half_positions, dim=0) - positions[None, :].repeat(k, 1, 1)
    mean_second_half_pos = torch.mean(second_half_positions, dim=0) - positions[None, :].repeat(k, 1, 1)
    mean_first_half_pos[half:], mean_second_half_pos[half:] = mean_second_half_pos[half:], mean_first_half_pos[half:]

    # Concatenate positions, velocities, relative positions, and relative goals
    batch_input = torch.cat([velocities[nearest_neighbors].reshape(num_soldiers, 2 * k), velocities[nearest_neighbors_2].reshape(num_soldiers, 2 * k), 
        healths[nearest_neighbors].reshape(num_soldiers, k), healths[nearest_neighbors_2].reshape(num_soldiers, k), 
        relative_positions.reshape(num_soldiers, 2 * k), relative_positions_2.reshape(num_soldiers, 2 * k), 
        mean_first_half_pos.reshape(num_soldiers, 2 * k), mean_second_half_pos.reshape(num_soldiers, 2 * k)], dim=1)
    return batch_input

class ArmyNet(nn.Module):
    def __init__(self, input_size, output_size, size=128):
        super(ArmyNet, self).__init__()
        def block(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat), nn.Mish()]
        self.model = nn.Sequential(*block(input_size, size), nn.Dropout(0.5),
             *block(size, size//2), nn.Dropout(0.5), *block(size//2, size//4), nn.Dropout(0.5), *block(size//4, output_size))
        self.model[-1] = nn.Tanh()
        #self.std = nn.Parameter(torch.zeros(output_size).cuda())
    def forward(self, x):
        mu = self.model(x)
        return mu #+ (self.std * torch.randn_like(mu))

# Define the prey and predator acceleration networks
army_1_net = ArmyNet(78, 2).cuda()
army_2_net = ArmyNet(78, 2).cuda()

# Define the optimizers
optimizer_1 = torch.optim.Adam(army_1_net.parameters())
optimizer_2 = torch.optim.Adam(army_2_net.parameters())

def loss_function():
    # Check for collisions
    distances = torch.norm(positions[:half][:, None] - positions[half:], dim=2)
    collisions = torch.lt(distances, 1.0)
    colliding_soldiers = torch.nonzero(collisions)

    army_1_healths = healths[:half]
    army_2_healths = healths[half:]

    # Update healths

    initial_army_1 = army_1_healths.mean()
    initial_army_2 = army_2_healths.mean()

    colliding_mask = torch.zeros(army_1_healths.shape, dtype=torch.bool).cuda()
    colliding_mask[colliding_soldiers[:,0]] = True

    #compute the relative healths of the soldiers
    relative_health_army1 = army_1_healths/ (army_1_healths+army_2_healths)
    relative_health_army2 = army_2_healths/ (army_1_healths+army_2_healths)

    outcome = torch.rand_like(healths)

    army_1_healths_new = torch.where(colliding_mask & (army_2_healths > 0), army_1_healths - relative_health_army2, army_1_healths)
    
    #army_1_healths_new = torch.where(colliding_mask & (army_2_healths > 0), army_1_healths - relative_health_army1*army_2_healths * (torch.randn_like(army_2_healths)-0.5), army_1_healths)

    colliding_mask = torch.zeros(army_2_healths.shape, dtype=torch.bool).cuda()
    colliding_mask[colliding_soldiers[:,1]] = True

    army_2_healths_new = torch.where(colliding_mask & (army_1_healths > 0), army_2_healths - relative_health_army1, army_2_healths)

    # Compute change in healths of both armies
    delta_army_1 = army_1_healths_new.mean() - initial_army_1
    delta_army_2 = army_2_healths_new.mean() - initial_army_2

    # Compute difference in change of healths between the two armies
    delta_diff = delta_army_1.mean() - delta_army_2.mean()

    hel = torch.cat([army_1_healths_new,army_2_healths_new])
    hel = torch.clamp(hel, 0, 1)

    return delta_diff, hel

for i in range(num_iterations):
    # Generate batch
    batch_input = generate_batch(positions, velocities, healths, k)

    # Compute the mean health of each army
    mean_health_army1 = torch.mean(healths[:half])
    mean_health_army2 = torch.mean(healths[half:])

    std_health_army1 = torch.std(healths[:half])
    std_health_army2 = torch.std(healths[half:])

    std_position_army1 = torch.std(positions[:half], dim = 0)
    std_position_army2 = torch.std(positions[half:], dim = 0)

    # Compute the ratio of alive soldiers in each army
    ratio_alive_army1 = torch.sum(healths[:half] > 0) / half
    ratio_alive_army2 = torch.sum(healths[half:] > 0) / half

    # Concatenate the mean health and ratio of alive soldiers for both armies for the first half
    first_half_stats = torch.cat([mean_health_army1.view(1, 1),
        #mean_health_army1.view(1,1), mean_health_army2.view(1,1), 
        std_health_army1.view(1,1), std_health_army2.view(1,1), 
        #ratio_alive_army1.view(1,1), ratio_alive_army2.view(1,1),
        std_position_army1.view(1, 2), std_position_army2.view(1, 2)], dim=1)

    # Concatenate the mean health and ratio of alive soldiers for both armies for the second half in reverse order
    second_half_stats = torch.cat([mean_health_army2.view(1, 1),
        #mean_health_army2.view(1,1), mean_health_army1.view(1,1),
        std_health_army2.view(1,1), std_health_army1.view(1,1), 
        #ratio_alive_army2.view(1,1), ratio_alive_army1.view(1,1),
        std_position_army2.view(1, 2), std_position_army1.view(1, 2)], dim=1)

    # repeat the army statistics for the first half
    first_half_stats = first_half_stats.repeat(half, 1)

    # repeat the army statistics for the second half
    second_half_stats = second_half_stats.repeat(half, 1)

    # concatenate both of those
    army_stats = torch.cat([first_half_stats, second_half_stats], dim= 0)

    batch_input = torch.cat([batch_input, army_stats, healths.view(num_soldiers, 1)], dim=1).detach()

    # Compute accelerations for army 1
    army_1_accelerations = army_1_net(batch_input[:half])
    # Compute accelerations for army 2
    army_2_accelerations = army_2_net(batch_input[half:])
    # Concatenate accelerations
    accelerations = torch.cat([army_1_accelerations, army_2_accelerations])

    # Update velocities
    velocities += accelerations
    # Limit the velocities to the maximum velocities
    velocities = healths[:, None] * velocities / torch.norm(velocities, dim=1, keepdim=True)
    # Update positions
    positions += velocities

    #positions = positions.clamp(-500, 500)

    ui = 50

    current_army = i // ui % 2


    if current_army:
        optimizer_1.zero_grad()
        loss, hel = loss_function()
        loss.backward(retain_graph=True)
        if i % ui == (ui - 1): optimizer_1.step()
    else:
        optimizer_2.zero_grad()
        loss, hel = loss_function()
        loss = loss * -1
        loss.backward(retain_graph=True)
        if i % ui == (ui - 1): optimizer_2.step()
    
    #print(f'Iteration {i}, Loss: {loss}')

    positions = positions.detach()
    velocities = velocities.detach()
    healths = hel.detach().requires_grad_()


    with torch.no_grad():

        army_1_alive = torch.gt(healths[:half], 0).float().mean()
        army_2_alive = torch.gt(healths[half:], 0).float().mean()

        print('army_1_alive %f army_2_alive %f' % (army_1_alive, army_2_alive))

        if (i % 10) == 0:
            diff = positions - velocities
                # Plot the simulation
            ind = torch.cat([torch.ones(half) * hel[:half].cpu(), -torch.ones(half) * hel[half:].cpu()])
            plt.clf()
            plt.title("alive_1 %.0f%% alive_2 %.0f%% health_1 %.0f%%, health_2 %.0f%%" % (army_1_alive*100, army_2_alive*100, healths[:half].mean()*100, healths[half:].mean()*100))
            plt.quiver(diff[:, 0].cpu(), diff[:, 1].cpu(), velocities[:, 0].cpu(), 
                velocities[:, 1].cpu(), ind.float().cpu().numpy(), cmap ='seismic')
            plt.savefig('%i.png' % (i+1000))

        if (i % 1000) == 999:
            # Reset the simulation and update the weights
            positions = torch.rand(num_soldiers, 2).cuda()

            positions[:half] *= -50
            positions[half:] *= 50

            if random.randint(0, 1): positions *= -1

            velocities = torch.zeros(num_soldiers, 2).cuda()
            healths = torch.ones(num_soldiers).cuda().requires_grad_()
            
            if army_1_alive < army_2_alive:
                army_1_net.load_state_dict(army_2_net.state_dict())
            else:
                army_2_net.load_state_dict(army_1_net.state_dict())
            
