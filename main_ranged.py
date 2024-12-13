import random
import torch, pdb
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

num_soldiers = 100
half = num_soldiers // 2
radius = 50.0

# Define the number of training iterations
num_iterations = 90000

# Define the number of nearest neighbors to consider
k = 20

# Global variables for positions, velocities, and healths
angles = torch.linspace(0, 2 * np.pi * (num_soldiers - 1) / num_soldiers, num_soldiers).cuda()

# Generate positions for the first half of the army (Army 1)
positions = torch.zeros(num_soldiers, 2).cuda()
positions[:half, 0] = radius * torch.cos(angles[:half])  # x-coordinates
positions[:half, 1] = radius * torch.sin(angles[:half])  # y-coordinates

# Generate positions for the second half of the army (Army 2)
positions[half:, 0] = radius * torch.cos(angles[half:])  # x-coordinates
positions[half:, 1] = radius * torch.sin(angles[half:])  # y-coordinates

velocities = torch.zeros(num_soldiers, 2).cuda()
directions = torch.zeros(num_soldiers, 2).cuda()
healths = torch.ones(num_soldiers).cuda().requires_grad_()

field_of_view = torch.tensor(2.0).cuda() #radians

def generate_batch(positions, velocities, healths, k):
    first_half_positions = positions[:half]
    second_half_positions = positions[half:]

    # Calculate the distance matrix
    distances = torch.norm(positions[:, None] - positions, dim=2)

    # Compute dot product between velocity and position difference
    dot_product = (positions[:, None] - positions) * velocities[None, :]
    dot_product = torch.sum(dot_product, dim=2)

    # Find angle between velocity and position difference
    cos_angle = dot_product / (torch.norm(positions[:, None] - positions, dim=2) * torch.norm(velocities, dim=1))
    angle = torch.acos(cos_angle)

    # Set distances to infinity for soldiers outside the field of view or dead soldiers
    distances[(angle > field_of_view) | (healths[:, None] <= 0) | (healths <= 0)] = float('inf')

    # Find the k nearest neighbors for each particle in the first half
    _, nearest_neighbors = torch.topk(-distances[:, :half], k, dim=1)
    nearest_neighbors = nearest_neighbors.T

    # Find the k nearest neighbors for each particle in the second half
    _, nearest_neighbors_2 = torch.topk(-distances[:, half:], k, dim=1)
    nearest_neighbors_2 = nearest_neighbors_2.T + half

    # Compute the relative positions of the nearest neighbors
    relative_positions = positions[nearest_neighbors] - positions[None, :].repeat(k, 1, 1)
    relative_positions_2 = positions[nearest_neighbors_2] - positions[None, :].repeat(k, 1, 1)

    # Compute mean positions for team dynamics, excluding dead soldiers
    mean_first_half_pos = torch.mean(first_half_positions[healths[:half] > 0], dim=0) - positions[None, :].repeat(k, 1, 1)
    mean_second_half_pos = torch.mean(second_half_positions[healths[half:] > 0], dim=0) - positions[None, :].repeat(k, 1, 1)
    mean_first_half_pos[half:], mean_second_half_pos[half:] = mean_second_half_pos[half:], mean_first_half_pos[half:]

    # Concatenate positions, velocities, relative positions, and relative goals
    batch_input = torch.cat([
        velocities[nearest_neighbors].reshape(num_soldiers, 2 * k),
        velocities[nearest_neighbors_2].reshape(num_soldiers, 2 * k),
        healths[nearest_neighbors].reshape(num_soldiers, k),
        healths[nearest_neighbors_2].reshape(num_soldiers, k),
        relative_positions.reshape(num_soldiers, 2 * k),
        relative_positions_2.reshape(num_soldiers, 2 * k),
        mean_first_half_pos.reshape(num_soldiers, 2 * k),
        mean_second_half_pos.reshape(num_soldiers, 2 * k),
        directions[nearest_neighbors].reshape(num_soldiers, 2 * k),
        directions[nearest_neighbors_2].reshape(num_soldiers, 2 * k)
    ], dim=1)

    return batch_input

class ArmyNet(nn.Module):
    def __init__(self, input_size, output_size, size=128):
        super(ArmyNet, self).__init__()
        def block(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat), nn.Mish()]
        self.model = nn.Sequential(*block(input_size, size), nn.Dropout(0.5),
             *block(size, size//2), nn.Dropout(0.5), *block(size//2, size//4), nn.Dropout(0.5), *block(size//4, output_size))
        self.model[-1] = nn.Tanh()
        self.std = nn.Parameter(torch.zeros(output_size).cuda())
    def forward(self, x):
        mu = self.model(x)
        return mu + (self.std * torch.randn_like(mu))

# Define the prey and predator acceleration networks
army_1_net = ArmyNet(372, 4).cuda()
army_2_net = ArmyNet(372, 4).cuda()

# Define the optimizers
optimizer_1 = torch.optim.Adam(army_1_net.parameters())
optimizer_2 = torch.optim.Adam(army_2_net.parameters())

def loss_function(healths):
    # Split healths into two armies
    army_1_healths = healths[:half]
    army_2_healths = healths[half:]

    # Save the initial mean health for both armies
    initial_army_1 = army_1_healths.mean()
    initial_army_2 = army_2_healths.mean()

    # Compute relative positions
    relative_positions_1 = positions[:half][:, None] - positions[half:]
    relative_positions_2 = positions[half:][:, None] - positions[:half]

    # Compute normalized direction vectors
    direction_vectors_1 = relative_positions_1 / (torch.norm(relative_positions_1, dim=-1, keepdim=True) + 1e-6)
    direction_vectors_2 = relative_positions_2 / (torch.norm(relative_positions_2, dim=-1, keepdim=True) + 1e-6)

    # Compute dot products for visibility
    dot_product_1 = torch.sum(direction_vectors_1 * directions[half:][:, None], dim=-1)
    dot_product_2 = torch.sum(direction_vectors_2 * directions[:half][:, None], dim=-1)

    # Adjust distances based on dot product and visibility
    relative_distance_1 = torch.norm(relative_positions_1, dim=-1) * torch.sigmoid(-dot_product_1)
    relative_distance_2 = torch.norm(relative_positions_2, dim=-1) * torch.sigmoid(-dot_product_2)

    # Exclude dead soldiers from being targeted
    relative_distance_1[:, healths[half:] <= 0] = float('inf')  # Army 2 soldiers cannot be targeted
    relative_distance_2[:, healths[:half] <= 0] = float('inf')  # Army 1 soldiers cannot be targeted

    # Find the closest visible opponent for each soldier
    _, first_seen_index_1 = torch.min(relative_distance_1, dim=1)  # Soldiers in Army 1 targeting Army 2
    _, first_seen_index_2 = torch.min(relative_distance_2, dim=1)  # Soldiers in Army 2 targeting Army 1

    # Revalidate targets: Exclude invalid targets for both armies
    valid_targets_mask_1 = (healths[half:][first_seen_index_1] > 0) & (first_seen_index_1 != -1)
    valid_targets_mask_2 = (healths[:half][first_seen_index_2] > 0) & (first_seen_index_2 != -1)

    # Filter out invalid indices
    first_seen_index_1 = torch.where(valid_targets_mask_1, first_seen_index_1, -1)
    first_seen_index_2 = torch.where(valid_targets_mask_2, first_seen_index_2, -1)

    # Compute damage contributions for each valid target
    damage_to_army_2 = torch.zeros(half).cuda()
    damage_to_army_1 = torch.zeros(half).cuda()

    # Use scatter_add_ to accumulate damage from attackers
    damage_to_army_2.scatter_add_(
        0, 
        first_seen_index_1[first_seen_index_1 != -1], 
        healths[:half][first_seen_index_1 != -1] * 0.01
    )
    damage_to_army_1.scatter_add_(
        0, 
        first_seen_index_2[first_seen_index_2 != -1], 
        healths[half:][first_seen_index_2 != -1] * 0.01
    )

    # Apply health reduction
    army_1_healths_new = torch.where(healths[:half] > 0, army_1_healths - damage_to_army_1, army_1_healths)
    army_2_healths_new = torch.where(healths[half:] > 0, army_2_healths - damage_to_army_2, army_2_healths)

    # Clamp health values between 0 and 1
    army_1_healths_new = army_1_healths_new.clamp(0, 1)
    army_2_healths_new = army_2_healths_new.clamp(0, 1)

    # Debugging: Check health updates
    print(f"Army 1 Healths: Initial Mean = {initial_army_1:.3f}, Updated Mean = {army_1_healths_new.mean():.3f}")
    print(f"Army 2 Healths: Initial Mean = {initial_army_2:.3f}, Updated Mean = {army_2_healths_new.mean():.3f}")
    print(f"Damage to Army 1: {damage_to_army_1.sum().item()}, Damage to Army 2: {damage_to_army_2.sum().item()}")

    # Compute the changes in health
    delta_army_1 = army_1_healths_new.mean() - initial_army_1
    delta_army_2 = army_2_healths_new.mean() - initial_army_2

    # Compute the difference and sum of health changes
    delta_diff = delta_army_1 - delta_army_2
    delta_sum = delta_army_1 + delta_army_2

    # Concatenate updated healths
    updated_healths = torch.cat([army_1_healths_new, army_2_healths_new])

    return delta_diff * delta_sum, updated_healths, first_seen_index_1, first_seen_index_2

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
        mean_health_army1.view(1,1), mean_health_army2.view(1,1), 
        std_health_army1.view(1,1), std_health_army2.view(1,1), 
        ratio_alive_army1.view(1,1), ratio_alive_army2.view(1,1),
        std_position_army1.view(1, 2), std_position_army2.view(1, 2)], dim=1)

    # Concatenate the mean health and ratio of alive soldiers for both armies for the second half in reverse order
    second_half_stats = torch.cat([mean_health_army2.view(1, 1),
        mean_health_army2.view(1,1), mean_health_army1.view(1,1),
        std_health_army2.view(1,1), std_health_army1.view(1,1), 
        ratio_alive_army2.view(1,1), ratio_alive_army1.view(1,1),
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

    directions += accelerations[:, 2:] / torch.norm(accelerations[:,2:], dim=1, keepdim=True)
    directions = directions / torch.norm(directions, dim=1, keepdim=True)

    # Update velocities
    velocities += accelerations[:,:2] / torch.norm(accelerations[:,:2], dim=1, keepdim=True)
    # Limit the velocities to the maximum velocities
    velocities = healths[:, None] * velocities / torch.norm(velocities, dim=1, keepdim=True)
    # Update positions
    positions += velocities

    #positions = positions.clamp(-500, 500)

    ui = 50

    current_army = i // ui % 2


    if current_army:
        optimizer_1.zero_grad()
        loss, hel, first_seen_index_1, first_seen_index_2 = loss_function(healths)
        loss.backward(retain_graph=True)
        if i % ui == (ui - 1): optimizer_1.step()
    else:
        optimizer_2.zero_grad()
        loss, hel, first_seen_index_1, first_seen_index_2 = loss_function(healths)
        loss = loss * -1
        loss.backward(retain_graph=True)
        if i % ui == (ui - 1): optimizer_2.step()
    
    #print(f'Iteration {i}, Loss: {loss}')

    positions = positions.detach()
    velocities = velocities.detach()
    directions = directions.detach()
    healths = hel.detach().requires_grad_()


    with torch.no_grad():

        army_1_alive = torch.gt(healths[:half], 0).float().mean()
        army_2_alive = torch.gt(healths[half:], 0).float().mean()

        print('army_1_alive %f army_2_alive %f' % (army_1_alive, army_2_alive))

        if (i % 5) == 0:
            # Plot the simulation
            plt.clf()
            plt.title("alive_1 %.0f%% alive_2 %.0f%% health_1 %.0f%%, health_2 %.0f%%" % (army_1_alive*100, army_2_alive*100, healths[:half].mean()*100, healths[half:].mean()*100))

            ind = torch.cat([torch.ones(half) * hel[:half].cpu(), -torch.ones(half) * hel[half:].cpu()])

            x_src = positions[:, 0]
            y_src = positions[:, 1]

            x_dst = torch.cat((positions[half:][:, 0][first_seen_index_1], positions[:half][:, 0][first_seen_index_2]), dim=0)
            y_dst = torch.cat((positions[half:][:, 1][first_seen_index_1], positions[:half][:, 1][first_seen_index_2]), dim=0)


                  # Compute the vectors between the source and target points
            u = x_dst - x_src
            v = y_dst - y_src 


            # Create the quiver plot

            from matplotlib.colors import Normalize

            # Set a fixed range for the colormap
            color_norm = Normalize(vmin=-1, vmax=1)  # Assuming health values range from -1 to 1

            # Plotting without normalization
            plt.scatter(positions[:, 0].cpu(), positions[:, 1].cpu(),
                        c=ind.float().cpu().numpy(), cmap='seismic', s=50, norm=color_norm)       
            plt.quiver(x_src.cpu(), y_src.cpu(), u.cpu(), v.cpu(), ind.float().cpu().numpy(), angles='xy', scale_units='xy', scale=1, alpha = 0.05, cmap ='seismic', norm=color_norm)


            '''
            diff = positions - directions
            plt.quiver(diff[:, 0].cpu(), diff[:, 1].cpu(), directions[:, 0].cpu(), 
                directions[:, 1].cpu(), ind.float().cpu().numpy(), cmap ='seismic')
            '''

            
            
            plt.savefig('%i.png' % (i+1000))

        if (torch.sum(healths[:half] > 0) == 0 or torch.sum(healths[half:] > 0) == 0 or i % 1000 == 999):

        #if (loss == 0.0):
            # Reset the simulation and update the weights
            rotation_angle = random.uniform(0, 2 * np.pi)

            # Generate angles for the circle and apply the rotation
            angles = torch.linspace(0, 2 * np.pi * (num_soldiers - 1) / num_soldiers, num_soldiers).cuda()
            angles = (angles + rotation_angle) % (2 * np.pi)

            # Generate positions for the first half of the army (Army 1)
            positions = torch.zeros(num_soldiers, 2).cuda()
            positions[:half, 0] = radius * torch.cos(angles[:half])  # x-coordinates
            positions[:half, 1] = radius * torch.sin(angles[:half])  # y-coordinates

            # Generate positions for the second half of the army (Army 2)
            positions[half:, 0] = radius * torch.cos(angles[half:])  # x-coordinates
            positions[half:, 1] = radius * torch.sin(angles[half:])  # y-coordinates

            #if random.randint(0, 1): positions *= -1

            velocities = torch.zeros(num_soldiers, 2).cuda()
            directions = torch.zeros(num_soldiers, 2).cuda()
            healths = torch.ones(num_soldiers).cuda().requires_grad_()
            
            if army_1_alive < army_2_alive:
                print("Army 2 wins. Transferring weights to Army 1.")
                army_1_net.load_state_dict(army_2_net.state_dict())
            elif army_2_alive < army_1_alive:
                print("Army 1 wins. Transferring weights to Army 2.")
                army_2_net.load_state_dict(army_1_net.state_dict())
            else:
                print("It's a tie. No weight transfer.")
