import random
import torch, pdb
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

num_soldiers = 200
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

#positions = torch.rand(num_soldiers, 2).cuda() * 100.0

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


class AttentionLayer(nn.Module):
    def __init__(self, input_size, embed_size):
        super(AttentionLayer, self).__init__()
        # Adjust the size of the query, key, and value layers to match the input size
        self.query = nn.Linear(input_size, embed_size)
        self.key = nn.Linear(input_size, embed_size)
        self.value = nn.Linear(input_size, embed_size)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Compute attention scores
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        # Multiply attention weights with values to get the attention output
        output = torch.matmul(attention_weights, value)
        return output


class ArmyNet(nn.Module):
    def __init__(self, input_size, output_size, size=256):
        super(ArmyNet, self).__init__()
        def block(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat), nn.Mish()]

        # Initialize the attention layer
        self.attention = AttentionLayer(input_size, size)
        
        # Fully connected layers after attention (need to flatten the attention output)
        self.model = nn.Sequential(*block(size, size//2), nn.Dropout(0.5),
                                   *block(size//2, size//4), nn.Dropout(0.5), 
                                   *block(size//4, size//8), nn.Dropout(0.5), 
                                   *block(size//8, output_size))
        self.model[-1] = nn.Tanh()

        # Learnable standard deviation for output noise
        self.std = nn.Parameter(torch.zeros(output_size).cuda())

    def forward(self, x):
        # Apply attention
        x = self.attention(x)
        
        # Flatten the output of attention to match fully connected layer input
        x = x.flatten(start_dim=1)  # Flatten from the second dimension onward
        
        # Pass the result through the fully connected layers
        mu = self.model(x)
        
        # Add noise (standard deviation learned)
        return mu + (self.std * torch.randn_like(mu))

# Define the prey and predator acceleration networks
army_1_net = ArmyNet(373, 4).cuda()
army_2_net = ArmyNet(373, 4).cuda()

# Define the optimizers
optimizer_1 = torch.optim.Adam(army_1_net.parameters())
optimizer_2 = torch.optim.Adam(army_2_net.parameters())

'''
def loss_function(healths):
    # Split healths into two armies
    army_1_healths = healths[:half]
    army_2_healths = healths[half:]

    # Save the initial mean health for both armies
    initial_army_1 = army_1_healths.mean()
    initial_army_2 = army_2_healths.mean()

    print(f"Initial Mean Healths - Army 1: {initial_army_1:.3f}, Army 2: {initial_army_2:.3f}")

    # Compute relative positions
    relative_positions_1 = positions[:half][:, None] - positions[half:]
    relative_positions_2 = positions[half:][:, None] - positions[:half]

    # Compute normalized direction vectors
    direction_vectors_1 = relative_positions_1 / (torch.norm(relative_positions_1, dim=-1, keepdim=True) + 1e-6)
    direction_vectors_2 = relative_positions_2 / (torch.norm(relative_positions_2, dim=-1, keepdim=True) + 1e-6)

    # Compute dot products for visibility (between direction vectors and relative positions)
    dot_product_1 = torch.sum(direction_vectors_1 * directions[half:][:, None], dim=-1)
    dot_product_2 = torch.sum(direction_vectors_2 * directions[:half][:, None], dim=-1)

    # Adjust distances based on dot product and visibility (using sigmoid)
    relative_distance_1 = torch.norm(relative_positions_1, dim=-1) * torch.sigmoid(-dot_product_1)
    relative_distance_2 = torch.norm(relative_positions_2, dim=-1) * torch.sigmoid(-dot_product_2)

    # Max attack range
    max_distance = 10.0  # Example max range for attacks
    smoothing_factor = max_distance / (torch.mean(relative_distance_1+relative_distance_2) / 4)

    # Apply a soft mask to limit interactions beyond the max_distance using sigmoid
    attack_mask_1 = torch.sigmoid((max_distance - relative_distance_1) / smoothing_factor)  # Smoothing factor (10.0)
    attack_mask_2 = torch.sigmoid((max_distance - relative_distance_2) / smoothing_factor)

    # Exclude dead soldiers from being targeted
    relative_distance_1[:, healths[half:] <= 0] = float('inf')  # Army 2 soldiers cannot be targeted
    relative_distance_2[:, healths[:half] <= 0] = float('inf')  # Army 1 soldiers cannot be targeted

    print(f"Relative Distance - Army 1 Mean: {relative_distance_1.mean().item():.3f}, Army 2 Mean: {relative_distance_2.mean().item():.3f}")

    # Find the closest visible opponent for each soldier (non-differentiable)
    _, first_seen_index_1 = torch.min(relative_distance_1, dim=1)  # Soldiers in Army 1 targeting Army 2
    _, first_seen_index_2 = torch.min(relative_distance_2, dim=1)  # Soldiers in Army 2 targeting Army 1

    # Softmax damage calculations for differentiability
    soft_weights_1 = torch.softmax(-relative_distance_1, dim=1)
    soft_weights_2 = torch.softmax(-relative_distance_2, dim=1)

    # Compute weighted damage contributions while ensuring only alive soldiers contribute
    weighted_damage_1 = torch.sum(soft_weights_1 * (healths[half:] > 0).float() * attack_mask_1, dim=1) * 0.02
    weighted_damage_2 = torch.sum(soft_weights_2 * (healths[:half] > 0).float() * attack_mask_2, dim=1) * 0.02

    print(f"Damage Contributions - Army 1 to Army 2: {weighted_damage_1.sum().item():.3f}, "
          f"Army 2 to Army 1: {weighted_damage_2.sum().item():.3f}")

    # Compute health gains based on damage inflicted (ensure dead soldiers don't get health)
    health_gain_1 = 0.25 * weighted_damage_1
    health_gain_2 = 0.25 * weighted_damage_2

    # Mask health gains and updates for dead soldiers
    alive_mask_1 = (healths[:half] > 0).float()  # 1 for alive, 0 for dead
    alive_mask_2 = (healths[half:] > 0).float()  # 1 for alive, 0 for dead

    print(f"Alive Counts - Army 1: {torch.sum(alive_mask_1).item()}, Army 2: {torch.sum(alive_mask_2).item()}")

    # Apply damage and health gains only to living soldiers
    updated_health_1 = alive_mask_1 * (healths[:half] - weighted_damage_2 + health_gain_1)
    updated_health_2 = alive_mask_2 * (healths[half:] - weighted_damage_1 + health_gain_2)

    # Set health of dead soldiers to zero explicitly (avoid invalid health updates)
    updated_health_1 = updated_health_1 * alive_mask_1
    updated_health_2 = updated_health_2 * alive_mask_2

    # Clamp health values to ensure they stay within [0, 1]
    updated_health_1 = updated_health_1.clamp(0, 1)
    updated_health_2 = updated_health_2.clamp(0, 1)

    print(f"Updated Healths - Army 1 Mean: {updated_health_1.mean().item():.3f}, "
          f"Army 2 Mean: {updated_health_2.mean().item():.3f}")

    # Compute the loss based on health changes
    delta_army_1 = updated_health_1.mean() - initial_army_1
    delta_army_2 = updated_health_2.mean() - initial_army_2

    loss = (delta_army_1 - delta_army_2) * (delta_army_1 + delta_army_2)

    print(f"Loss: {loss.item():.3f}")

    # Concatenate updated healths for returning the result
    updated_healths = torch.cat([updated_health_1, updated_health_2])


    threshold = 1e-4
    no_attack_mask_1 = torch.all(soft_weights_1 < threshold, dim=1)  # Soldiers in Army 1 not attacking
    no_attack_mask_2 = torch.all(soft_weights_2 < threshold, dim=1)  # Soldiers in Army 2 not attacking

    # Set `first_seen_index` to -1 for soldiers not attacking
    first_seen_index_1[no_attack_mask_1] = -1
    first_seen_index_2[no_attack_mask_2] = -1

    # Return the loss, updated healths, and the non-differentiable indices
    return loss, updated_healths, first_seen_index_1, first_seen_index_2
'''

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

    # Max attack range
    max_distance = 10.0  # Example max range for attacks
    smoothing_factor = max_distance / (torch.mean(relative_distance_1 + relative_distance_2) / 4)

    # Apply a soft mask to limit interactions beyond the max_distance using sigmoid
    attack_mask_1 = torch.sigmoid((max_distance - relative_distance_1) / smoothing_factor)
    attack_mask_2 = torch.sigmoid((max_distance - relative_distance_2) / smoothing_factor)

    # Exclude dead soldiers from being targeted
    relative_distance_1[:, healths[half:] <= 0] = float('inf')
    relative_distance_2[:, healths[:half] <= 0] = float('inf')

    # Compute softmax weights for damage contribution
    soft_weights_1 = torch.softmax(-relative_distance_1, dim=1)
    soft_weights_2 = torch.softmax(-relative_distance_2, dim=1)

    # Compute damage contributions to each army
    damage_to_army_2 = torch.sum(soft_weights_1.T * (healths[:half] > 0).float() * attack_mask_1.T, dim=1) * 0.02
    damage_to_army_1 = torch.sum(soft_weights_2.T * (healths[half:] > 0).float() * attack_mask_2.T, dim=1) * 0.02

    # Compute health gains
    health_gain_1 = 0.25 * damage_to_army_1
    health_gain_2 = 0.25 * damage_to_army_2

    # Mask for alive soldiers
    alive_mask_1 = (healths[:half] > 0).float()
    alive_mask_2 = (healths[half:] > 0).float()

    # Apply damage and health gains
    updated_health_1 = alive_mask_1 * (healths[:half] - damage_to_army_1 + health_gain_1)
    updated_health_2 = alive_mask_2 * (healths[half:] - damage_to_army_2 + health_gain_2)

    # Clamp health values to [0, 1]
    updated_health_1 = updated_health_1.clamp(0, 1)
    updated_health_2 = updated_health_2.clamp(0, 1)

    # Compute the loss based on health changes
    delta_army_1 = updated_health_1.mean() - initial_army_1
    delta_army_2 = updated_health_2.mean() - initial_army_2

    # Compute the loss (maximize health difference and total health)
    loss = (delta_army_1 - delta_army_2) * (delta_army_1 + delta_army_2)

    # Concatenate updated healths
    updated_healths = torch.cat([updated_health_1, updated_health_2])

    # Handle non-differentiable indices for nearest visible opponents
    threshold = 1e-4
    no_attack_mask_1 = torch.all(soft_weights_1 < threshold, dim=1)  # Soldiers in Army 1 not attacking
    no_attack_mask_2 = torch.all(soft_weights_2 < threshold, dim=1)  # Soldiers in Army 2 not attacking

    # Set `first_seen_index` to -1 for soldiers not attacking
    _, first_seen_index_1 = torch.min(relative_distance_1, dim=1)
    _, first_seen_index_2 = torch.min(relative_distance_2, dim=1)
    first_seen_index_1[no_attack_mask_1] = -1
    first_seen_index_2[no_attack_mask_2] = -1

    # Return the loss, updated healths, and the non-differentiable indices
    return loss, updated_healths, first_seen_index_1, first_seen_index_2

previous_healths = torch.ones(num_soldiers).cuda()

for i in range(num_iterations):
    # Generate batch
    batch_input = generate_batch(positions, velocities, healths, k)

    health_change = healths - previous_healths
    batch_input = torch.cat([batch_input, health_change.view(num_soldiers, 1)], dim=1)
    previous_healths = healths.clone()

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
    velocities = velocities / torch.norm(velocities, dim=1, keepdim=True)


    # Update positions
    #positions += healths[:, None] * velocities


    # Compute proposed positions
    proposed_positions = positions + healths[:, None] * velocities

    # Compute pairwise distances for nearest neighbor detection
    displacement = positions[:, None] - positions  # Shape: (num_soldiers, num_soldiers, 2)
    distances = torch.norm(displacement, dim=2)  # Shape: (num_soldiers, num_soldiers)

    # Mask self-distances
    distances += torch.eye(num_soldiers, device=positions.device) * float('inf')  # Ignore self-distances

    # Find the nearest neighbor for each soldier
    nearest_distances, nearest_indices = torch.min(distances, dim=1)

    # Identify collisions (distance to nearest neighbor is below the threshold)
    min_distance = 25.0  # Set your minimum allowed distance
    colliding_mask = nearest_distances < min_distance

    # Prevent updates for colliding soldiers
    non_colliding_mask = ~colliding_mask
    positions[non_colliding_mask] = proposed_positions[non_colliding_mask]

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

        if (i % 2) == 0:
            # Plot the simulation
            plt.clf()
            plt.title("alive_1 %.0f%% alive_2 %.0f%% health_1 %.0f%%, health_2 %.0f%%" % (army_1_alive*100, army_2_alive*100, healths[:half].mean()*100, healths[half:].mean()*100))

            ind = torch.cat([torch.ones(half) * hel[:half].cpu(), -torch.ones(half) * hel[half:].cpu()])

            x_src = positions[:, 0]
            y_src = positions[:, 1]

            x_dst = torch.cat((positions[half:][:, 0][first_seen_index_1], positions[:half][:, 0][first_seen_index_2]), dim=0)
            y_dst = torch.cat((positions[half:][:, 1][first_seen_index_1], positions[:half][:, 1][first_seen_index_2]), dim=0)


            # Create validity masks for both armies
            valid_mask_1 = first_seen_index_1 >= 0
            valid_mask_2 = first_seen_index_2 >= 0

            # Combine validity masks for both armies
            valid_mask = torch.cat([valid_mask_1, valid_mask_2]).cpu()

            # Compute the vectors between the source and target points
            u = x_dst - x_src
            v = y_dst - y_src 

            # Create the quiver plot

            from matplotlib.colors import Normalize

            # Set a fixed range for the colormap
            color_norm = Normalize(vmin=-1, vmax=1)  # Assuming health values range from -1 to 1

            # Split indices for the two armies
            army_1_indices = torch.arange(half)
            army_2_indices = torch.arange(half, num_soldiers)

            # Plot Army 1 (e.g., circles)
            plt.scatter(positions[army_1_indices, 0].cpu(), positions[army_1_indices, 1].cpu(),
                        c=ind[army_1_indices].float().cpu().numpy(), cmap='seismic', s=50, norm=color_norm, marker='o', label='Army 1')

            # Plot Army 2 (e.g., triangles)
            plt.scatter(positions[army_2_indices, 0].cpu(), positions[army_2_indices, 1].cpu(),
                        c=ind[army_2_indices].float().cpu().numpy(), cmap='seismic', s=50, norm=color_norm, marker='o', label='Army 2')

            # Plot directional vectors (quiver)
            plt.quiver(x_src[valid_mask].cpu(), y_src[valid_mask].cpu(), u[valid_mask].cpu(), v[valid_mask].cpu(), ind[valid_mask].float().cpu().numpy(),
                       angles='xy', scale_units='xy', scale=1, alpha=0.05, cmap='seismic', norm=color_norm)

            '''
            diff = positions - directions
            plt.quiver(diff[:, 0].cpu(), diff[:, 1].cpu(), directions[:, 0].cpu(), 
                directions[:, 1].cpu(), ind.float().cpu().numpy(), cmap ='seismic')
            '''

            
            
            plt.savefig('%i.png' % (i+100000))

        if (torch.sum(healths[:half] > 0) <= 1 or torch.sum(healths[half:] > 0) <= 1 or i % 1000 == 999):

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

            #positions = torch.rand(num_soldiers, 2).cuda() * 100.0

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
            
            
