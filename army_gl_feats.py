import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
# Note: We no longer use torchvision.models since we now use a simple feed-forward ArmyNet.

# Simulation parameters
num_soldiers = 200
half = num_soldiers // 2
num_iterations = 90000
ui = 50  # update interval for optimizer steps
window_width, window_height = 1024, 1024
image_size = 128       # used for capturing the rendered image (if needed)
local_view_size = 32   # not used in this version (left for compatibility)
k = 5  # number of nearest neighbors; this yields 14*k + 8 = 78 features

# Field of view parameter (radians) used in generating the aux features.
field_of_view = np.pi / 4

# Global simulation state variables
positions = None
velocities = None
healths = None

# Global variable for previous buffer image (for temporal difference in green channel)
prev_buffer = None

def reset_simulation():
    """Reset positions, velocities, and healths to initial conditions."""
    global positions, velocities, healths, prev_buffer
    positions = torch.rand(num_soldiers, 2).cuda() * 100  # uniformly in 100x100 area
    velocities = torch.zeros(num_soldiers, 2).cuda()
    healths = torch.ones(num_soldiers).cuda().requires_grad_()
    prev_buffer = None
    print("Simulation reset: All soldiers reinitialized.")

def setup_opengl():
    """Set up the OpenGL environment and enable blending."""
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(window_width, window_height)
    glutInitWindowPosition(100, 100)
    glutCreateWindow(b"Army Simulation")
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

def draw_arrow(x, y, dx, dy, health, team, base_size=0.5):
    """
    Draw a soldier's arrow if alive.
    The arrow's color is fixed by team:
       - Team 1: red
       - Team 2: blue
    The alpha value is set to the soldier's current health.
    """
    if health.item() <= 0:
        return

    velocity_magnitude = np.sqrt(dx**2 + dy**2)
    arrow_size = base_size + 0.05 * velocity_magnitude
    glLineWidth(2.0 + 0.5 * velocity_magnitude)

    # Fixed color based on team; alpha = health.
    if team == 1:
        glColor4f(1.0, 0.0, 0.0, health.item())
    else:
        glColor4f(0.0, 0.0, 1.0, health.item())

    glBegin(GL_LINES)
    glVertex2f(x, y)
    glVertex2f(x + dx, y + dy)
    glEnd()

    angle = np.arctan2(dy, dx)
    arrow_angle1 = angle + np.pi / 6
    arrow_angle2 = angle - np.pi / 6

    arrow_dx1 = arrow_size * np.cos(arrow_angle1)
    arrow_dy1 = arrow_size * np.sin(arrow_angle1)
    arrow_dx2 = arrow_size * np.cos(arrow_angle2)
    arrow_dy2 = arrow_size * np.sin(arrow_angle2)

    glBegin(GL_LINES)
    glVertex2f(x + dx, y + dy)
    glVertex2f(x + dx - arrow_dx1, y + dy - arrow_dy1)
    glVertex2f(x + dx, y + dy)
    glVertex2f(x + dx - arrow_dx2, y + dy - arrow_dy2)
    glEnd()

def render_simulation_opengl():
    """Render the simulation (only living soldiers are drawn)."""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    pos_np = positions.cpu().detach().numpy()
    health_np = healths.cpu().detach().numpy()
    living_mask = (health_np > 0).flatten()
    living_positions = pos_np[living_mask] if np.sum(living_mask) > 0 else pos_np

    min_x, min_y = living_positions.min(axis=0)
    max_x, max_y = living_positions.max(axis=0)
    padding = 10
    gluOrtho2D(min_x - padding, max_x + padding, min_y - padding, max_y + padding)

    for i in range(num_soldiers):
        if healths[i].item() > 0:
            team = 1 if i < half else 2
            draw_arrow(
                positions[i, 0].item(),
                positions[i, 1].item(),
                velocities[i, 0].item(),
                velocities[i, 1].item(),
                healths[i],
                team
            )
    glutSwapBuffers()

def capture_simulation_image():
    """
    Capture the OpenGL-rendered image and resize it.
    Also update the green channel to reflect temporal change.
    """
    global prev_buffer
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, window_width, window_height, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape(window_height, window_width, 3)
    image = torch.tensor(image[::-1].copy()).permute(2, 0, 1).float() / 255.0  # (3, H, W)

    # Convert image to grayscale.
    grayscale = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    if prev_buffer is None:
        prev_buffer = grayscale.clone()
        diff = torch.zeros_like(grayscale)
    else:
        diff = torch.abs(grayscale - prev_buffer)
        prev_buffer = grayscale.clone()
    # Replace green channel with normalized difference.
    image[1] = diff / diff.max().clamp(min=1e-6)
    image = F.interpolate(image.unsqueeze(0), size=(image_size, image_size)).squeeze(0)
    return image

# ------------------ New Aux Feature Functions ------------------

def generate_batch(positions, velocities, healths, k):
    """
    Generate auxiliary features for each soldier by computing k-nearest neighbors
    based on distance (after discounting those outside the field-of-view) and then
    concatenating neighbor velocities, healths, relative positions, and mean position differences.
    The resulting tensor has shape (num_soldiers, 14*k).
    """
    first_half_positions = positions[:half]
    second_half_positions = positions[half:]
    # Compute pairwise distances (num_soldiers x num_soldiers)
    distances = torch.norm(positions[:, None] - positions, dim=2)

    # Compute the dot product between the difference vectors and velocities.
    diffs = positions[:, None] - positions  # shape (N, N, 2)
    dot_product = (diffs * velocities[None, :]).sum(dim=2)

    # Calculate the angle between the difference vector and the velocity.
    norm_diffs = torch.norm(diffs, dim=2)
    norm_vel = torch.norm(velocities, dim=1) + 1e-6  # avoid zero division
    cos_angle = dot_product / (norm_diffs * norm_vel.unsqueeze(0) + 1e-6)
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle = torch.acos(cos_angle)

    # For soldiers outside the field-of-view or dead ones, add a large value.
    distances[(angle > field_of_view) | (healths <= 0)] += distances.max()

    # For each soldier, get the k nearest neighbors from the first half.
    _, nearest_neighbors = torch.topk(-distances[:, :half], k, dim=1)
    nearest_neighbors = nearest_neighbors.T  # shape (k, num_soldiers)

    # For each soldier, get the k nearest neighbors from the second half.
    _, nearest_neighbors_2 = torch.topk(-distances[:, half:], k, dim=1)
    nearest_neighbors_2 = nearest_neighbors_2.T + half  # shape (k, num_soldiers)

    # Compute relative positions.
    relative_positions = positions[nearest_neighbors] - positions[None, :].repeat(k, 1, 1)
    relative_positions_2 = positions[nearest_neighbors_2] - positions[None, :].repeat(k, 1, 1)

    # Compute mean positions of each half relative to each soldier.
    mean_first_half_pos = torch.mean(first_half_positions, dim=0) - positions[None, :].repeat(k, 1, 1)
    mean_second_half_pos = torch.mean(second_half_positions, dim=0) - positions[None, :].repeat(k, 1, 1)
    # Swap for soldiers in the second half.
    mean_first_half_pos[half:], mean_second_half_pos[half:] = mean_second_half_pos[half:], mean_first_half_pos[half:]

    batch_input = torch.cat([
        velocities[nearest_neighbors].reshape(num_soldiers, 2 * k),
        velocities[nearest_neighbors_2].reshape(num_soldiers, 2 * k),
        healths[nearest_neighbors].reshape(num_soldiers, k),
        healths[nearest_neighbors_2].reshape(num_soldiers, k),
        relative_positions.reshape(num_soldiers, 2 * k),
        relative_positions_2.reshape(num_soldiers, 2 * k),
        mean_first_half_pos.reshape(num_soldiers, 2 * k),
        mean_second_half_pos.reshape(num_soldiers, 2 * k)
    ], dim=1)
    return batch_input

class ArmyNet(nn.Module):
    def __init__(self, input_size, output_size, size=128):
        """
        A feed-forward network that accepts the concatenated auxiliary features.
        The input size should be set based on the k parameter: 14*k + 8.
        """
        super(ArmyNet, self).__init__()
        def block(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat), nn.Mish()]
        self.model = nn.Sequential(
            *block(input_size, size), nn.Dropout(0.5),
            *block(size, size // 2), nn.Dropout(0.5),
            *block(size // 2, size // 4), nn.Dropout(0.5),
            *block(size // 4, output_size)
        )
        self.model[-1] = nn.Tanh()
    def forward(self, x):
        return self.model(x)

# Instantiate networks.
# Input size = 14*k (from generate_batch) + 8 (from extra army statistics) = 14*5 + 8 = 78.
input_size = 14 * k + 8
army_1_net = ArmyNet(input_size, 2).cuda()
army_2_net = ArmyNet(input_size, 2).cuda()

optimizer_1 = torch.optim.Adam(army_1_net.parameters(), lr=1e-4)
optimizer_2 = torch.optim.Adam(army_2_net.parameters(), lr=1e-4)

def compute_repulsive_forces(positions, healths, d_min=1.0, repulsion_strength=0.01):
    """
    Compute repulsive forces between soldiers that are too close.
    For each pair with distance < d_min, add a force proportional to (d_min - distance).
    """
    n = positions.shape[0]
    diffs = positions.unsqueeze(1) - positions.unsqueeze(0)  # (n, n, 2)
    dists = torch.norm(diffs, dim=2)  # (n, n)
    eps = 1e-6
    mask = (dists < d_min) & (dists > 0)
    magnitudes = repulsion_strength * (d_min - dists) * mask.float()
    norm_diffs = diffs / (dists.unsqueeze(2) + eps)
    forces = norm_diffs * magnitudes.unsqueeze(2)
    repulsive_force = forces.sum(dim=1)
    repulsive_force[healths <= 0] = 0
    return repulsive_force

def loss_function():
    """
    Compute loss based on collisions between living soldiers from different armies.
    Additionally, if a collision causes an enemy soldier to be killed (health drops to 0),
    then the colliding soldier(s) that were alive remain fully restored.
    
    For each collision between soldier i (army1) and soldier j (army2):
       Damage to i = damage_factor * health[j]
       Damage to j = damage_factor * health[i]
    """
    # Clone current health for computations.
    army1_health = healths[:half]
    army2_health = healths[half:]
    
    # Compute collisions between every soldier in Army 1 and Army 2.
    distances = torch.norm(positions[:half][:, None] - positions[half:], dim=2)
    collisions = distances < 1.0  # collision threshold (boolean mask)
    
    damage_factor = 0.1
    
    # Compute the total damage inflicted.
    damage_army1 = damage_factor * torch.matmul(collisions.float(), army2_health.unsqueeze(1)).squeeze(1)
    damage_army2 = damage_factor * torch.matmul(collisions.float().t(), army1_health.unsqueeze(1)).squeeze(1)
    
    # Apply damage (preliminary new health).
    new_army1 = torch.clamp(army1_health - damage_army1, 0, 1)
    new_army2 = torch.clamp(army2_health - damage_army2, 0, 1)
    
    # Identify kills: enemy soldiers that drop from >0 to 0.
    kill_mask_army2 = (army2_health > 0) & (new_army2 <= 0)
    kill_mask_army1 = (army1_health > 0) & (new_army1 <= 0)
    
    # For each soldier in Army 1, check if it collided with any killed Army 2 soldier.
    bonus_army1_mask = (collisions[:, kill_mask_army2]).any(dim=1)
    bonus_army2_mask = (collisions.t()[:, kill_mask_army1]).any(dim=1)
    
    # Only restore those that were originally alive.
    bonus_army1_mask = bonus_army1_mask & (army1_health > 0)
    bonus_army2_mask = bonus_army2_mask & (army2_health > 0)
    
    new_army1[bonus_army1_mask] = 1.0
    new_army2[bonus_army2_mask] = 1.0
    
    total_damage_army1 = (army1_health - new_army1).sum().item()
    total_damage_army2 = (army2_health - new_army2).sum().item()
    print(f"DEBUG: Army1 damage = {total_damage_army1:.3f}, Army2 damage = {total_damage_army2:.3f}")
    
    loss = total_damage_army1 + total_damage_army2
    new_healths = torch.cat([new_army1, new_army2])
    return torch.tensor(loss, device=positions.device, requires_grad=True), new_healths

def loss_function():
    """
    Compute the loss based on collisions between the two armies.
    For each collision (distance < 1.0), reduce the health of the colliding soldier
    by an amount proportional to the relative health of the enemy.
    Returns the difference in mean health change (delta_diff) and the updated health tensor.
    """
    distances = torch.norm(positions[:half][:, None] - positions[half:], dim=2)
    collisions = distances < 1.0  # collision threshold
    colliding_soldiers = torch.nonzero(collisions)

    army_1_healths = healths[:half]
    army_2_healths = healths[half:]

    initial_army_1 = army_1_healths.mean()
    initial_army_2 = army_2_healths.mean()

    colliding_mask = torch.zeros(army_1_healths.shape, dtype=torch.bool).cuda()
    if colliding_soldiers.size(0) > 0:
        colliding_mask[colliding_soldiers[:, 0]] = True

    total_health = army_1_healths + army_2_healths + 1e-6
    relative_health_army1 = army_1_healths / total_health
    relative_health_army2 = army_2_healths / total_health

    army_1_healths_new = torch.where(
        colliding_mask & (army_2_healths > 0),
        army_1_healths - relative_health_army2,
        army_1_healths
    )

    colliding_mask = torch.zeros(army_2_healths.shape, dtype=torch.bool).cuda()
    if colliding_soldiers.size(0) > 0:
        colliding_mask[colliding_soldiers[:, 1]] = True

    army_2_healths_new = torch.where(
        colliding_mask & (army_1_healths > 0),
        army_2_healths - relative_health_army1,
        army_2_healths
    )

    delta_army_1 = army_1_healths_new.mean() - initial_army_1
    delta_army_2 = army_2_healths_new.mean() - initial_army_2
    delta_diff = delta_army_1 - delta_army_2

    hel = torch.cat([army_1_healths_new, army_2_healths_new])
    hel = torch.clamp(hel, 0, 1)

    return delta_diff, hel

def main_training_loop():
    setup_opengl()
    global positions, velocities, healths

    reset_simulation()

    for i in range(num_iterations):
        # Render simulation (for visualization)
        render_simulation_opengl()
        
        # Generate the batch of auxiliary features based on k nearest neighbors.
        batch_input = generate_batch(positions, velocities, healths, k)

        # Compute additional army statistics.
        mean_health_army1 = torch.mean(healths[:half])
        mean_health_army2 = torch.mean(healths[half:])
        std_health_army1 = torch.std(healths[:half])
        std_health_army2 = torch.std(healths[half:])
        std_position_army1 = torch.std(positions[:half], dim=0)
        std_position_army2 = torch.std(positions[half:], dim=0)

        # For soldiers in the first half.
        first_half_stats = torch.cat([
            mean_health_army1.view(1, 1),
            std_health_army1.view(1, 1),
            std_health_army2.view(1, 1),
            std_position_army1.view(1, 2),
            std_position_army2.view(1, 2)
        ], dim=1)
        # For soldiers in the second half (reverse order).
        second_half_stats = torch.cat([
            mean_health_army2.view(1, 1),
            std_health_army2.view(1, 1),
            std_health_army1.view(1, 1),
            std_position_army2.view(1, 2),
            std_position_army1.view(1, 2)
        ], dim=1)

        first_half_stats = first_half_stats.repeat(half, 1)
        second_half_stats = second_half_stats.repeat(half, 1)
        army_stats = torch.cat([first_half_stats, second_half_stats], dim=0)

        # Append the extra army stats and each soldier’s current health.
        batch_input = torch.cat([batch_input, army_stats, healths.view(num_soldiers, 1)], dim=1).detach()

        # Compute accelerations for each army.
        army_1_accelerations = army_1_net(batch_input[:half].cuda())
        army_2_accelerations = army_2_net(batch_input[half:].cuda())
        accelerations = torch.cat([army_1_accelerations, army_2_accelerations], dim=0)

        # Zero out accelerations for dead soldiers.
        dead_mask = (healths <= 0).unsqueeze(1)
        accelerations = torch.where(dead_mask, torch.zeros_like(accelerations), accelerations)

        # Add repulsive forces.
        repulsive = compute_repulsive_forces(positions, healths, d_min=1.0, repulsion_strength=0.01)
        accelerations = accelerations + repulsive

        # Update velocities and apply damping.
        velocities_new = velocities + accelerations
        damping = 0.99
        velocities_new = velocities_new * damping

        # Limit maximum speed (proportional to current health).
        base_max_speed = 5.0
        max_speed = base_max_speed * healths.unsqueeze(1)
        norms = torch.norm(velocities_new, dim=1, keepdim=True)
        velocities_new = torch.where(norms > max_speed, velocities_new / norms * max_speed, velocities_new)
        velocities = velocities_new

        # Update positions.
        positions = positions + velocities

        # Enforce boundaries.
        min_bound, max_bound = 0.0, 100.0
        mask_x_low = positions[:, 0] < min_bound
        mask_x_high = positions[:, 0] > max_bound
        if mask_x_low.any():
            positions[mask_x_low, 0] = min_bound
            velocities[mask_x_low, 0] = -velocities[mask_x_low, 0]
        if mask_x_high.any():
            positions[mask_x_high, 0] = max_bound
            velocities[mask_x_high, 0] = -velocities[mask_x_high, 0]
        mask_y_low = positions[:, 1] < min_bound
        mask_y_high = positions[:, 1] > max_bound
        if mask_y_low.any():
            positions[mask_y_low, 1] = min_bound
            velocities[mask_y_low, 1] = -velocities[mask_y_low, 1]
        if mask_y_high.any():
            positions[mask_y_high, 1] = max_bound
            velocities[mask_y_high, 1] = -velocities[mask_y_high, 1]

        # Alternate optimizer updates.
        current_army = (i // ui) % 2
        if current_army == 1:
            optimizer_1.zero_grad()
            loss, new_health = loss_function()
            loss.backward(retain_graph=True)
            if i % ui == (ui - 1):
                optimizer_1.step()
        else:
            optimizer_2.zero_grad()
            loss, new_health = loss_function()
            loss = -loss  # flip loss for enemy update
            loss.backward(retain_graph=True)
            if i % ui == (ui - 1):
                optimizer_2.step()

        positions = positions.detach()
        velocities = velocities.detach()
        healths = new_health.detach().clone().requires_grad_()

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.4f}")
        avg_health_army1 = healths[:half].mean().item()
        avg_health_army2 = healths[half:].mean().item()
        if avg_health_army1 > avg_health_army2:
            army_2_net.load_state_dict(army_1_net.state_dict())
            print("Army1 is winning. Army2 network updated.")
        elif avg_health_army2 > avg_health_army1:
            army_1_net.load_state_dict(army_2_net.state_dict())
            print("Army2 is winning. Army1 network updated.")

        if (healths[:half] <= 0).all() or (healths[half:] <= 0).all():
            print("One army wiped out. Resetting simulation.")
            reset_simulation()

if __name__ == "__main__":
    main_training_loop()
