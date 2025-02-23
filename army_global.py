import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from torchvision import models

# Simulation parameters
num_soldiers = 200
half = num_soldiers // 2
num_iterations = 90000
ui = 50  # update interval for optimizer steps
window_width, window_height = 1024, 1024
global_image_size = 128  # global simulation image size
local_view_size = 32     # patch size for local view

# Global simulation state variables
positions = None
velocities = None
healths = None

# Global variable for previous buffer image (for temporal difference in green channel)
prev_buffer = None

def update_last_frame(current_frame, alpha=0.5):
    global prev_buffer
    if prev_buffer is None:
        return current_frame
    updated_frame = alpha * current_frame + (1 - alpha) * prev_buffer
    return updated_frame

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
    Also, update the green channel to reflect temporal change:
      Compute the grayscale of the current image, then calculate the absolute difference
      with the previous buffer, and store the normalized difference in the green channel.
    """
    global prev_buffer
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, window_width, window_height, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape(window_height, window_width, 3)
    image = torch.tensor(image[::-1].copy()).permute(2, 0, 1).float() / 255.0

    grayscale = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    if prev_buffer is None:
        prev_buffer = grayscale.clone()
        diff = torch.zeros_like(grayscale)
    else:
        diff = torch.abs(grayscale - prev_buffer)
        prev_buffer = grayscale.clone()
    image[1] = diff / diff.max().clamp(min=1e-6)
    image = F.interpolate(image.unsqueeze(0), size=(global_image_size, global_image_size)).squeeze(0)
    return image

def create_local_views(positions, simulation_image):
    """
    Extract a local view (patch) around each soldier from the simulation image.
    The simulation image is assumed to span a 100x100 area.
    """
    _, h, w = simulation_image.shape
    scale_x, scale_y = 100 / w, 100 / h
    simulation_image = simulation_image.to(positions.device)
    pixel_positions = (positions / torch.tensor([scale_x, scale_y], device=positions.device)).long()
    half_view = local_view_size // 2
    y_indices = torch.arange(-half_view, half_view, device=positions.device).view(1, -1) + pixel_positions[:, 1].unsqueeze(1)
    x_indices = torch.arange(-half_view, half_view, device=positions.device).view(1, -1) + pixel_positions[:, 0].unsqueeze(1)
    y_indices = torch.clamp(y_indices, 0, h - 1)
    x_indices = torch.clamp(x_indices, 0, w - 1)
    local_views = simulation_image[:, y_indices.unsqueeze(2), x_indices.unsqueeze(1)]
    return local_views.permute(1, 0, 2, 3)  # (batch, channels, local_view_size, local_view_size)

###############################################################################
# Dual-View Network Using a Shared MobileNet Instance
###############################################################################
class DualViewSharedCNNArmyNet(nn.Module):
    def __init__(self, aux_input_size, output_size, fusion_size=128):
        """
        A CNN-based network that uses the same MobileNet feature extractor for both
        local views (soldier patches) and the global view (entire simulation image),
        then fuses these features with auxiliary global statistics.
        """
        super(DualViewSharedCNNArmyNet, self).__init__()
        # Shared MobileNet instance (pre-trained and frozen)
        mobilenet = models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Assume feature extractor outputs a feature vector of dimension 576.
        feature_dim = 576  
        # Combined features: local + global + auxiliary.
        combined_feature_dim = feature_dim * 2 + aux_input_size
        self.fc = nn.Sequential(
            nn.Linear(combined_feature_dim, fusion_size),
            nn.Mish(),
            nn.Dropout(0.5),
            nn.Linear(fusion_size, fusion_size // 2),
            nn.Mish(),
            nn.Dropout(0.5),
            nn.Linear(fusion_size // 2, fusion_size // 4),
            nn.Mish(),
            nn.Dropout(0.5),
            nn.Linear(fusion_size // 4, output_size),
            nn.Tanh()
        )

    def forward(self, local_views, global_view, aux_input):
        # Extract features from local views (batch of patches).
        local_features = self.feature_extractor(local_views)
        local_features = local_features.view(local_features.size(0), -1)
        
        # Extract features from the global view (a single simulation image).
        global_features = self.feature_extractor(global_view.unsqueeze(0))
        global_features = global_features.view(global_features.size(0), -1)
        # Replicate global features for each soldier.
        global_features = global_features.expand(local_features.size(0), -1)
        
        # Concatenate local, global, and auxiliary features.
        combined = torch.cat([local_features, global_features, aux_input], dim=1)
        output = self.fc(combined)
        return output

# Instantiate networks for both armies (auxiliary input size = 11)
army_1_net = DualViewSharedCNNArmyNet(aux_input_size=11, output_size=2).cuda()
army_2_net = DualViewSharedCNNArmyNet(aux_input_size=11, output_size=2).cuda()

optimizer_1 = torch.optim.Adam(army_1_net.parameters(), lr=1e-4)
optimizer_2 = torch.optim.Adam(army_2_net.parameters(), lr=1e-4)

def compute_repulsive_forces(positions, healths, d_min=1.0, repulsion_strength=0.01):
    """
    Compute repulsive forces between soldiers that are too close.
    For each pair (i,j) with distance < d_min, add a force proportional to (d_min - distance).
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
    Only living soldiers (health > 0) are considered.
    For each collision between soldier i (army1) and soldier j (army2):
       Damage to i = damage_factor * health[j]
       Damage to j = damage_factor * health[i]
    """
    new_healths = healths.clone()
    army1_healths = healths[:half]
    army2_healths = healths[half:]
    distances = torch.norm(positions[:half][:, None] - positions[half:], dim=2)
    collisions = distances < 1.0  # collision threshold
    damage_factor = 0.1
    damage_army1 = damage_factor * torch.matmul(collisions.float(), army2_healths.unsqueeze(1)).squeeze(1)
    damage_army2 = damage_factor * torch.matmul(collisions.float().t(), army1_healths.unsqueeze(1)).squeeze(1)
    army1_healths_new = torch.clamp(army1_healths - damage_army1, 0, 1)
    army2_healths_new = torch.clamp(army2_healths - damage_army2, 0, 1)
    new_healths = torch.cat([army1_healths_new, army2_healths_new])
    total_damage_army1 = (army1_healths - army1_healths_new).sum().item()
    total_damage_army2 = (army2_healths - army2_healths_new).sum().item()
    print(f"DEBUG: Army1 damage = {total_damage_army1:.3f}, Army2 damage = {total_damage_army2:.3f}")
    loss = total_damage_army1 + total_damage_army2
    return torch.tensor(loss, device=positions.device, requires_grad=True), new_healths

def main_training_loop():
    setup_opengl()
    global positions, velocities, healths

    reset_simulation()

    for i in range(num_iterations):
        render_simulation_opengl()
        
        # Capture the simulation image (global view).
        sim_image_army1 = capture_simulation_image()  # Army 1: as-is.
        sim_image_army2 = sim_image_army1[[2, 1, 0], :, :]  # Army 2: swap red and blue.
        
        # Create local views for each soldier.
        local_views_army1 = create_local_views(positions, sim_image_army1)
        local_views_army2 = create_local_views(positions, sim_image_army2)
        
        # Compute auxiliary inputs for each soldier.
        army1_health = healths[:half]
        army2_health = healths[half:]
        mean_health_army1 = army1_health.mean()
        std_health_army1 = army1_health.std()
        std_pos_army1 = positions[:half].std(dim=0)
        ratio_alive_army1 = (army1_health > 0).float().mean()
        mean_health_army2 = army2_health.mean()
        std_health_army2 = army2_health.std()
        std_pos_army2 = positions[half:].std(dim=0)
        ratio_alive_army2 = (army2_health > 0).float().mean()
        
        aux_army1 = torch.cat([
            mean_health_army1.expand(half, 1),
            std_health_army1.expand(half, 1),
            std_pos_army1[0].unsqueeze(0).expand(half, 1),
            std_pos_army1[1].unsqueeze(0).expand(half, 1),
            ratio_alive_army1.expand(half, 1),
            mean_health_army2.expand(half, 1),
            std_health_army2.expand(half, 1),
            std_pos_army2[0].unsqueeze(0).expand(half, 1),
            std_pos_army2[1].unsqueeze(0).expand(half, 1),
            ratio_alive_army2.expand(half, 1),
            healths[:half].unsqueeze(1)
        ], dim=1)
        
        aux_army2 = torch.cat([
            mean_health_army2.expand(half, 1),
            std_health_army2.expand(half, 1),
            std_pos_army2[0].unsqueeze(0).expand(half, 1),
            std_pos_army2[1].unsqueeze(0).expand(half, 1),
            ratio_alive_army2.expand(half, 1),
            mean_health_army1.expand(half, 1),
            std_health_army1.expand(half, 1),
            std_pos_army1[0].unsqueeze(0).expand(half, 1),
            std_pos_army1[1].unsqueeze(0).expand(half, 1),
            ratio_alive_army1.expand(half, 1),
            healths[half:].unsqueeze(1)
        ], dim=1)
        
        aux_input = torch.cat([aux_army1, aux_army2], dim=0)
        
        # Compute network accelerations for each army using the shared feature extractor.
        # Army 1 processes its local views with the original global image.
        army_1_acc = army_1_net(
            local_views_army1[:half].cuda(),
            sim_image_army1.cuda(),
            aux_input[:half].cuda()
        )
        # Army 2 uses its local views and the swapped-color global image.
        army_2_acc = army_2_net(
            local_views_army2[half:].cuda(),
            sim_image_army2.cuda(),
            aux_input[half:].cuda()
        )
        accelerations = torch.cat([army_1_acc, army_2_acc], dim=0)
        
        # Zero out accelerations for dead soldiers.
        dead_mask = (healths <= 0).unsqueeze(1)
        accelerations = torch.where(dead_mask, torch.zeros_like(accelerations), accelerations)
        
        # Compute repulsive forces.
        repulsive = compute_repulsive_forces(positions, healths, d_min=1.0, repulsion_strength=0.01)
        accelerations = accelerations + repulsive
        
        velocities = velocities + accelerations
        damping = 0.99
        velocities = velocities * damping
        
        # Limit maximum speed proportional to current health.
        base_max_speed = 5.0
        max_speed = base_max_speed * healths.unsqueeze(1)
        norms = torch.norm(velocities, dim=1, keepdim=True)
        velocities = torch.where(norms > max_speed, velocities / norms * max_speed, velocities)
        
        positions = positions + velocities
        
        # Enforce simulation boundaries.
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
            loss = -loss  # Flip loss for adversarial update.
            loss.backward(retain_graph=True)
            if i % ui == (ui - 1):
                optimizer_2.step()
        
        positions = positions.detach()
        velocities = velocities.detach()
        healths = new_health.detach().clone().requires_grad_()
        
        if i % 1000 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.4f}")
        
            # Update the losing army's network with the winning army's weights.
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
