import gym, torch
from gym import spaces
import torch.nn as nn
import matplotlib.pyplot as plt

class MultiAgentEnv(gym.Env):
    def __init__(self):
        self.num_particles = 1000
        self.half = self.num_particles // 2
        self.k = 10
        self.positions = torch.rand(self.num_particles, 2).cuda() * 100.0
        self.velocities = torch.zeros(self.num_particles, 2).cuda()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=0, high=100, shape=(6*self.k+2,))

    def generate_batch(self, k):
        self.positions = self.positions.detach()
        self.velocities = self.velocities.detach()

        half = self.num_particles // 2
        first_half_positions = self.positions[:half]
        second_half_positions = self.positions[half:]
        # calculate the distance matrix
        distances = torch.norm(self.positions[:, None] - self.positions, dim=2)
        #Find the k nearest neighbors for each particle in the first half
        _, nearest_neighbors = torch.topk(-distances[:,:half], k, dim=1)
        nearest_neighbors = nearest_neighbors.T
        # Find the k nearest neighbors for each particle in the second half
        _, nearest_neighbors_2 = torch.topk(-distances[:, half:], k, dim=1)
        nearest_neighbors_2 = nearest_neighbors_2.T + half
        # Compute the relative positions of the nearest neighbors
        relative_positions = self.positions[nearest_neighbors] - self.positions[None, :].repeat(k, 1, 1)
        relative_positions_2 = self.positions[nearest_neighbors_2] - self.positions[None, :].repeat(k, 1, 1)
        # Compute the relative positionsy of the goals
        first_half_goals = torch.mean(second_half_positions, dim=0)
        second_half_goals = torch.mean(first_half_positions, dim=0)
        relative_goals = torch.cat([first_half_goals - first_half_positions, second_half_goals - second_half_positions])
        # Concatenate positions, velocities, relative positions, and relative goals
        batch_input = torch.cat([self.velocities[nearest_neighbors].reshape(self.num_particles, 2 * k), relative_positions.reshape(self.num_particles, 2 * k), 
            relative_positions_2.reshape(self.num_particles, 2 * k), relative_goals], dim=1)
        return batch_input

    def loss_function(self, first_half):
        half = self.num_particles // 2
        # Check for collisions
        distances = torch.norm(self.positions[:half, None] - self.positions[half:], dim=2)
        collision_ratio = 1 - torch.sigmoid(distances-0.1) # changed here
        indices = (collision_ratio > 0.0).nonzero()
        if not first_half: return -collision_ratio.mean()
        # Compute the mean position of the predators
        predator_mean_position = self.positions[half:].mean(dim=0)
        # Compute the mean distance of preys to the mean position of the predators
        mean_predator_distance = torch.mean(torch.norm(self.positions[:half]-predator_mean_position, dim=1))
        return (collision_ratio + mean_predator_distance).mean()
        
    def step(self, action, first_half, dt = 0.1):
        # update the position and velocity based on the action
        self.velocities += action * dt

        # Multiply accelerations of the predators by 2
        self.velocities = self.velocities/torch.norm(self.velocities, dim=1, keepdim=True)
        self.velocities[self.num_particles//2:] *= 2

        self.positions += self.velocities * dt

        #compute the loss
        return False, self.loss_function(first_half), False, {}
    
    def reset(self):
        self.positions = torch.rand(self.num_particles, 2).cuda() * 100.0
        self.velocities = torch.zeros(self.num_particles, 2).cuda()
    
    def render(self):
        diff = self.positions - self.velocities

        ind = torch.cat((torch.zeros(self.num_particles//2), 
            torch.ones(self.num_particles//2)), dim=0)

        plt.clf()
        plt.quiver(diff[:, 0].cpu(), diff[:, 1].cpu(), self.velocities[:, 0].cpu(), 
            self.velocities[:, 1].cpu(), ind.float().cpu().numpy(), cmap ='cool')
        plt.show()


# Initialize the environment
env = MultiAgentEnv()

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

# Define the prey and predator acceleration networks
prey_net = AccelerationNetwork(input_size=6*env.k+2, hidden_size=256, output_size=2).cuda()
predator_net = AccelerationNetwork(input_size=6*env.k+2, hidden_size=256, output_size=2).cuda()

# Define the optimizers for the networks
prey_optimizer = torch.optim.AdamW(prey_net.parameters(), lr=0.001)
predator_optimizer = torch.optim.AdamW(predator_net.parameters(), lr=0.001)

# Define the number of episodes and steps per episode
num_epochs = 1000
num_episodes = 1000

for episode in range(num_episodes):

    env.reset()
    
    for epoch in range(num_epochs):
        observation = env.generate_batch(10)
        # Get the actions for the prey and predator
        prey_action = prey_net(observation[:env.half])
        predator_action = predator_net(observation[env.half:])
        # Combine the actions and send them to the environment
        actions = torch.cat([prey_action, predator_action])
        _, loss, _, _ = env.step(actions, epoch % 2)

        # Alternate between training the prey and predator networks
        if epoch % 2:
            prey_optimizer.zero_grad()
            loss.backward()
            prey_optimizer.step()
            string = 'Prey'
        else:
            predator_optimizer.zero_grad()
            loss.backward()
            predator_optimizer.step()
            string = 'Predator'

        env.render()

        # Print the episode and loss information
        print("Iteration: %i %s Loss: %f" % (epoch, string, loss.item()))