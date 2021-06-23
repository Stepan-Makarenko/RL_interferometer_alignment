import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from utils import action_rescale, Replay_buffer
# from utils import unif_weight_init as weight_init
from utils import ortog_weight_init as weight_init


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, encoder='standart', device='cpu'):
        super(Actor, self).__init__()
        self.device = device
        self.action_dim = action_dim
        # Changed Body 
        if encoder == 'standart':
            self.body = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.Flatten()
            ).to(self.device)
        elif encoder == 'VGG':
            self.body = nn.Sequential( #receive 16 x 64 x 64
                nn.Conv2d(16, 32, kernel_size=3, padding=1), # 32 x 64 x 64
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), # 32 x 64 x 64
                nn.ReLU(),
                nn.MaxPool2d(2, 2), 
                nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64 x 32 x 32
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), # 64 x 32 x 32
                nn.ReLU(),
                nn.MaxPool2d(2, 2), 
                nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128 x 16 x 16
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), # 128 x 16 x 16
                nn.ReLU(),
                nn.MaxPool2d(2, 2), 
                nn.Conv2d(128, 256, kernel_size=3, padding=1), # 256 x 8 x 8
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), # 256 x 8 x 8
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten()
            ).to(self.device)


        self.body_out_shape = self._get_layer_out(self.body, state_dim)
        
        self.nonlinearity = nn.Sequential(
            nn.Linear(self.body_out_shape, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU()
        ).to(self.device)
        

        # mu head
        self.mu_layer = nn.Sequential(
            nn.Linear(n_latent_var, action_dim),
            nn.Tanh()
        ).to(self.device)
        
        
        self.outputs = dict()
        self.apply(weight_init)
        
    def _get_layer_out(self, layer, shape):
        o = layer(torch.zeros(1, *shape).to(self.device))
        return int(np.prod(o.shape))

    def forward(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device)
        
        obs = self.body(obs)
        obs = self.nonlinearity(obs)
    
        mu = self.mu_layer(obs)
        self.outputs['mu'] = mu
        
        return mu
    
    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.add_histogram(f'train_actor/{k}_hist', v, step)
    
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, activation=nn.ReLU(), encoder='standart', device='cpu'):
        super(Critic, self).__init__()
        self.device = device
        
        # Changed Body 
        if encoder == 'standart':
            self.body = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.Flatten()
            ).to(self.device)
        elif encoder == 'VGG':
            self.body = nn.Sequential( #receive 16 x 64 x 64
                nn.Conv2d(16, 32, kernel_size=3, padding=1), # 32 x 64 x 64
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), # 32 x 64 x 64
                nn.ReLU(),
                nn.MaxPool2d(2, 2), 
                nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64 x 32 x 32
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), # 64 x 32 x 32
                nn.ReLU(),
                nn.MaxPool2d(2, 2), 
                nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128 x 16 x 16
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), # 128 x 16 x 16
                nn.ReLU(),
                nn.MaxPool2d(2, 2), 
                nn.Conv2d(128, 256, kernel_size=3, padding=1), # 256 x 8 x 8
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1), # 256 x 8 x 8
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # 256 x 4 x 4
                nn.Flatten()
            ).to(self.device)


        self.body_out_shape = self._get_layer_out(self.body, state_dim)
        

        # Critic head
        self.Q_layer = nn.Sequential(
            nn.Linear(self.body_out_shape + action_dim, n_latent_var),
            activation,
            nn.Linear(n_latent_var, n_latent_var),
            activation,
            nn.Linear(n_latent_var, 1)
        ).to(self.device)
        
        self.outputs = dict()
        self.apply(weight_init)
        
    def _get_layer_out(self, layer, shape):
        o = layer(torch.zeros(1, *shape).to(self.device))
        return int(np.prod(o.shape))
    
    def forward(self, obs, act):
        obs = torch.from_numpy(obs).float().to(self.device)
        if isinstance(act, np.ndarray):
            act = torch.from_numpy(act).float().to(self.device)
        
        obs = self.body(obs)
                 
        value = self.Q_layer(torch.cat((obs, act), dim=-1))
        return value

class TD3:
    def __init__(self, writer, state_dim=[16,64,64], action_dim=4, n_latent_var=256, q_lr=5e-4, pi_lr=1e-5, betas=(0.9, 0.999),
                 gamma=0.99, epochs=8, batch_size=32, device='cpu', polyak=0.995, max_grad_norm=20, activation=nn.ReLU(), target_noise=0.2, noise_clip=0.5, policy_delay=2, critic_L2_norm=1e-2, encoder='standart'):
        self.q_lr = q_lr
        self.pi_lr = pi_lr
        self.action_dim = action_dim
        self.betas = betas
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.writer = writer
        self.polyak = polyak
        self.max_grad_norm = max_grad_norm
        self.timestep = 0
        self.target_noise = target_noise
        self.noise_clip = noise_clip 
        self.policy_delay = policy_delay
        
        self.policy = Actor(state_dim=state_dim,
                            action_dim=action_dim,
                            n_latent_var=n_latent_var,
                            device=self.device,
                            encoder=encoder
                            ).to(self.device)
        self.policy_old = Actor(state_dim,
                                action_dim,
                                n_latent_var,
                                device=self.device,
                                encoder=encoder
                                ).to(self.device)
        
        self.critic1 = Critic(state_dim=state_dim,
                             action_dim=action_dim,
                             n_latent_var=n_latent_var,
                             activation=activation,
                             device=self.device,
                             encoder=encoder
                             ).to(self.device)
        self.target_critic1 = Critic(state_dim,
                                 action_dim,
                                 n_latent_var,
                                 activation,
                                 device=self.device,
                                 encoder=encoder
                                 ).to(self.device)
        
        self.critic2 = Critic(state_dim=state_dim,
                             action_dim=action_dim,
                             n_latent_var=n_latent_var,
                             activation=activation,
                             device=self.device,
                             encoder=encoder
                             ).to(self.device)
        self.target_critic2 = Critic(state_dim,
                                 action_dim,
                                 n_latent_var,
                                 activation,
                                 device=self.device,
                                 encoder=encoder
                                 ).to(self.device)

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=pi_lr, betas=betas)
        self.critic_optimizer1 = torch.optim.Adam(self.critic1.parameters(), lr=q_lr, betas=betas, weight_decay=critic_L2_norm)
        self.critic_optimizer2 = torch.optim.Adam(self.critic2.parameters(), lr=q_lr, betas=betas, weight_decay=critic_L2_norm)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
       
    
    def select_action(self, state, action_noise):
        self.policy.eval()
        mu = self.policy(state)
        self.policy.train()
        action = mu.detach().cpu().numpy()[0]
        if action_noise is not None:
            action = np.clip(action + action_noise.noise(), -1, 1)
        return action
    
    def _soft_update_net(self, old, new):
        for new_param, old_param in zip(new.parameters(), old.parameters()):
            old_param.data.copy_(new_param.data * (1 - self.polyak) + old_param.data * self.polyak)

    def update(self, memory, timestep):
        # Convert lists from memory to tensors
        self.timestep = timestep

        epoch_target_loss1 = 0
        epoch_target_loss2 = 0
        epoch_policy_loss = 0
        epoch_entropy = 0
        epoch_grad1_norm = 0
        epoch_grad2_norm = 0
        epoch_grad3_norm = 0
        mean_Q1_value = 0
        max_Q1_value = -float('inf')
        mean_Q2_value = 0
        max_Q2_value = -float('inf')
        for i in range(self.epochs):
            batch_ind, batch_curr_states, batch_actions, batch_reward, batch_next_states, batch_mask = memory.sample(self.batch_size)
            batch_reward = torch.FloatTensor(batch_reward).to(self.device).detach()
            batch_mask = torch.LongTensor(batch_mask).to(self.device).detach()
                
            Q_values1 = torch.squeeze(self.critic1(batch_curr_states, batch_actions)) 
            Q_values2 = torch.squeeze(self.critic2(batch_curr_states, batch_actions))

            target_noise = Normal(torch.zeros((self.batch_size, self.action_dim)), self.target_noise * torch.ones((self.batch_size, self.action_dim))).sample()
            target_noise = torch.clamp(target_noise, -self.noise_clip, self.noise_clip).to(self.device)
            target_action = torch.clamp(self.policy_old(batch_next_states) + target_noise, -1, 1)
            
            target_Q_values1 = self.target_critic1(batch_next_states, target_action)
            target_Q_values2 = self.target_critic2(batch_next_states, target_action)

            max_Q1_value = max(max_Q1_value, max(Q_values1.cpu().detach()).item())
            mean_Q1_value += torch.mean(Q_values1.cpu().detach()).item()
            max_Q2_value = max(max_Q2_value, max(Q_values2.cpu().detach()).item())
            mean_Q2_value += torch.mean(Q_values2.cpu().detach()).item()

            td_target = batch_reward + self.gamma * torch.min(torch.cat((target_Q_values1, target_Q_values2), -1), dim=1)[0] * batch_mask

            # Optimize Critic
    
            self.critic_optimizer1.zero_grad()
            target_loss1 = nn.MSELoss()(Q_values1, td_target.detach())
            target_loss1.backward()
            grad_norm1 = nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
            self.critic_optimizer1.step()
            
            self.critic_optimizer2.zero_grad()
            target_loss2 = nn.MSELoss()(Q_values2, td_target.detach())
            target_loss2.backward()
            grad_norm2 = nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
            self.critic_optimizer2.step()
            
            if i % self.policy_delay == 0:
                # Optimize policy
                action = self.policy(batch_curr_states)
                policy_Q = self.critic1(batch_curr_states, action)

                self.policy_optimizer.zero_grad()
                policy_loss = -torch.mean(policy_Q)
                policy_loss.backward()

                grad_norm3 = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                epoch_grad3_norm += grad_norm3
                epoch_policy_loss += policy_loss.item()
                
                # Copy new weights into old policy:
                self._soft_update_net(self.policy_old, self.policy)
                self._soft_update_net(self.target_critic1, self.critic1)
                self._soft_update_net(self.target_critic2, self.critic2)
            

            epoch_grad1_norm += grad_norm1
            epoch_grad2_norm += grad_norm2
            epoch_target_loss1 += target_loss1.item()
            epoch_target_loss2 += target_loss1.item()

        self.writer.add_scalar('Hyperparameters/Lr',
                               self.policy_optimizer.param_groups[0]['lr'],
                               self.timestep
        )
        self.writer.add_scalar('Values/maxQ1_value',
                                   max_Q1_value,
                                   timestep
                                   )
        self.writer.add_scalar('Values/meanQ1_value',
                                   mean_Q1_value / (self.epochs),
                                   timestep
                                   )
        self.writer.add_scalar('Values/maxQ2_value',
                                   max_Q2_value,
                                   timestep
                                   )
        self.writer.add_scalar('Values/meanQ2_value',
                                   mean_Q2_value / (self.epochs),
                                   timestep
                                   )
        
        self.writer.add_scalar('Losses/Target1_loss',
                               epoch_target_loss1 / (self.epochs),
                               self.timestep
        )
        self.writer.add_scalar('Losses/Target2_loss',
                               epoch_target_loss2 / (self.epochs),
                               self.timestep
        )
        self.writer.add_scalar('Losses/Policy_loss',
                               epoch_policy_loss / (self.epochs),
                               self.timestep
        )
        self.writer.add_scalar('Losses/Critic1_grad_norm',
                               epoch_grad1_norm / (self.epochs),
                               self.timestep
        )
        self.writer.add_scalar('Losses/Critic2_grad_norm',
                               epoch_grad2_norm / (self.epochs),
                               self.timestep
        )
        self.writer.add_scalar('Losses/Policy_grad_norm',
                               epoch_grad3_norm / (self.epochs),
                               self.timestep
        )
        
        # Log hists
        self.policy.log(self.writer, timestep)