from typing import Optional, List
import numpy as np
import torch
import torch.nn.functional as F

class TrajectoryDiffusion(torch.nn.Module):   
    def __init__(self, model,linear_start: float=0.00085,linear_end: float=0.0120,full_n_steps=1000,ddim_step=50,ddim_eta=0.0,device=None):
        super().__init__()
        self.model=model
        self.device=device
        self.linear_start=linear_start
        self.linear_end=linear_end
        self.full_n_steps=full_n_steps

        beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, self.full_n_steps, dtype=torch.float32,device=self.device) ** 2
        self.beta = torch.nn.Parameter(beta.to(torch.float32), requires_grad=False)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar=alpha_bar
        self.alpha_bar = torch.nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)
        self.ddpm_time_steps = np.asarray(list(range(self.full_n_steps)))

        self.ddim_step=ddim_step
        self.ddim_eta=ddim_eta

        alpha_bar_prev = torch.cat([alpha_bar.new_tensor([1.]), alpha_bar[:-1]])
        self.sqrt_alpha_bar = alpha_bar ** .5
        self.sqrt_1m_alpha_bar = (1. - alpha_bar) ** .5
        self.sqrt_recip_alpha_bar = alpha_bar ** -.5
        self.sqrt_recip_m1_alpha_bar = (1 / alpha_bar - 1) ** .5
        variance = beta * (1. - alpha_bar_prev) / (1. - alpha_bar)
        self.log_var = torch.log(torch.clamp(variance, min=1e-20))
        self.mean_x0_coef = beta * (alpha_bar_prev ** .5) / (1. - alpha_bar)
        self.mean_xt_coef = (1. - alpha_bar_prev) * ((1 - beta) ** 0.5) / (1. - alpha_bar)

    def diffusion_process(self, x0: torch.Tensor, index: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        return self.sqrt_alpha_bar[index].view(x0.shape[0],1,1) * x0 + self.sqrt_1m_alpha_bar[index].view(x0.shape[0],1,1) * noise

    def get_uncertainty(self, x: torch.Tensor, t: torch.Tensor,condtion,temporal_condition:torch.Tensor):
        e_t=self.model.TrajGenerator(x, t,condtion,temporal_condition)
        return e_t

    def get_pred_x0(self, x: torch.Tensor, t: torch.Tensor,home_locations,poi_condition:torch.Tensor,trav_pattern: Optional[torch.Tensor] = None):
        pred_x0=self.model.TrajGenerator(x, t,home_locations,poi_condition,trav_pattern)
        return pred_x0

    def pred_x0(self, e_t: torch.Tensor, index: torch.Tensor, x: torch.Tensor,):
        sqrt_recip_alpha_bar = self.sqrt_recip_alpha_bar[index].view(x.shape[0], 1,  1)
        sqrt_recip_m1_alpha_bar = self.sqrt_recip_m1_alpha_bar[index].view(x.shape[0], 1,  1)
        x0 = sqrt_recip_alpha_bar * x - sqrt_recip_m1_alpha_bar * e_t
        return x0

    def generation_training(self,latent,home_locations: Optional[torch.Tensor] = None,poi_condition: Optional[torch.Tensor] = None,trav_pattern: Optional[torch.Tensor] = None):
        batch_size = latent.shape[0]
        t = torch.randint(low=0, high=self.full_n_steps, size=(batch_size,), device=self.device,dtype=torch.long)  # need to revise when
        x0=latent

        noise_mask=torch.randn(batch_size,self.model.input_len,self.model.loc_size,device=self.device)

        xt = self.diffusion_process(x0, t, noise_mask)

        pred_x0 = self.get_pred_x0(xt, t,home_locations,poi_condition,trav_pattern)
        return pred_x0

    @torch.no_grad()
    def sampler(self,x,shape):
        x = self.model.LocationEncoder(locs=x,lab=self.lab,maxi=self.maxi)
        x0 = self.model.TrajEncoder(sequence=x)
        noise = torch.randn_like(x0)
        xt = self.sqrt_alpha_bar[self.full_n_steps - 1] * x0 + self.sqrt_1m_alpha_bar[self.full_n_steps - 1] * noise
        xtmean = float(torch.mean(xt))  # .view(-1,self.model.loc_size),dim=0
        xtstd = float(torch.std(xt))  # /1.2 #.view(-1,self.model.loc_size),dim=0
        x_last = torch.normal(xtmean, xtstd, size=shape).to(self.device)
        return x_last

    @torch.no_grad()
    def sampling(self, x: torch.Tensor, t: torch.Tensor, condtion: torch.Tensor, temporal_condition: torch.Tensor, step: int,temperature=1.0,):
        e_t = self.get_uncertainty(x, t,condtion,temporal_condition)
        if t[0] == 0:
            temperature = 0.
        bs = x.shape[0]
        sqrt_recip_alpha_bar = x.new_full((bs, 1, 1), self.sqrt_recip_alpha_bar[step])
        sqrt_recip_m1_alpha_bar = x.new_full((bs, 1, 1), self.sqrt_recip_m1_alpha_bar[step])
        x0 = sqrt_recip_alpha_bar * x - sqrt_recip_m1_alpha_bar * e_t
        mean_x0_coef = x.new_full((bs, 1, 1), self.mean_x0_coef[step])
        mean_xt_coef = x.new_full((bs, 1, 1), self.mean_xt_coef[step])
        mean = mean_x0_coef * x0 + mean_xt_coef * x
        log_var = x.new_full((bs, 1, 1), self.log_var[step])
        noise = torch.randn(x.shape).to(self.device)
        noise = noise * temperature
        x_prev = mean + (0.5 * log_var).exp() * noise
        return x_prev, x0, e_t

    @torch.no_grad()
    def sampling_process(self,shape,x_last,temperature: float = 1.):
        x = x_last
        time_steps = np.flip(self.ddpm_time_steps)
        for i, step in zip(range(len(time_steps)), time_steps):
            ts = x.new_full((shape[0],), step, dtype=torch.long,device=self.device)
            x, pred_x0, e_t = self.sampling(x, ts, step, temperature=temperature)
        return x

    @torch.no_grad()
    def single_step_sample(self, latent, home_locations, poi_condition, trav_pattern):
        """
        简化采样：仅在 t=T 添加噪声，然后一步预测回 x0。
        """
        x0 = latent  # 或者 encoder(latent) 看你的结构
        t = torch.full((x0.size(0),), self.full_n_steps - 1, device=self.device, dtype=torch.long)
        
        noise = torch.randn_like(x0)
        xt = self.diffusion_process(x0, t, noise)  # 添加噪声

        pred_x0 = self.get_pred_x0(xt, t, home_locations, poi_condition, trav_pattern)
        return pred_x0


    @torch.no_grad()
    def TrajGenerating(self, num_samples=16 ,home_locations:torch.Tensor=None,poi_condition:torch.Tensor=None,trav_pattern: Optional[torch.Tensor] = None):
        batch_size = num_samples
        shape = [batch_size, self.model.input_len, self.model.loc_size]

        x_last=torch.randn(shape, device=self.device)

        x0 = self.sampling_process_ddim(shape=shape, temperature=1, x_last=x_last,home_locations=home_locations,poi_condition=poi_condition,trav_pattern=trav_pattern)  # starts

        return x0
    
    @torch.no_grad()
    def sampling_process_ddim(self, shape, x_last, home_locations, poi_condition,  ddim_steps=50, temperature: float = 1., eta=0.2,trav_pattern: Optional[torch.Tensor] = None):
        x = x_last
        
        # 使用局部变量而不是类成员变量
        time_steps = np.asarray(list(range(0, self.full_n_steps + 1, self.full_n_steps // ddim_steps)))
        time_steps[-1] = time_steps[-1] - 1
        
        beta = torch.linspace(
            self.linear_start ** 0.5, 
            self.linear_end ** 0.5, 
            self.full_n_steps,
            dtype=torch.float32, 
            device=self.device
        ) ** 2
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        
        ddim_alpha = alpha_bar[time_steps].clone().to(torch.float32)
        ddim_alpha_sqrt = torch.sqrt(ddim_alpha)
        ddim_alpha_prev = torch.cat([alpha_bar[0:1], alpha_bar[time_steps[:-1]]])
        ddim_sigma = (eta * ((1 - ddim_alpha_prev) / (1 - ddim_alpha) * 
                        (1 - ddim_alpha / ddim_alpha_prev)) ** .5)
        ddim_sqrt_one_minus_alpha = (1. - ddim_alpha) ** .5
        
        # 修改 ddim_sampling 方法以接受这些参数
        time_steps = np.flip(time_steps)
        for i in range(len(time_steps)):
            index = len(time_steps) - i - 1
            ts = x.new_full((shape[0],), time_steps[i], dtype=torch.long)
            
            x, pred_x0, uncertainty = self.ddim_sampling(
                x=x, 
                t=ts, 
                home_locations=home_locations,
                poi_condition=poi_condition,
                index=index,
                temperature=temperature,
                trav_pattern=trav_pattern,
                ddim_params={
                    'ddim_alpha': ddim_alpha,
                    'ddim_alpha_prev': ddim_alpha_prev,
                    'ddim_sigma': ddim_sigma,
                    'ddim_sqrt_one_minus_alpha': ddim_sqrt_one_minus_alpha
                }
            )
        return x

    def ddim_sampling(self, x: torch.Tensor, t: torch.Tensor, home_locations: torch.Tensor,poi_condition: torch.Tensor, index: int, *, 
                 temperature: float = 1., ddim_params: dict = None,trav_pattern: Optional[torch.Tensor] = None):
        """
        Args:
            x: 输入张量
            t: 时间步
            index: 索引
            temperature: 温度参数
            ddim_params: DDIM采样参数字典，包含所需的alpha等参数
        """
        pred_x0= self.get_pred_x0(x, t,home_locations,poi_condition,trav_pattern)

        alpha = ddim_params['ddim_alpha'][index]
        alpha_prev = ddim_params['ddim_alpha_prev'][index]
        sigma = ddim_params['ddim_sigma'][index]
        sqrt_one_minus_alpha = ddim_params['ddim_sqrt_one_minus_alpha'][index]

        if index == 0:
            temperature = 0
        uncertainty=(x - alpha.sqrt() * pred_x0) / (1-alpha).sqrt().clamp(min = 1e-8)

        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * uncertainty

        if sigma == 0.:
            noise = 0.
        else:
            noise = torch.randn(x.shape, device=x.device)

        noise = noise * temperature
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise

        return x_prev, pred_x0, uncertainty

