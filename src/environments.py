import torch
import numpy as np
import os.path as osp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def random_ipd_batched(bs, gamma_inner=0.96, batch_size=4096):
    # https://en.wikipedia.org/wiki/Prisoner%27s_dilemma#Generalized_form
    dims = [5, 5]
    rand = lambda x,y: np.random.uniform(x,y)

    def pay():
        S,P,R,T = 0,0,0,0
        while 2*R <= T+S: # Condition for IPD, alongside T>R>P>S
            T = rand(-2, 0)
            R = rand(-4, T)
            P = rand(-6, R)
            S = rand(-8, P)

            return torch.Tensor([[R, S], [T, P]]).to(device)

    payout_mat_1 = torch.cat([pay() for _ in range(bs)], axis=-1)
    payout_mat_1 = payout_mat_1.reshape((bs, 2, 2))
    payout_mat_2 = payout_mat_1.mT
    

    def Ls(th):  # th is a list of two different tensors. First one is first agent? tensor size is List[Tensor(bs, 5), Tensor(bs,5)].
        p_1_0 = torch.sigmoid(th[0][:, 0:1])
        p_2_0 = torch.sigmoid(th[1][:, 0:1])
        p = torch.cat([p_1_0 * p_2_0, p_1_0 * (1 - p_2_0), (1 - p_1_0) * p_2_0, (1 - p_1_0) * (1 - p_2_0)], dim=-1)
        p_1 = torch.reshape(torch.sigmoid(th[0][:, 1:5]), (bs, 4, 1))
        p_2 = torch.reshape(torch.sigmoid(torch.cat([th[1][:, 1:2], th[1][:, 3:4], th[1][:, 2:3], th[1][:, 4:5]], dim=-1)), (bs, 4, 1))
        P = torch.cat([p_1 * p_2, p_1 * (1 - p_2), (1 - p_1) * p_2, (1 - p_1) * (1 - p_2)], dim=-1)

        M = torch.matmul(p.unsqueeze(1), torch.inverse(torch.eye(4).to(device) - gamma_inner * P))
        L_1 = -torch.matmul(M, torch.reshape(payout_mat_1, (bs, 4, 1)))
        L_2 = -torch.matmul(M, torch.reshape(payout_mat_2, (bs, 4, 1)))

        return [L_1.squeeze(-1), L_2.squeeze(-1), M]
    
    payout = torch.cat([torch.cat([payout_mat_1[i], payout_mat_2[i]]) for i in range(bs)])
    payout = torch.reshape(payout, (bs,1,8))
    #payout = torch.sigmoid(payout)

    return dims, Ls, payout


def random_batched(bs, gamma_inner=0.96, batch_size=4096):
    dims = [5, 5]
    rand = lambda x,y: round(np.random.uniform(x,y))

    payout_mat_1 = torch.cat([torch.Tensor([[rand(-10,0), rand(-10,0)], 
                                 [rand(-10,0), rand(-10,0)]]).to(device) for _ in range(bs)], axis=-1)
    payout_mat_1 = payout_mat_1.reshape((bs, 2, 2))
    #print(payout_mat_1) # Test to see if main_mfos_ppo.py creates a new environment
    payout_mat_2 = payout_mat_1.mT

    def Ls(th):  # th is a list of two different tensors. First one is first agent? tensor size is List[Tensor(bs, 5), Tensor(bs,5)].
        p_1_0 = torch.sigmoid(th[0][:, 0:1])
        p_2_0 = torch.sigmoid(th[1][:, 0:1])
        p = torch.cat([p_1_0 * p_2_0, p_1_0 * (1 - p_2_0), (1 - p_1_0) * p_2_0, (1 - p_1_0) * (1 - p_2_0)], dim=-1)
        p_1 = torch.reshape(torch.sigmoid(th[0][:, 1:5]), (bs, 4, 1))
        p_2 = torch.reshape(torch.sigmoid(torch.cat([th[1][:, 1:2], th[1][:, 3:4], th[1][:, 2:3], th[1][:, 4:5]], dim=-1)), (bs, 4, 1))
        P = torch.cat([p_1 * p_2, p_1 * (1 - p_2), (1 - p_1) * p_2, (1 - p_1) * (1 - p_2)], dim=-1)

        M = torch.matmul(p.unsqueeze(1), torch.inverse(torch.eye(4).to(device) - gamma_inner * P))
        L_1 = -torch.matmul(M, torch.reshape(payout_mat_1, (bs, 4, 1)))
        L_2 = -torch.matmul(M, torch.reshape(payout_mat_2, (bs, 4, 1)))

        return [L_1.squeeze(-1), L_2.squeeze(-1), M]

    payout = torch.cat([torch.cat([payout_mat_1[i], payout_mat_2[i]]) for i in range(bs)])
    payout = torch.reshape(payout, (bs,1,8))

    return dims, Ls, payout # Need a try, except when calling environments


def extreme_ipd_batched(bs, gamma_inner=0.96, batch_size=4096):
    dims = [5, 5]
    payout_mat_1 = torch.Tensor([[-1, -8], [0, -5]]).to(device)
    payout_mat_2 = payout_mat_1.T
    payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)
    payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)

    def Ls(th):  # th is a list of two different tensors. First one is first agent? tnesor size is List[Tensor(bs, 5), Tensor(bs,5)].
        p_1_0 = torch.sigmoid(th[0][:, 0:1])
        p_2_0 = torch.sigmoid(th[1][:, 0:1])
        p = torch.cat([p_1_0 * p_2_0, p_1_0 * (1 - p_2_0), (1 - p_1_0) * p_2_0, (1 - p_1_0) * (1 - p_2_0)], dim=-1)
        p_1 = torch.reshape(torch.sigmoid(th[0][:, 1:5]), (bs, 4, 1))
        p_2 = torch.reshape(torch.sigmoid(torch.cat([th[1][:, 1:2], th[1][:, 3:4], th[1][:, 2:3], th[1][:, 4:5]], dim=-1)), (bs, 4, 1))
        P = torch.cat([p_1 * p_2, p_1 * (1 - p_2), (1 - p_1) * p_2, (1 - p_1) * (1 - p_2)], dim=-1)

        M = torch.matmul(p.unsqueeze(1), torch.inverse(torch.eye(4).to(device) - gamma_inner * P))
        L_1 = -torch.matmul(M, torch.reshape(payout_mat_1, (bs, 4, 1)))
        L_2 = -torch.matmul(M, torch.reshape(payout_mat_2, (bs, 4, 1)))

        return [L_1.squeeze(-1), L_2.squeeze(-1), M]

    payout = torch.flatten(payout_mat_1[0]).repeat((batch_size,2))
    #payout = torch.sigmoid(payout)

    #payout = torch.reshape(torch.stack((payout_mat_1[0], payout_mat_2[0])), (1,2,4))

    return dims, Ls, payout


def noisy_ipd_batched(bs, gamma_inner=0.96, batch_size=4096, noise=0.5):
    dims = [5, 5]
    payout_mat_1 = torch.Tensor([[-1, -3], [0, -2]]).to(device)
    payout_mat_2 = payout_mat_1.T
    payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)
    payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)

    payout_noise = noise * torch.randn(size=(bs, 2, 2)).to(device)
    payout_mat_1 = payout_mat_1 + payout_noise
    payout_mat_2 = payout_mat_2 + payout_noise
    #print(payout_mat_1)

    def Ls(th):  # th is a list of two different tensors. First one is first agent? tnesor size is List[Tensor(bs, 5), Tensor(bs,5)].
        p_1_0 = torch.sigmoid(th[0][:, 0:1])
        p_2_0 = torch.sigmoid(th[1][:, 0:1])
        p = torch.cat([p_1_0 * p_2_0, p_1_0 * (1 - p_2_0), (1 - p_1_0) * p_2_0, (1 - p_1_0) * (1 - p_2_0)], dim=-1)
        p_1 = torch.reshape(torch.sigmoid(th[0][:, 1:5]), (bs, 4, 1))
        p_2 = torch.reshape(torch.sigmoid(torch.cat([th[1][:, 1:2], th[1][:, 3:4], th[1][:, 2:3], th[1][:, 4:5]], dim=-1)), (bs, 4, 1))
        P = torch.cat([p_1 * p_2, p_1 * (1 - p_2), (1 - p_1) * p_2, (1 - p_1) * (1 - p_2)], dim=-1)

        M = torch.matmul(p.unsqueeze(1), torch.inverse(torch.eye(4).to(device) - gamma_inner * P))
        L_1 = -torch.matmul(M, torch.reshape(payout_mat_1, (bs, 4, 1)))
        L_2 = -torch.matmul(M, torch.reshape(payout_mat_2, (bs, 4, 1)))

        return [L_1.squeeze(-1), L_2.squeeze(-1), M]

    payout = torch.cat([torch.cat([payout_mat_1[i], payout_mat_2[i]]) for i in range(bs)])
    payout = torch.reshape(payout, (bs,1,8))

    return dims, Ls, payout


def ipd_batched(bs, gamma_inner=0.96, batch_size=4096):
    dims = [5, 5]
    payout_mat_1 = torch.Tensor([[-1, -3], [0, -2]]).to(device)
    payout_mat_2 = payout_mat_1.T
    payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)
    payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)

    def Ls(th):  # th is a list of two different tensors. First one is first agent? tnesor size is List[Tensor(bs, 5), Tensor(bs,5)].
        p_1_0 = torch.sigmoid(th[0][:, 0:1])
        p_2_0 = torch.sigmoid(th[1][:, 0:1])
        p = torch.cat([p_1_0 * p_2_0, p_1_0 * (1 - p_2_0), (1 - p_1_0) * p_2_0, (1 - p_1_0) * (1 - p_2_0)], dim=-1)
        p_1 = torch.reshape(torch.sigmoid(th[0][:, 1:5]), (bs, 4, 1))
        p_2 = torch.reshape(torch.sigmoid(torch.cat([th[1][:, 1:2], th[1][:, 3:4], th[1][:, 2:3], th[1][:, 4:5]], dim=-1)), (bs, 4, 1))
        P = torch.cat([p_1 * p_2, p_1 * (1 - p_2), (1 - p_1) * p_2, (1 - p_1) * (1 - p_2)], dim=-1)

        M = torch.matmul(p.unsqueeze(1), torch.inverse(torch.eye(4).to(device) - gamma_inner * P))
        L_1 = -torch.matmul(M, torch.reshape(payout_mat_1, (bs, 4, 1)))
        L_2 = -torch.matmul(M, torch.reshape(payout_mat_2, (bs, 4, 1)))

        return [L_1.squeeze(-1), L_2.squeeze(-1), M]

    payout = torch.cat([torch.cat([payout_mat_1[i], payout_mat_2[i]]) for i in range(bs)])
    payout = torch.reshape(payout, (bs,1,8))
    #payout = torch.sigmoid(payout)

    return dims, Ls, payout


def imp_batched(bs, gamma_inner=0.96):
    dims = [5, 5]
    payout_mat_1 = torch.Tensor([[-1, 1], [1, -1]]).to(device)
    payout_mat_2 = -payout_mat_1
    payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)
    payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(bs, 1, 1).to(device)

    def Ls(th):  # th is a list of two different tensors. First one is first agent? tnesor size is List[Tensor(bs, 5), Tensor(bs,5)].
        p_1_0 = torch.sigmoid(th[0][:, 0:1])
        p_2_0 = torch.sigmoid(th[1][:, 0:1])
        p = torch.cat([p_1_0 * p_2_0, p_1_0 * (1 - p_2_0), (1 - p_1_0) * p_2_0, (1 - p_1_0) * (1 - p_2_0)], dim=-1)
        p_1 = torch.reshape(torch.sigmoid(th[0][:, 1:5]), (bs, 4, 1))
        p_2 = torch.reshape(torch.sigmoid(torch.cat([th[1][:, 1:2], th[1][:, 3:4], th[1][:, 2:3], th[1][:, 4:5]], dim=-1)), (bs, 4, 1))
        P = torch.cat([p_1 * p_2, p_1 * (1 - p_2), (1 - p_1) * p_2, (1 - p_1) * (1 - p_2)], dim=-1)

        M = torch.matmul(p.unsqueeze(1), torch.inverse(torch.eye(4).to(device) - gamma_inner * P))
        L_1 = -torch.matmul(M, torch.reshape(payout_mat_1, (bs, 4, 1)))
        L_2 = -torch.matmul(M, torch.reshape(payout_mat_2, (bs, 4, 1)))

        return [L_1.squeeze(-1), L_2.squeeze(-1), M]

    return dims, Ls


def get_gradient(function, param):
    grad = torch.autograd.grad(function, param, create_graph=True, allow_unused=True)[0]
    return grad


def compute_best_response(outer_th_ba):
    batch_size = 1
    std = 0
    num_steps = 1000
    lr = 1

    ipd_batched_env = ipd_batched(batch_size, gamma_inner=0.96)[1]
    inner_th_ba = torch.nn.init.normal_(torch.empty((batch_size, 5), requires_grad=True), std=std).cuda()
    for i in range(num_steps):
        th_ba = [inner_th_ba, outer_th_ba.detach()]
        l1, l2, M = ipd_batched_env(th_ba)
        grad = get_gradient(l1.sum(), inner_th_ba)
        with torch.no_grad():
            inner_th_ba -= grad * lr
    print(l1.mean() * (1 - 0.96))
    return inner_th_ba


def matching_pennies_batch(batch_size=128):
    dims = [1, 1]
    payout_mat_1 = torch.Tensor([[1, -1], [-1, 1]]).to(device)
    payout_mat_2 = -payout_mat_1
    payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(batch_size, 1, 1)
    payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(batch_size, 1, 1)

    def Ls(th):
        p_1, p_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])
        x, y = torch.cat([p_1, 1 - p_1], dim=-1), torch.cat([p_2, 1 - p_2], dim=-1)
        L_1 = torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_1), y.unsqueeze(-1))
        L_2 = torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_2), y.unsqueeze(-1))
        return [L_1.squeeze(-1), L_2.squeeze(-1)]

    return dims, Ls


def chicken_game_batch(batch_size=128):
    dims = [1, 1]
    payout_mat_1 = torch.Tensor([[0, -1], [1, -100]]).to(device)
    payout_mat_2 = torch.Tensor([[0, 1], [-1, -100]]).to(device)
    payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(batch_size, 1, 1)
    payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(batch_size, 1, 1)

    def Ls(th):
        p_1, p_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])
        x, y = torch.cat([p_1, 1 - p_1], dim=-1), torch.cat([p_2, 1 - p_2], dim=-1)
        L_1 = -torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_1), y.unsqueeze(-1))
        L_2 = -torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_2), y.unsqueeze(-1))
        return [L_1.squeeze(-1), L_2.squeeze(-1), None]

    return dims, Ls


def generate_mamaml(b, d, inner_env, game, inner_lr=1):
    """
    This is an improved version of the algorithm presented in this paper:
    https://arxiv.org/pdf/2011.00382.pdf
    Rather than calculating the loss using multiple policy gradients terms,
    this approach instead directly takes all of the gradients through because the environment is differentiable.
    """
    outer_lr = 0.01
    mamaml = torch.nn.init.normal_(torch.empty((1, d), requires_grad=True, device=device), std=1.0)
    alpha = torch.rand(1, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([mamaml, alpha], lr=outer_lr)

    for ep in range(1000):
        agent = mamaml.clone().repeat(b, 1)
        opp = torch.nn.init.normal_(torch.empty((b, d), requires_grad=True), std=1.0).cuda()
        total_agent_loss = 0
        total_opp_loss = 0
        for step in range(100):
            l1, l2, M = inner_env([opp, agent])
            total_agent_loss = total_agent_loss + l2.sum()
            total_opp_loss = total_opp_loss + l1.sum()

            opp_grad = get_gradient(l1.sum(), opp)
            agent_grad = get_gradient(l2.sum(), agent)
            opp = opp - opp_grad * inner_lr
            agent = agent - agent_grad * alpha

        optimizer.zero_grad()
        total_agent_loss.sum().backward()
        optimizer.step()
        print(total_agent_loss.sum().item())

    torch.save((mamaml, alpha), f"mamaml_{game}.th")


class MetaGames:
    def __init__(self, b, opponent="NL", game="IPD", mmapg_id=0, payout=None, opp_lr=1, 
                 rand_opp=False):
        """
        Opponent can be:
        NL = Naive Learner (gradient updates through environment).
        LOLA = Gradient through NL.
        STATIC = Doesn't learn.
        COPYCAT = Copies what opponent played last step.
        """
        self.gamma_inner = 0.96
        self.b = b

        self.game = game
        if self.game == "IPD":
            d, self.game_batched, self.payout = ipd_batched(b, gamma_inner=self.gamma_inner,
                                                             batch_size=self.b)
            self.std = 1
            self.lr = opp_lr
        elif self.game == "IMP":
            d, self.game_batched = imp_batched(b, gamma_inner=self.gamma_inner)
            self.std = 1
            self.lr = opp_lr
        elif self.game == "chicken":
            d, self.game_batched = chicken_game_batch(b)
            self.std = 1
            self.lr = opp_lr
        elif self.game == "extremeIPD":
            d, self.game_batched, self.payout = extreme_ipd_batched(b, gamma_inner=self.gamma_inner,
                                                             batch_size=self.b)
            self.std = 1
            self.lr = opp_lr
        elif self.game == "noisyIPD":
            d, self.game_batched, self.payout = noisy_ipd_batched(b, gamma_inner=self.gamma_inner,
                                                                batch_size=self.b)
            self.std = 1
            self.lr = opp_lr
        elif self.game == "random":
            d, self.game_batched, self.payout = random_batched(b, batch_size=self.b)
            self.std = 1
            self.lr = opp_lr
        elif self.game == "randIPD":
            d, self.game_batched, self.payout = random_ipd_batched(b, batch_size=self.b)
            self.std = 1
            self.lr = opp_lr
        else:
            raise NotImplementedError
        self.d = d[0]

        self.opponent = opponent
        if self.opponent == "MAMAML":
            f = f"data/mamaml_{self.game}_{mmapg_id}.th"
            assert osp.exists(f), "Generate the MAMAML weights first"
            self.init_th_ba = torch.load(f)
        else:
            self.init_th_ba = None

        self.rand_opp = rand_opp
        if self.rand_opp == True: # self.lr to sample a new lr each meta-episode
            self.lr = torch.reshape(torch.tensor(np.array(np.random.uniform(0.1, 2.5, size=b))), (b,1)).to(device)
            self.lr = self.lr.repeat((1, 5))

    def reset(self, info=False):
        if self.game == 'random':
            d, self.game_batched, self.payout = random_batched(self.b, batch_size=self.b) # Reset random matrix
        elif self.game == 'randIPD':
            d, self.game_batched, self.payout = random_ipd_batched(self.b, batch_size=self.b) # Reset random matrix
        elif self.game == 'noisyIPD':
            d, self.game_batched, self.payout = noisy_ipd_batched(self.b, batch_size=self.b)
        if self.init_th_ba is not None:
            self.inner_th_ba = self.init_th_ba.detach() * torch.ones((self.b, self.d), requires_grad=True).to(device)
        else:
            self.inner_th_ba = torch.nn.init.normal_(torch.empty((self.b, self.d), requires_grad=True), std=self.std).to(device)
        outer_th_ba = torch.nn.init.normal_(torch.empty((self.b, self.d), requires_grad=True), std=self.std).to(device)
        
        try:
            state, _, _, M, self.payout = self.step(outer_th_ba)
        except ValueError:
            state, _, _, M = self.step(outer_th_ba)

        if self.rand_opp == True: # self.lr to sample a new lr each meta-episode (element-wise multiplication)
            self.lr = torch.reshape(torch.tensor(np.array(np.random.uniform(0.1, 2.5, size=self.b))), (self.b,1)).to(device)
            self.lr = self.lr.repeat((1, 5))

        if info:
            try:
                return state, M, self.payout
            except ValueError:
                return state, M
            except AttributeError:
                return state, M
        else:
            try:
                return state, self.payout
            except ValueError:
                return state
            except AttributeError:
                return state

    def step(self, outer_th_ba):
        last_inner_th_ba = self.inner_th_ba.detach().clone()
        if self.opponent == "NL" or self.opponent == "MAMAML":
            th_ba = [self.inner_th_ba, outer_th_ba.detach()]
            l1, l2, M = self.game_batched(th_ba)
            grad = get_gradient(l1.sum(), self.inner_th_ba)
            with torch.no_grad():
                self.inner_th_ba -= grad * self.lr
        elif self.opponent == "LOLA":
            th_ba = [self.inner_th_ba, outer_th_ba.detach()]
            th_ba[1].requires_grad = True
            l1, l2, M = self.game_batched(th_ba)
            losses = [l1, l2]
            grad_L = [[get_gradient(losses[j].sum(), th_ba[i]) for j in range(2)] for i in range(2)]
            term = (grad_L[1][0] * grad_L[1][1]).sum()
            grad = grad_L[0][0] - self.lr * get_gradient(term, th_ba[0])
            with torch.no_grad():
                self.inner_th_ba -= grad * self.lr
        elif self.opponent == "BR":
            num_steps = 1000
            inner_th_ba = torch.nn.init.normal_(torch.empty((self.b, 5), requires_grad=True), std=self.std).to(device)
            for i in range(num_steps):
                th_ba = [inner_th_ba, outer_th_ba.detach()]
                l1, l2, M = self.game_batched(th_ba)
                grad = get_gradient(l1.sum(), inner_th_ba)
                with torch.no_grad():
                    inner_th_ba -= grad * self.lr
            with torch.no_grad():
                self.inner_th_ba = inner_th_ba
                th_ba = [self.inner_th_ba, outer_th_ba.detach()]
                l1, l2, M = self.game_batched(th_ba)
        elif self.opponent == "STATIC":
            self.inner_th_ba = inner_th_ba
            th_ba = [inner_th_ba, outer_th_ba.detach()]
            l1, l2, M = self.game_batched(th_ba)
        else:
            raise NotImplementedError

        if self.game == "IPD" or self.game == "IMP" or self.game=='extremeIPD' or self.game=='noisyIPD' or self.game == 'randIPD' or self.game=='random':
            return torch.sigmoid(torch.cat((outer_th_ba, last_inner_th_ba), dim=-1)).detach(), (-l2 * (1 - self.gamma_inner)).detach(), (-l1 * (1 - self.gamma_inner)).detach(), M
        else:
            return torch.sigmoid(torch.cat((outer_th_ba, last_inner_th_ba), dim=-1)).detach(), -l2.detach(), -l1.detach(), M


class SymmetricMetaGames:
    def __init__(self, b, game="IPD"):
        self.gamma_inner = 0.96

        self.b = b
        self.game = game
        if self.game == "IPD" or self.game == 'inputIPD':
            d, self.game_batched = ipd_batched(b, gamma_inner=self.gamma_inner)
            self.std = 1
        elif self.game == "IMP":
            d, self.game_batched = imp_batched(b, gamma_inner=self.gamma_inner)
            self.std = 1
        elif self.game == "chicken":
            d, self.game_batched = chicken_game_batch(b)
            self.std = 1
        elif self.game == "random":
            d, self.game_batched = random_batched(b)
            self.std = 1
        elif self.game == "randIPD":
            d, self.game_batched = random_ipd_batched(b)
            self.std = 1
        else:
            raise NotImplementedError

        self.d = d[0]

    def reset(self, info=False):
        p_ba_0 = torch.nn.init.normal_(torch.empty((self.b, self.d)), std=self.std).to(device)
        p_ba_1 = torch.nn.init.normal_(torch.empty((self.b, self.d)), std=self.std).to(device)
        state_0 = torch.sigmoid(torch.cat((p_ba_0.detach(), p_ba_1.detach()), dim=-1))
        state_1 = torch.sigmoid(torch.cat((p_ba_1.detach(), p_ba_0.detach()), dim=-1))

        if info:
            state, _, M = self.step(p_ba_0, p_ba_1)
            return state, M
        else:
            return [state_0, state_1]

    def step(self, p_ba_0, p_ba_1):
        th_ba = [p_ba_0.detach(), p_ba_1.detach()]
        l1, l2, M = self.game_batched(th_ba)
        state_0 = torch.sigmoid(torch.cat((p_ba_0.detach(), p_ba_1.detach()), dim=-1))
        state_1 = torch.sigmoid(torch.cat((p_ba_1.detach(), p_ba_0.detach()), dim=-1))

        if self.game == 'inputIPD':
            state_0 = torch.cat(torch.sigmoid(torch.cat((p_ba_0.detach(), p_ba_1.detach()), dim=-1)),
                                torch.Tensor([[-1, -3], [0, -2]]).to(device), axis=-1)
            state_1 = torch.cat(torch.sigmoid(torch.cat((p_ba_1.detach(), p_ba_0.detach()), dim=-1)),
                                torch.Tensor([[-1, -3], [0, -2]]).to(device), axis=-1)           

        if self.game == "IPD" or self.game == "IMP" or self.game == 'inputIPD':
            return [state_0, state_1], [-l1 * (1 - self.gamma_inner), -l2 * (1 - self.gamma_inner)], M
        else:
            return [state_0, state_1], [-l1.detach(), -l2.detach()], M


class NonMfosMetaGames:
    def __init__(self, b, lr=1, opp_lr=1, p1="NL", p2="NL", game="IPD", mmapg_id=None):
        """
        Opponent can be:
        NL = Naive Learner (gradient updates through environment).
        LOLA = Gradient through NL.
        STATIC = Doesn't learn. Used for sanity checking.
        """
        self.gamma_inner = 0.96
        self.b = b

        self.p1 = p1
        self.p2 = p2
        self.game = game
        if self.game == "IPD":
            d, self.game_batched, self.payout = ipd_batched(b, gamma_inner=self.gamma_inner,
                                                             batch_size=self.b)
            self.std = 1
            self.lr = lr
            self.opp_lr = opp_lr
        elif self.game == "IMP":
            d, self.game_batched = imp_batched(b, gamma_inner=self.gamma_inner)
            self.std = 1
            self.lr = lr
            self.opp_lr = opp_lr
        elif self.game == "chicken":
            d, self.game_batched = chicken_game_batch(b)
            self.std = 1
            self.lr = lr
            self.opp_lr = opp_lr
        elif self.game == "extremeIPD":
            d, self.game_batched, self.payout = extreme_ipd_batched(b, gamma_inner=self.gamma_inner,
                                                             batch_size=self.b)
            self.std = 1
            self.lr = lr
            self.opp_lr = opp_lr
        elif self.game == "noisyIPD":
            d, self.game_batched, self.payout = noisy_ipd_batched(b, gamma_inner=self.gamma_inner,
                                                                batch_size=self.b)
            self.std = 1
            self.lr = lr
            self.opp_lr = opp_lr
        elif self.game == "random":
            d, self.game_batched, self.payout = random_batched(b, batch_size=self.b)
            self.std = 1
            self.lr = lr
            self.opp_lr = opp_lr
        elif self.game == "randIPD":
            d, self.game_batched, self.payout = random_ipd_batched(b, batch_size=self.b)
            self.std = 1
            self.lr = lr
            self.opp_lr = opp_lr
        else:
            raise NotImplementedError

        self.d = d[0]

        self.init_th_ba = None
        if self.p1 == "MAMAML" or self.p2 == "MAMAML":
            if self.init_th_ba is not None:
                raise NotImplementedError
            f = f"data/mamaml_{self.game}_{mmapg_id}.th"
            assert osp.exists(f), "Generate the MAMAML weights first"
            # print(f"GENERATING MAPG WEIGHTS TO {f}")
            # generate_meta_mapg(self.b, self.d, self.game_batched, self.game, inner_lr=self.lr)
            self.init_th_ba = torch.load(f)
            print(self.init_th_ba)

    def reset(self, p1_th_ba=None, p2_th_ba=None, info=False):
        if self.game == 'random':
            d, self.game_batched, self.payout = random_batched(self.b, batch_size=self.b) # Reset random matrix
        elif self.game == 'randIPD':
            d, self.game_batched, self.payout = random_ipd_batched(self.b, batch_size=self.b) # Reset random matrix
        elif self.game == 'noisyIPD':
            d, self.game_batched, self.payout = noisy_ipd_batched(self.b, batch_size=self.b)

        if p1_th_ba != None and p2_th_ba != None:
            self.p1_th_ba = p1_th_ba
            self.p2_th_ba = p2_th_ba
        else:
            self.p1_th_ba = torch.nn.init.normal_(torch.empty((self.b, self.d), requires_grad=True), std=self.std).to(device)
            self.p2_th_ba = torch.nn.init.normal_(torch.empty((self.b, self.d), requires_grad=True), std=self.std).to(device)

        if self.p1 == "MAMAML":
            self.p1_th_ba = self.init_th_ba.detach() * torch.ones((self.b, self.d), requires_grad=True).to(device)
        if self.p2 == "MAMAML":
            self.p2_th_ba = self.init_th_ba.detach() * torch.ones((self.b, self.d), requires_grad=True).to(device)

        state, _, _, M = self.step()
        if info:
            return state, M
        
        try:
            state, _, _, M, self.payout = self.step()
        except ValueError:
            state, _, _, M = self.step()

        if info:
            try:
                return state, M, self.payout
            except ValueError or AttributeError:
                return state, M
        else:
            try:
                return state, self.payout
            except ValueError or AttributeError:
                return state

        return None

    def step(self, info=False):
        if self.game == 'random':
            d, self.game_batched, self.payout = random_batched(self.b, batch_size=self.b) # Reset random matrix
        elif self.game == 'randIPD':
            d, self.game_batched, self.payout = random_ipd_batched(self.b, batch_size=self.b) # Reset random matrix
        elif self.game == 'noisyIPD':
            d, self.game_batched, self.payout = noisy_ipd_batched(self.b, batch_size=self.b)
            
        last_p1_th_ba = self.p1_th_ba.clone()
        last_p2_th_ba = self.p2_th_ba.clone()
        th_ba = [self.p2_th_ba, self.p1_th_ba]
        l1, l2, M = self.game_batched(th_ba)

        # UPDATE P1
        if self.p1 == "NL" or self.p1 == "MAMAML":
            grad = get_gradient(l2.sum(), self.p1_th_ba)
            with torch.no_grad():
                self.p1_th_ba -= grad * self.lr
        elif self.p1 == "LOLA":
            losses = [l1, l2]
            grad_L = [[get_gradient(losses[j].sum(), th_ba[i]) for j in range(2)] for i in range(2)]
            term = (grad_L[0][0] * grad_L[0][1]).sum()
            grad = grad_L[1][1] - self.lr * get_gradient(term, th_ba[1])
            with torch.no_grad():
                self.p1_th_ba -= grad * self.lr
        elif self.p1 == "STATIC":
            pass
        else:
            raise NotImplementedError

        # UPDATE P2
        if self.p2 == "NL" or self.p2 == "MAMAML":
            grad = get_gradient(l1.sum(), self.p2_th_ba)
            with torch.no_grad():
                self.p2_th_ba -= grad * self.opp_lr
        elif self.p2 == "LOLA":
            losses = [l1, l2]
            grad_L = [[get_gradient(losses[j].sum(), th_ba[i]) for j in range(2)] for i in range(2)]
            term = (grad_L[1][0] * grad_L[1][1]).sum()
            grad = grad_L[0][0] - self.opp_lr * get_gradient(term, th_ba[0])
            with torch.no_grad():
                self.p2_th_ba -= grad * self.opp_lr
        elif self.p2 == "STATIC":
            pass
        else:
            raise NotImplementedError

        if self.game == "IPD" or self.game == "IMP" or self.game=='extremeIPD' or self.game=='noisyIPD' or self.game == 'randIPD' or self.game=='random':
            return torch.sigmoid(torch.cat([last_p1_th_ba, last_p2_th_ba])), -l2 * (1 - self.gamma_inner), -l1 * (1 - self.gamma_inner), M
        else:
            return torch.sigmoid(torch.cat([last_p1_th_ba, last_p2_th_ba])), -l2.detach(), -l1.detach(), M
