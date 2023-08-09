import torch
from families import ALL_GAMES, WIN_WIN, HARMONIOUS, STAG_HUNTS, BIASED 
from families import BATTLE, SELF_SERVING, ALTRUISTIC, SECOND_BEST, UNFAIR, WINNER, LOSER, PD_FAMILY
from families import TRAGIC, PD, ALIBI, CYCLIC, ALL_GROUPS, SYMMETRIC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NormalFormMG:
    def __init__(self, batch_size):
        self.num_games = 144
        self.batch_size = batch_size
        self.player1_payouts = []
        self.all_groups = ALL_GROUPS
        self.games = ALL_GAMES
        self.win_win = WIN_WIN
        self.harmonious = HARMONIOUS
        self.stag_hunts = STAG_HUNTS
        self.biased = BIASED
        self.battle = BATTLE
        self.self_serving = SELF_SERVING
        self.altruistic = ALTRUISTIC
        self.second_best = SECOND_BEST
        self.unfair = UNFAIR
        self.winner = WINNER
        self.loser = LOSER
        self.pd_family = PD_FAMILY
        self.tragic = TRAGIC
        self.PD = PD
        self.alibi = ALIBI
        self.cyclic = CYCLIC
        self.symmetric = SYMMETRIC

    def get_game_list(self):
        return self.games.keys()

    def create_payout(self, player1, player2, one, two, three, four):
        self.player1_payouts = [
            torch.Tensor([[three, two],[four, one]]).to(device),
            torch.Tensor([[two, three],[four, one]]).to(device),
            torch.Tensor([[one, three],[four, two]]).to(device),
            torch.Tensor([[one, two],[four, three]]).to(device),
            torch.Tensor([[two, one],[four, three]]).to(device),
            torch.Tensor([[three, one],[four, two]]).to(device),
            torch.Tensor([[four, one],[three, two]]).to(device),
            torch.Tensor([[four, one],[two, three]]).to(device),
            torch.Tensor([[four, two],[one, three]]).to(device),
            torch.Tensor([[four, three],[one, two]]).to(device),
            torch.Tensor([[four, three],[two, one]]).to(device),
            torch.Tensor([[four, two],[three, one]]).to(device),
        ]
        self.player2_payouts = [
            torch.Tensor([[four, three],[two, one]]).to(device),
            torch.Tensor([[four, two],[three, one]]).to(device),
            torch.Tensor([[four, one],[three, two]]).to(device),
            torch.Tensor([[four, one],[two, three]]).to(device),
            torch.Tensor([[four, two],[one, three]]).to(device),
            torch.Tensor([[four, three],[one, two]]).to(device),
            torch.Tensor([[three, four],[one, two]]).to(device),
            torch.Tensor([[two, four],[one, three]]).to(device),
            torch.Tensor([[one, four],[two, three]]).to(device),
            torch.Tensor([[one, four],[three, two]]).to(device),
            torch.Tensor([[two, four],[three, one]]).to(device),
            torch.Tensor([[three, four],[two, one]]).to(device),
        ]

        payout_mat_1 = self.player1_payouts[player1]
        payout_mat_2 = self.player1_payouts[player2]
        payout_mat_1 = payout_mat_1.reshape((1, 2, 2)).repeat(self.batch_size, 1, 1)
        payout_mat_2 = payout_mat_2.reshape((1, 2, 2)).repeat(self.batch_size, 1, 1)

        return [payout_mat_1, payout_mat_2]

    def infinitely_iterated_value(self, payout_mat_1, payout_mat_2, gamma_inner=0.96):
        dims = [5, 5]
        def Ls(th): # th is a list of two different tensors. First one is first agent? tnesor size is List[Tensor(bs, 5), Tensor(bs,5)].
            p_1_0 = torch.sigmoid(th[0][:, 0:1])
            p_2_0 = torch.sigmoid(th[1][:, 0:1])
            p = torch.cat([p_1_0*p_2_0, p_1_0*(1-p_2_0), (1-p_1_0)*p_2_0, (1-p_1_0)*(1-p_2_0)], dim=-1)
            p_1 = torch.reshape(torch.sigmoid(th[0][:, 1:5]), (self.batch_size, 4, 1))
            # p_2 = torch.reshape(torch.sigmoid(th[1][:, 1:5]), (bs, 4, 1))
            p_2 = torch.reshape(torch.sigmoid(
                    torch.cat([th[1][:,1:2], th[1][:,3:4], th[1][:,2:3], th[1][:,4:5]], dim=-1)
            ), (self.batch_size, 4, 1))
            P = torch.cat([p_1*p_2, p_1*(1-p_2), (1-p_1)*p_2, (1-p_1)*(1-p_2)], dim=-1)

            M = torch.matmul(p.unsqueeze(1), torch.inverse(torch.eye(4).to(device)-gamma_inner*P))
            L_1 = -torch.matmul(M, torch.reshape(payout_mat_1, (self.batch_size, 4, 1)))
            L_2 = -torch.matmul(M, torch.reshape(payout_mat_2, (self.batch_size, 4, 1)))
    #         return [L_1.squeeze(-1), L_2.squeeze(-1)]

            return [L_1.squeeze(-1), L_2.squeeze(-1), M]
        return dims, Ls

    def one_iteration_value(self, payout_mat_1, payout_mat_2):
        dims = [1, 1]
        def Ls(th):
            p_1, p_2 = torch.sigmoid(th[0]), torch.sigmoid(th[1])
            x, y = torch.cat([p_1, 1-p_1], dim=-1), torch.cat([p_2, 1-p_2], dim=-1)
            L_1 = -torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_1), y.unsqueeze(-1))
            L_2 = -torch.matmul(torch.matmul(x.unsqueeze(1), payout_mat_2), y.unsqueeze(-1))
            return [L_1.squeeze(-1), L_2.squeeze(-1), None]
        return dims, Ls

    def get_game(self, game, payoffs, infinite=True):
        assert payoffs[0] <= payoffs[1] <= payoffs[2] <= payoffs[3]

        [player1, player2] = self.games[game]
        [payout_mat_1, payout_mat_2] = self.create_payout(player1, player2, payoffs[0], payoffs[1], payoffs[2], payoffs[4])
        if infinite:
            return self.infinitely_iterated_value(payout_mat_1, payout_mat_2)
        else:
            return self.one_iteration_value(payout_mat_1, payout_mat_2)

    def sample_game(self, infinite=True):
        ind = torch.randint(12, size=(2, ))
        payout_range = [1, 10]
        # one = torch.randint(low=payout_range[0], high=payout_range[1], size=(1,))
        # two = torch.randint(low=one.item(), high=payout_range[1], size=(1,))
        # three = torch.randint(low=two.item(), high=payout_range[1], size=(1,))
        # four = torch.randint(low=three.item(), high=payout_range[1], size=(1,))
        one = -3.0
        two = -2.0
        three = -1.0
        four = 0.0
        [payout_mat_1, payout_mat_2] = self.create_payout(ind[0], ind[1], one, two, three, four)
        if infinite:
            return self.infinitely_iterated_value(payout_mat_1, payout_mat_2)
        else:
            return self.one_iteration_value(payout_mat_1, payout_mat_2)

    def sample_groups(self, infinite=True, groups=[]):
        main_dict = {}
        for group in groups:
            main_dict = {**main_dict, **self.all_groups[group]}
        keys = list(main_dict.keys())
        size = len(keys)
        ind = torch.randint(low=0, high=size, size=(1, ))
        game = keys[ind.item()]
        [ind_x, ind_y] = main_dict[game]
        one = -3.0
        two = -2.0
        three = -1.0
        four = 0.0
        [payout_mat_1, payout_mat_2] = self.create_payout(ind_x, ind_y, one, two, three, four)
        if infinite:
            return self.infinitely_iterated_value(payout_mat_1, payout_mat_2)
        else:
            return self.one_iteration_value(payout_mat_1, payout_mat_2)