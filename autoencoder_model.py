import torch
from torch import nn
from matplotlib import pyplot as plt

# contains MMAE_ALL, MMAE_ALL_DEEP, MMVAE_ALL_DEEP, MMAE_SINGLES

class MMAE_ALL(nn.Module):
    def __init__(self, in_feas_dim, latent_dim, modality_weights):

        super(MMAE_ALL, self).__init__()
        self.modality_weights = modality_weights
        self.in_feas = in_feas_dim
        self.latent = latent_dim

        # Create Encoder and decoder for each modality
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        for modality in self.in_feas:
            self.encoder_list.append(
                nn.Sequential(
                    nn.Linear(modality, self.latent),
                    nn.BatchNorm1d(self.latent),
                    nn.Sigmoid()
                )
            )
            self.decoder_list.append(
                nn.Sequential(
                    nn.Linear(self.latent, modality))
            )

        # Variable initialization
        for name, param in MMAE_ALL.named_parameters(self):
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if 'bias' in name:
                torch.nn.init.constant_(param, val=0)

    def forward(self, omics_list):
        # encode all omics
        encoded_omics_list = []
        for i, omics in enumerate(omics_list):
            encoded_omics_list.append(self.encoder_list[i](omics))

        # combine encoded into one latent dim with modality_weights
        for i, omics in enumerate(encoded_omics_list):
            if i == 0:
                latent_data = torch.mul(omics, self.modality_weights[i])
            else:
                latent_data = latent_data + torch.mul(omics, self.modality_weights[i])  # they share one latent_dim

        # decode each omics from latent dim
        decoded_omics_list = []
        for i in range(len(self.decoder_list)):
            decoded_omics_list.append(self.decoder_list[i](latent_data))

        return latent_data, decoded_omics_list

    def train_MMAE(self, train_loader, learning_rate=0.001, device=torch.device('cpu'), epochs=100, wandb=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        loss_ls = []
        for epoch in range(epochs):
            train_loss_sum = 0.0  # Record the loss of each epoch
            for (x, y) in train_loader:
                prev = 0
                omics_list = []
                for feas in self.in_feas:
                    omics_list.append(x[:, prev:prev + feas].to(device))
                    prev += feas

                latent_data, decoded_omics_list = self.forward(omics_list)
                loss = 0
                for i, decoded_omics in enumerate(decoded_omics_list):
                    loss += self.modality_weights[i] * loss_fn(decoded_omics, omics_list[i])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.sum().item()

            loss_ls.append(train_loss_sum)
            print('epoch: %d | loss: %.4f' % (epoch + 1, train_loss_sum))
            if wandb:
                wandb.log({'AE_loss': train_loss_sum})
            # save the model every 10 epochs, used for feature extraction
            if (epoch + 1) % 10 == 0:
                torch.save(self, 'model/AE/model_{}.pkl'.format(epoch + 1))

        # draw the training loss curve
        plt.plot([i + 1 for i in range(epochs)], loss_ls)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('result/AE_train_loss.png')


class MMAE_ALL_DEEP(nn.Module):
    def __init__(self, in_feas_dim, latent_dim, modality_weights):

        super(MMAE_ALL_DEEP, self).__init__()
        self.modality_weights = modality_weights
        self.in_feas = in_feas_dim
        self.latent = latent_dim

        # Create Encoder and decoder for each modality
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        for modality in self.in_feas:
            self.encoder_list.append(
                nn.Sequential(
                    nn.Linear(modality, self.latent * 2),
                    nn.BatchNorm1d(self.latent * 2),
                    nn.Linear(self.latent * 2, self.latent),
                    nn.BatchNorm1d(self.latent),
                    nn.Sigmoid()
                )
            )
            self.decoder_list.append(
                nn.Sequential(
                    nn.Linear(self.latent, modality))
            )

        # Variable initialization
        for name, param in MMAE_ALL_DEEP.named_parameters(self):
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if 'bias' in name:
                torch.nn.init.constant_(param, val=0)

    def forward(self, omics_list):
        # encode all omics
        encoded_omics_list = []
        for i, omics in enumerate(omics_list):
            encoded_omics_list.append(self.encoder_list[i](omics))

        # combine encoded into one latent dim with modality_weights
        for i, omics in enumerate(encoded_omics_list):
            if i == 0:
                latent_data = torch.mul(omics, self.modality_weights[i])
            else:
                latent_data = latent_data + torch.mul(omics, self.modality_weights[i])  # they share one latent_dim

        # decode each omics from latent dim
        decoded_omics_list = []
        for i in range(len(self.decoder_list)):
            decoded_omics_list.append(self.decoder_list[i](latent_data))

        return latent_data, encoded_omics_list, decoded_omics_list

    def train_MMAE(self, train_loader, learning_rate=0.001, device=torch.device('cpu'), epochs=100, wandb=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        loss_ls = []
        for epoch in range(epochs):
            train_loss_sum = 0.0  # Record the loss of each epoch
            for (x, y) in train_loader:
                prev = 0
                omics_list = []
                for feas in self.in_feas:
                    omics_list.append(x[:, prev:prev + feas].to(device))
                    prev += feas

                latent_data, _, decoded_omics_list = self.forward(omics_list)
                loss = 0
                for i, decoded_omics in enumerate(decoded_omics_list):
                    loss += self.modality_weights[i] * loss_fn(decoded_omics, omics_list[i])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.sum().item()

            loss_ls.append(train_loss_sum)
            print('epoch: %d | loss: %.4f' % (epoch + 1, train_loss_sum))
            if wandb:
                wandb.log({'AE_loss': train_loss_sum})
            # save the model every 10 epochs, used for feature extraction
            if (epoch + 1) % 10 == 0:
                torch.save(self, 'model/AE/model_{}.pkl'.format(epoch + 1))

        # draw the training loss curve
        plt.plot([i + 1 for i in range(epochs)], loss_ls)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('result/AE_train_loss.png')


class MMVAE_ALL_DEEP(nn.Module):
    def __init__(self, in_feas_dim, latent_dim, modality_weights):

        super(MMAE_ALL_DEEP, self).__init__()
        self.modality_weights = modality_weights
        self.in_feas = in_feas_dim
        self.latent = latent_dim

        # Create Encoder and decoder for each modality
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        for modality in self.in_feas:
            self.encoder_list.append(
                nn.Sequential(
                    nn.Linear(modality, self.latent * 2),
                    nn.BatchNorm1d(self.latent * 2),
                    nn.Linear(self.latent * 2, self.latent),
                    nn.BatchNorm1d(self.latent),
                    nn.Sigmoid()
                )
            )
            self.decoder_list.append(
                nn.Sequential(
                    nn.Linear(self.latent, modality))
            )

        # Variable initialization
        for name, param in MMAE_ALL_DEEP.named_parameters(self):
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if 'bias' in name:
                torch.nn.init.constant_(param, val=0)

    def forward(self, omics_list):
        # encode all omics
        encoded_omics_list = []
        for i, omics in enumerate(omics_list):
            encoded_omics_list.append(self.encoder_list[i](omics))

        # combine encoded into one latent dim with modality_weights
        for i, omics in enumerate(encoded_omics_list):
            if i == 0:
                latent_data = torch.mul(omics, self.modality_weights[i])
            else:
                latent_data = latent_data + torch.mul(omics, self.modality_weights[i])  # they share one latent_dim

        # decode each omics from latent dim
        decoded_omics_list = []
        for i in range(len(self.decoder_list)):
            decoded_omics_list.append(self.decoder_list[i](latent_data))

        return latent_data, encoded_omics_list, decoded_omics_list

    def train_MMAE(self, train_loader, learning_rate=0.001, device=torch.device('cpu'), epochs=100, wandb=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        loss_ls = []
        for epoch in range(epochs):
            train_loss_sum = 0.0  # Record the loss of each epoch
            for (x, y) in train_loader:
                prev = 0
                omics_list = []
                for feas in self.in_feas:
                    omics_list.append(x[:, prev:prev + feas].to(device))
                    prev += feas

                latent_data, _, decoded_omics_list = self.forward(omics_list)
                loss = 0
                for i, decoded_omics in enumerate(decoded_omics_list):
                    loss += self.modality_weights[i] * loss_fn(decoded_omics, omics_list[i])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.sum().item()

            loss_ls.append(train_loss_sum)
            print('epoch: %d | loss: %.4f' % (epoch + 1, train_loss_sum))
            if wandb:
                wandb.log({'AE_loss': train_loss_sum})
            # save the model every 10 epochs, used for feature extraction
            if (epoch + 1) % 10 == 0:
                torch.save(self, 'model/AE/model_{}.pkl'.format(epoch + 1))

        # draw the training loss curve
        plt.plot([i + 1 for i in range(epochs)], loss_ls)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('result/AE_train_loss.png')


class MMAE_SINGLES(nn.Module):
    def __init__(self, in_feas_dim, latent_dim, modality_weights):

        super(MMAE_SINGLES, self).__init__()
        self.modality_weights = modality_weights
        self.in_feas = in_feas_dim
        self.latent = latent_dim

        # Create Encoder and decoder for each modality
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        for modality in self.in_feas:
            self.encoder_list.append(
                nn.Sequential(
                    nn.Linear(modality, self.latent),  # maybe exchange modality with self feas
                    nn.BatchNorm1d(self.latent),
                    nn.Sigmoid()
                )
            )
            self.decoder_list.append(
                nn.Sequential(
                    nn.Linear(self.latent, modality))
            )

        # Variable initialization
        for name, param in MMAE_SINGLES.named_parameters(self):
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if 'bias' in name:
                torch.nn.init.constant_(param, val=0)

    def forward(self, omics_list):
        # encode all omics
        encoded_omics_list = []
        for i, omics in enumerate(omics_list):
            encoded_omics_list.append(self.encoder_list[i](omics))

        # combine encoded into one latent dim with modality_weights
        latent_dims_list = []
        for i, omics in enumerate(encoded_omics_list):
            latent_dims_list.append(omics)
        latent_data = [element.detach().cpu().numpy() for nestedlist in latent_dims_list for element in nestedlist]

        # decode each omics from latent dim
        decoded_omics_list = []
        for i in range(len(self.decoder_list)):
            decoded_omics_list.append(self.decoder_list[i](latent_dims_list[i]))

        return latent_data, decoded_omics_list

    def train_MMAE(self, train_loader, learning_rate=0.001, device=torch.device('cpu'), epochs=100, wandb=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        loss_ls = []
        for epoch in range(epochs):
            train_loss_sum = 0.0  # Record the loss of each epoch
            for (x, y) in train_loader:
                prev = 0
                omics_list = []
                for feas in self.in_feas:
                    omics_list.append(x[:, prev:prev + feas].to(device))
                    prev += feas

                latent_data, decoded_omics_list = self.forward(omics_list)
                loss = 0
                for i, decoded_omics in enumerate(decoded_omics_list):
                    loss += self.modality_weights[i] * loss_fn(decoded_omics, omics_list[i])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.sum().item()

            loss_ls.append(train_loss_sum)
            print('epoch: %d | loss: %.4f' % (epoch + 1, train_loss_sum))
            if wandb:
                wandb.log({'AE_loss': train_loss_sum})
            # save the model every 10 epochs, used for feature extraction
            if (epoch + 1) % 10 == 0:
                torch.save(self, 'model/AE/model_{}.pkl'.format(epoch + 1))

        # draw the training loss curve
        plt.plot([i + 1 for i in range(epochs)], loss_ls)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('result/AE_train_loss.png')